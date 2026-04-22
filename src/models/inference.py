from typing import Literal
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput

from .decoding_policy import normalize_scores


def generate_step_scores(
    prompts: list[str],
    model: AutoModelForCausalLM | LLM,
    n_samples: int,
    max_new_tokens: int,
    backend: Literal["local", "vllm"],
    tokenizer: AutoTokenizer,
    temperature: float = 1.0,
    top_k: int | None = 0,
    top_p: float | None = 1.0,
    seed: int | None = 42,
    logprobs: int = -1,
    sample_batch_size: int = 8,
    device: str | torch.device | None = None,
    enable_thinking: bool | None = None,
    use_chat_template: bool = True,
) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]]:
    """
    Generates sampled completions and per-step candidate scores for one or more prompts.

    Local HF generation processes prompts sequentially. vLLM generation processes
    all provided prompts together.

    Args:
        prompts: List of prompt strings.
        model: Backend model instance.
        n_samples: Number of model completions to sample for each prompt (M).
        max_new_tokens: Maximum number of new generated tokens per completion.
        backend: Generation backend, either local or vllm.
        tokenizer: Tokenizer used to render and tokenize the generation prompt.
        temperature: Sampling temperature.
        top_k: Top-k truncation parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for sampling, incremented by sample batch position, 
            not by prompt position.
        logprobs: Number of candidate-token scores to retain per step.
        sample_batch_size: Number of completions to request per prompt in one
            generation call.
        device: Device override. Backend-specific default device is used if not set.
        enable_thinking: Optional chat-template hint for models that advertise
            reasoning-mode control through `enable_thinking`.
        use_chat_template: When False (for base models), return the prompt unchanged.

    Returns:
        List of per-prompt tuples preserving input prompt order: 
        each (generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids, prompt_token_ids)
        - generated_texts: List of M decoded strings.
        - sequence_step_scores: List of M tensors containing the normalized top-candidate log 
            probabilities for each generated step. For local backend, the i-th tensor has shape 
            [T_i, K], where `K == logprobs`.
            For vLLM backend, the i-th tensor has shape [T_i, K_i_max], with missing candidates
            padded by `-inf` per sequence.
        - sequence_lengths: Tensor of shape [M], where the i-th item is the length of the
            i-th generated sequence.
        - generated_token_ids: List of length M. The i-th tensor has shape [T_i]
            and stores the generated token ids aligned with the saved rollout.
        - prompt_token_ids: Tensor of shape [P] containing the token ids of the rendered 
            generation prompt.
    """
    rendered_prompts = [
        _render_generation_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            use_chat_template=use_chat_template,
        )
        for prompt in prompts
    ]

    if backend == "local":
        return [
            _generate_local_step_scores(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=logprobs,
                sample_batch_size=sample_batch_size,
                seed=seed,
                device=device,
            )
            for prompt in rendered_prompts
        ]

    elif backend == "vllm":
        return _generate_vllm_step_scores_batch(
            prompts=rendered_prompts,
            llm=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            logprobs=logprobs,
            sample_batch_size=sample_batch_size,
            device=device,
        )

    raise ValueError("backend must be one of {'local', 'vllm'}")


def _render_generation_prompt(
    prompt: str,
    tokenizer: AutoTokenizer,
    enable_thinking: bool | None,
    use_chat_template: bool,
) -> str:
    """
    Renders one user prompt into the string passed to generation.

    Args:
        prompt: Raw user prompt.
        tokenizer: Tokenizer whose chat template is used when available.
        enable_thinking: Optional chat-template hint for templates that support
            reasoning-mode control through `enable_thinking`.
        use_chat_template: When False (for base models), return the prompt unchanged.

    Returns:
        Rendered generation prompt string. If `use_chat_template` is False,
        returns `prompt` unchanged.

    Notes:
        This function only renders text; it does not tokenize.
    """
    if not use_chat_template:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    return tokenizer.apply_chat_template(messages, **template_kwargs)


@torch.inference_mode()
def _generate_vllm_step_scores_batch(
    prompts: list[str],
    llm: LLM,
    tokenizer: AutoTokenizer,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    logprobs: int,
    sample_batch_size: int,
    device: str | torch.device | None,
) -> list[tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]]:
    """
    Generates M sampled completions and per-step candidate scores with vLLM.

    Args:
        prompts: Rendered prompts (N).
        llm: Initialized vLLM engine.
        tokenizer: Tokenizer used to recover prompt token ids.
        n_samples: Number of completions to generate for each prompt (M).
        max_new_tokens: Maximum number of generated tokens per completion.
        temperature: Sampling temperature.
        top_k: Top-k truncation parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for sampling, incremented by sample batch position, 
            not by prompt position.
        logprobs: Number of candidate-token log probabilities to retain per step.
        sample_batch_size: Maximum number of completions per prompt requested from vLLM at once (B).
        device: Device used for returned tensors.

    Returns:
        List of per-prompt tuples preserving `prompts` order:
        each (generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids, prompt_token_ids)
        - generated_texts: List of M decoded strings.
        - sequence_step_scores: List of M tensors, where the i-th tensor has shape [T_i, K_i_max] 
            and contains the normalized top-candidate log probabilities for each generated step.
        - sequence_lengths: Tensor of shape [M], where the i-th item is the length of the i-th 
            generated sequence.
        - generated_token_ids: List of M tensors, where the i-th tensor has shape [T_i]
            and stores the generated token ids aligned with the saved rollout.
        - prompt_token_ids: Tensor of shape [P] containing the token ids of the rendered 
            generation prompt.

    Notes:
        T_i: Length of the i-th generated sequence.
        K_i_max: Maximum number of logprob candidates across all steps of the i-th sequence.
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )

    generated_texts_by_prompt = [[] for _ in prompts]
    sequence_step_scores_by_prompt = [[] for _ in prompts]
    sequence_lengths_chunks_by_prompt = [[] for _ in prompts]
    generated_token_ids_by_prompt = [[] for _ in prompts]
    
    vllm_top_k = -1 if top_k is None else top_k
    vllm_top_p = 1.0 if top_p is None else top_p

    for start in tqdm(range(0, n_samples, sample_batch_size), desc="Generating batched completions"):
        current_batch_size = min(sample_batch_size, n_samples - start)
        sampling_params = SamplingParams(
            n=current_batch_size,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=vllm_top_k,
            top_p=vllm_top_p,
            seed=None if seed is None else seed + start,
            logprobs=logprobs,
        )

        request_outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        ) # list[RequestOutput] of N

        for prompt_idx, request_output in enumerate(request_outputs):
            (
                generated_texts,
                sequence_step_scores,
                sequence_lengths,
                generated_token_ids,
            ) = _convert_vllm_completion_outputs(request_output.outputs, device=device)

            generated_texts_by_prompt[prompt_idx].extend(generated_texts)
            sequence_step_scores_by_prompt[prompt_idx].extend(sequence_step_scores)
            sequence_lengths_chunks_by_prompt[prompt_idx].append(sequence_lengths)
            generated_token_ids_by_prompt[prompt_idx].extend(generated_token_ids)

    prompt_results = []
    for prompt_idx, prompt in enumerate(prompts):
        prompt_token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device=device)
        prompt_results.append(
            (
                # list[str], length M
                generated_texts_by_prompt[prompt_idx],
                # list[Tensor], length M, i-th tensor has shape [T_i, K_i_max]
                sequence_step_scores_by_prompt[prompt_idx],
                # Tensor of shape [M]
                torch.cat(sequence_lengths_chunks_by_prompt[prompt_idx], dim=0),
                # list[Tensor], length M, i-th tensor has shape [T_i]
                generated_token_ids_by_prompt[prompt_idx],
                # Tensor of shape [P]
                prompt_token_ids,
            )
        )

    return prompt_results


def _convert_vllm_completion_outputs(
    completion_outputs: list[RequestOutput],
    device: torch.device,
) -> tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """
    Converts one prompt's vLLM completion outputs from a single request.

    Args:
        completion_outputs: list of vLLM RequestOutput objects for one prompt's generate request.
        device: Device used for returned tensors.

    Returns:
        (generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids): Tuple containing
        - generated_texts: List of decoded completion strings with length B.
        - sequence_step_scores: List of B tensors, where the j-th tensor has shape [T_j, K_j_max].
        - sequence_lengths: Tensor of shape [B] containing generated token lengths.
        - generated_token_ids: List of B tensors, where the j-th tensor has shape [T_j].
    
     Notes:
        T_j: Length of the j-th generated sequence.
        K_j_max: Maximum number of logprob candidates across all steps of the j-th sequence
    """
    generated_texts = [output.text for output in completion_outputs]
    sequence_lengths = torch.tensor(
        [len(output.token_ids) for output in completion_outputs],
        dtype=torch.long,
        device=device,
    )
    sequence_step_scores = [
        normalize_scores(_pad_vllm_step_scores(output.logprobs, device=device))
        for output in completion_outputs
    ]
    generated_token_ids = [
        torch.tensor(output.token_ids, dtype=torch.long, device=device)
        for output in completion_outputs
    ]
    return generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids


def _pad_vllm_step_scores(
    step_logprobs_sequence: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Pads the i-th sequence's step logprobs to shape [T_i, K_i_max] with `-inf`.
    
    Args:
        step_logprobs_sequence: List of length T_i, where each item is a list of logprobs for that step.
    
    Returns:
        padded_scores: Tensor of shape [T_i, K_i_max].

    Notes: 
        This function processes one sequence (i-th) at a time.
        T_i: Length of the i-th sequence.
        K_i_max: Maximum number of logprob candidates across all steps of the i-th sequence.
    """
    num_steps = len(step_logprobs_sequence)
    step_logprob_len = [len(step_logprobs) for step_logprobs in step_logprobs_sequence]
    max_candidates = max(step_logprob_len)

    use_cuda = device.type == "cuda"
    # [T_i, max_logprob_len]
    padded_scores = torch.full(
        (num_steps, max_candidates),
        -torch.inf,
        dtype=torch.float32,
        pin_memory=use_cuda,
    )

    for step_idx, step_logprobs in enumerate(step_logprobs_sequence):
        step_len = step_logprob_len[step_idx]
        padded_scores[step_idx, :step_len] = torch.tensor(
            [item.logprob for item in step_logprobs.values()],
            dtype=torch.float32,
        )

    return padded_scores.to(device, non_blocking=True)


@torch.inference_mode()
def _generate_local_step_scores(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    logprobs: int,
    sample_batch_size: int,
    device: str | torch.device | None,
) -> tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """
    Generates M sampled completions and per-step candidate scores with a local HF model.

    Args:
        prompt: Rendered prompt.
        model: Initialized causal LM.
        tokenizer: Tokenizer for the causal LM.
        n_samples: Number of completions to generate for the prompt (M).
        max_new_tokens: Maximum number of generated tokens per completion.
        temperature: Sampling temperature.
        top_k: Top-k truncation parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for sampling, incremented by sample batch position.
        logprobs: Number of top candidate log probabilities to retain per step.
        sample_batch_size: Maximum number of completions generated in one model call.
        device: Device used for generation and returned tensors.

    Returns:
        (generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids, prompt_token_ids):
        - generated_texts: List of M decoded completion strings.
        - sequence_step_logprobs: List of M tensors. The i-th tensor has shape [T_i, K], 
            where `K == logprobs`, and contains the normalized top-candidate log probabilities 
            for each generated step.
        - sequence_lengths: Tensor of shape [M], where the i-th item is the length of the 
            i-th generated sequence.
        - generated_token_ids: List of M tensors. The i-th tensor has shape [T_i] and contains 
            the generated token ids up to the first EOS token (inclusive when present).
        - prompt_token_ids: Tensor of shape [P] containing the rendered prompt token ids shared 
            by all completions.

    Notes:
        T_i: Length of the i-th generated sequence.
    """
    if device is None:
        device = model.device
    device = torch.device(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    eos_token_id = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)

    generated_texts = []
    sequence_step_logprobs = []
    sequence_lengths_chunks = []
    generated_token_ids = []

    for start in range(0, n_samples, sample_batch_size):
        current_batch_size = min(sample_batch_size, n_samples - start)
        if seed is not None:
            torch.manual_seed(seed + start)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + start)

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_k=0 if top_k is None else top_k,
            top_p=1.0 if top_p is None else top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=current_batch_size,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = outputs.sequences[:, prompt_len:]
        cur_sequence_lengths = _sequence_lengths_from_generated_ids(
            generated_ids,
            eos_token_id=eos_token_id,
        ) # [M]

        # [T_max, B, V] => [B, T_max, V]
        all_scores = torch.stack(outputs.scores, dim=0).transpose(0, 1)
        all_logprobs = torch.log_softmax(all_scores, dim=-1)

        # [B, T_max, K]
        top_vals, _ = torch.topk(all_logprobs, k=logprobs, dim=-1)

        lengths = cur_sequence_lengths.tolist()
        sequence_step_logprobs.extend(
            normalize_scores(top_vals[i, :length])
            for i, length in enumerate(lengths)
        )
        generated_token_ids.extend(
            generated_ids[i, :length].clone()
            for i, length in enumerate(lengths)
        )
        generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        sequence_lengths_chunks.append(cur_sequence_lengths)

    sequence_lengths = torch.cat(sequence_lengths_chunks, dim=0)
    prompt_token_ids = inputs["input_ids"][0].clone()
    return generated_texts, sequence_step_logprobs, sequence_lengths, generated_token_ids, prompt_token_ids


def _sequence_lengths_from_generated_ids(
    generated_ids: torch.Tensor,
    eos_token_id: int | list[int],
) -> torch.Tensor:
    """
    Computes the sequence lengths of model generations.
    
    Args:
        generated_ids: Token ids of shape [M, T].
        eos_token_id: EOS token id or list of EOS token ids.
    
    Returns:
        sequence_lengths: Tensor of shape [M], 
            where the i-th item is the length of the i-th generated sequence.

    Notes: 
        Length includes the first EOS token when EOS is present.
    """
    _, max_steps = generated_ids.shape

    eos_ids = (
        torch.tensor([eos_token_id], device=generated_ids.device, dtype=generated_ids.dtype)
        if isinstance(eos_token_id, int)
        else torch.tensor(eos_token_id, device=generated_ids.device, dtype=generated_ids.dtype)
    )

    # [M, T, 1] == [eos_num] => [M, T, eos_num] => [M, T]
    eos_mask = (generated_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
    has_eos = eos_mask.any(dim=-1) # [M]
    first_eos_pos = eos_mask.to(dtype=torch.long).argmax(dim=-1) # [M], index of the first eos (True => 1)
    return torch.where(
        has_eos,
        first_eos_pos + 1,
        torch.full_like(first_eos_pos, max_steps), # [M]
    )
