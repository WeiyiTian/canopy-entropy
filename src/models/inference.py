from typing import Literal
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from .decoding_policy import normalize_scores


def generate_step_scores(
    prompt: str,
    model: AutoModelForCausalLM | LLM,
    n_samples: int,
    max_new_tokens: int,
    backend: Literal["local", "vllm"],
    tokenizer: AutoTokenizer | None = None,
    temperature: float = 1.0,
    top_k: int | None = 0,
    top_p: float | None = 1.0,
    seed: int | None = 42,
    logprobs: int = -1,
    sample_batch_size: int = 8,
    device: str | torch.device | None = None,
    enable_thinking: bool | None = None,
) -> tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """
    Generates sampled completions and per-step candidate scores for one prompt,
    where the scores are post-processed and renormalized.

    Args:
        prompt: Single input prompt string used for sampled generations.
        model: Backend model instance. Use `AutoModelForCausalLM` with local backend
            and `LLM` with vllm backend.
        n_samples: Number of model completions to sample for the input prompt (M).
        max_new_tokens: Maximum number of new generated tokens per completion.
        backend: Generation backend, either local or vllm.
        tokenizer: Tokenizer required only for the local backend.
        temperature: Sampling temperature.
        top_k: Top-k truncation parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for sampling. For vLLM, seed is further incremented by batch index.
        logprobs: Number of candidate-token scores to retain per step.
        sample_batch_size: Batch size used only for vLLM sampling loops.
        device: Device override. Backend-specific default device is used if not set.
        enable_thinking: Optional chat-template hint for models that advertise
            reasoning-mode control through `enable_thinking`.

    Returns:
        (generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids): Tuple that contains
        - generated_texts: List of M decoded strings.
        - sequence_step_scores: List of length M. 
            For local backend, the i-th tensor has shape [T_i, K], where `K == logprobs`.
            For vLLM backend, the i-th tensor has shape [T_i, K_i_max], 
                with missing candidates padded by `-inf` per sequence.
        - sequence_lengths: Tensor of shape [M], where the i-th item is the length of the i-th generated sequence.
        - generated_token_ids: List of length M. The i-th tensor has shape [T_i]
            and stores the generated token ids aligned with the saved rollout.
    """

    rendered_prompt = _render_generation_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )

    if backend == "local":
        if tokenizer is None:
            raise ValueError("tokenizer is required when backend='local'")
        return _generate_local_step_scores(
            prompt=rendered_prompt,
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            logprobs=logprobs,
            seed=seed,
            device=device,
        )

    elif backend == "vllm":
        return _generate_vllm_step_scores(
            prompt=rendered_prompt,
            llm=model,
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
) -> str:
    """
    Renders a chat prompt when the tokenizer exposes a chat template.
    Thinking control is only applied when the current chat template supports `enable_thinking`.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if enable_thinking is not None and "enable_thinking" in chat_template:
        template_kwargs["enable_thinking"] = enable_thinking

    return tokenizer.apply_chat_template(messages, **template_kwargs)


@torch.inference_mode()
def _generate_vllm_step_scores(
    prompt: str,
    llm: LLM,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    logprobs: int,
    sample_batch_size: int,
    device: torch.device | None,
) -> tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """
    Generates M = `n_samples` completions from a vLLM model and returns:
        generated_texts: List of M decoded strings.
        sequence_step_scores: List of M tensors, where the i-th tensor has shape [T_i, K_i_max] 
            and contains per-step vocabulary scores that are processed and renormalized.
        sequence_lengths: Tensor of shape [M], where the i-th item is the length of the i-th generated sequence.
        generated_token_ids: List of M tensors, where the i-th tensor has shape [T_i]
            and contains the generated token ids returned by vLLM.

    Notes:
        T_i: the length of the i-th generated sequence.
        K_i_max: Maximum number of logprob candidates across all steps of the i-th sequence.
        Seed is incremented by batch index.
    """
    completion_outputs, generated_texts = _generate_vllm_completions(
        prompt=prompt,
        llm=llm,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        logprobs=logprobs,
        sample_batch_size=sample_batch_size,
    )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )

    sequence_lengths = torch.tensor(
        [len(output.token_ids) for output in completion_outputs],
        dtype=torch.long,
        device=device,
    )

    generated_token_ids = [
        torch.tensor(output.token_ids, dtype=torch.long, device=device)
        for output in completion_outputs
    ]

    sequence_step_scores = [
        normalize_scores(_pad_vllm_step_scores(output.logprobs, device=device))
        for output in completion_outputs
    ]

    return generated_texts, sequence_step_scores, sequence_lengths, generated_token_ids


def _generate_vllm_completions(
    prompt: str,
    llm: LLM,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    logprobs: int,
    sample_batch_size: int,
) -> tuple[list, list[str]]:
    """
    Runs vLLM batched generation and returns a list of raw completion objects and a list of decoded texts.
    """
    completion_outputs = []
    generated_texts = []
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

        request_output = llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0]
        
        batch_outputs = request_output.outputs
        completion_outputs.extend(batch_outputs)
        generated_texts.extend(output.text for output in batch_outputs)

    return completion_outputs, generated_texts


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
    device: str | torch.device | None,
) -> tuple[list[str], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """
    Generates M = `n_samples` completions from a local model for one prompt and returns:
        generated_texts: List of M decoded strings.
        sequence_step_logprobs: List of M tensors, where the i-th tensor has shape [T_i, V] 
            and contains per-step vocabulary log probability scores that are processed and renormalized.
        sequence_lengths: Tensor of shape [M], where the i-th item is the length of the i-th generated sequence.
        generated_token_ids: List of M tensors, where the i-th tensor has shape [T_i]
            and contains the exact generated token ids up to the first EOS (inclusive when present).

    Notes:
        T_i: the length of the i-th generated sequence.
        T_max: the length of the longest generated sequence among the M samples.
        K == `logprobs`: number of logits to keep per step.
    """

    if device is None:
        device = model.device
    device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_k=0 if top_k is None else top_k,
        top_p=1.0 if top_p is None else top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_samples,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, prompt_len:]
    eos_token_id = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
    sequence_lengths = _sequence_lengths_from_generated_ids(generated_ids, eos_token_id=eos_token_id) # [M]

    # [T_max, M, V] => [M, T_max, V]
    all_scores = torch.stack(outputs.scores, dim=0).transpose(0, 1)
    all_logprobs = torch.log_softmax(all_scores, dim=-1)

    # [M, T_max, K]
    top_vals, _ = torch.topk(all_logprobs, k=logprobs, dim=-1)

    lengths = sequence_lengths.tolist()
    # M x [T_i, K]
    sequence_step_logprobs = [normalize_scores(top_vals[i, :l]) for i, l in enumerate(lengths)]
    generated_token_ids = [generated_ids[i, :l].clone() for i, l in enumerate(lengths)]
    
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts, sequence_step_logprobs, sequence_lengths, generated_token_ids


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
