from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase


@dataclass(slots=True)
class EOSScoringResult:
    """
    Batched EOS signals aligned with realized generated sequences.
    Results are from raw next-token logprobs before any decoding-time processing.

    Attributes:
        eos_logprobs: Per-step EOS logprob trajectories.
            List of length M, where the i-th tensor has shape [T_i]. Entry t
            (zero-based) equals `log P(EOS | prompt, y_<t)`, where `y_<t` denotes the
            first t generated tokens, and `y_<0` is the empty prefix.
        eos_in_topk: Optional per-step EOS top-k membership trajectories.
            None if top_k is not provided. 
            Otherwise, list of length M, where the i-th boolean tensor has shape [T_i]. 
            Entry t is True if EOS is present in top-k of P(⋅| prompt, y_<t).
    """

    eos_logprobs: list[torch.Tensor]
    eos_in_topk: list[torch.Tensor] | None = None


@torch.inference_mode()
def score_eos_trajectories(
    prompt_token_ids: torch.Tensor,
    generated_token_ids: list[torch.Tensor],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
    top_k: int | None = None,
) -> EOSScoringResult:
    """
    Scores realized generations and derives EOS signals from rescored next-token logprobs.

    Args:
        prompt_token_ids: Exact token ids of the rendered generation prefix shared
            by all generated rollouts.
        generated_token_ids: List of length M of rollout token-id tensors. The i-th
            tensor has shape [T_i] and stores the generated token ids for rollout i.
        model: Causal language model used for rescoring.
        tokenizer: Tokenizer that provides EOS and padding token ids.
        batch_size: Number of trajectories to score in one forward pass.
        top_k: Optional top-k threshold for computing EOS membership from per-step
            next-token scores.

    Returns:
        `EOSScoringResult` containing EOS logprob trajectories and, when top_k
        is provided, EOS top-k membership trajectories from the same batched pass.

    Notes:
        Results are from raw next-token logprobs before any decoding-time processing.
    """
    device = model.device
    eos_token_ids_tensor = _get_eos_token_ids_tensor(model=model, tokenizer=tokenizer, device=device)

    sequence_eos_logprobs = [None] * len(generated_token_ids)
    sequence_eos_in_topk = [None] * len(generated_token_ids) if top_k else None

    for batch_generation_logprobs, batch_indices, batch_generated_lengths in _iter_rescored_logprobs(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    ):
        batch_eos_logprobs = torch.logsumexp(
            # [B, max(T_i), num_eos_tokens]
            batch_generation_logprobs.index_select(dim=-1, index=eos_token_ids_tensor),
            dim=-1,
        ).cpu() # [B, max(T_i)]

        batch_eos_in_topk = None
        if top_k:
             # [B, max(T_i), top_k]
            batch_topk_token_ids = torch.topk(batch_generation_logprobs, k=top_k, dim=-1).indices
            batch_eos_in_topk = (
                # [B, max(T_i), top_k, 1] == [1, 1, 1, num_eos] => [B, max(T_i), top_k, num_eos]
                batch_topk_token_ids.unsqueeze(-1) == eos_token_ids_tensor.view(1, 1, 1, -1)
            ).any(dim=-1).any(dim=-1).cpu() # [B, max(T_i)]

        for batch_idx, (original_idx, generated_length) in enumerate(
            zip(batch_indices, batch_generated_lengths, strict=True)
        ):
            sequence_eos_logprobs[original_idx] = batch_eos_logprobs[batch_idx, :generated_length]
            if batch_eos_in_topk is not None and sequence_eos_in_topk is not None:
                sequence_eos_in_topk[original_idx] = batch_eos_in_topk[batch_idx, :generated_length]

    return EOSScoringResult(
        eos_logprobs=[trajectory for trajectory in sequence_eos_logprobs],
        eos_in_topk=(
            [trajectory for trajectory in sequence_eos_in_topk]
            if sequence_eos_in_topk else None
        ),
    )


def _get_eos_token_ids_tensor(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> torch.Tensor:
    """Returns EOS token ids as a device-local tensor."""
    eos_token_id = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
    eos_token_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
    return torch.tensor(eos_token_ids, dtype=torch.long, device=device)


@torch.inference_mode()
def _iter_rescored_logprobs(
    prompt_token_ids: torch.Tensor,
    generated_token_ids: list[torch.Tensor],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 4,
) -> Iterator[tuple[torch.Tensor, list[int], list[int]]]:
    """
    Streams rescored next-token logprobs batch by batch for one prompt.

    Args:
        prompt_token_ids: Exact token ids of the rendered generation prefix used
            during rollout sampling.
        generated_token_ids: Realized generated token ids for each rollout.
        model: Causal language model used for rescoring.
        tokenizer: Tokenizer that provides prompt tokenization and padding ids.
        batch_size: Number of trajectories to score per forward pass.

    Yields:
        Tuples containing:
        - batch_generation_logprobs: Tensor of shape [B, max(T_i), V].
        - batch_indices: Original rollout indices for the current batch.
        - batch_generated_lengths: Realized generated lengths for the batch.
    """
    device = model.device
    prompt_ids = prompt_token_ids.to(device=device, dtype=torch.long)
    prompt_len = prompt_ids.numel()

    generated_token_ids = [
        token_ids.to(device=device, dtype=torch.long)
        for token_ids in generated_token_ids
    ]
    sorted_indices = sorted(
        range(len(generated_token_ids)),
        key=lambda idx: generated_token_ids[idx].numel(),
    )

    for start in range(0, len(sorted_indices), batch_size):
        batch_indices = sorted_indices[start : start + batch_size]
        batch_generated_token_ids = [generated_token_ids[idx] for idx in batch_indices]

        batch_full_sequences = [
            torch.cat((prompt_ids, token_ids), dim=0)
            for token_ids in batch_generated_token_ids
        ]
        batch_lengths = torch.tensor(
            [sequence.numel() for sequence in batch_full_sequences],
            dtype=torch.long,
            device=device,
        ) # [B]
        batch_generated_lengths = batch_lengths - prompt_len # [B]

        batch_input_ids = pad_sequence(
            batch_full_sequences,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ) # [B, S], where S = prompt_len + max(T_i) in batch
        attention_mask = (
            torch.arange(batch_input_ids.shape[1], device=device).unsqueeze(0)
            < batch_lengths.unsqueeze(1)
        ) #[1, S] < [B, 1] -> [B, S]

        # [B, S, V]
        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits
        batch_generation_logits = batch_logits[:, prompt_len - 1 : -1, :] # [B, max(T_i), V]
        batch_generation_logprobs = torch.log_softmax(batch_generation_logits, dim=-1)
        
        yield batch_generation_logprobs, batch_indices, batch_generated_lengths.tolist()
