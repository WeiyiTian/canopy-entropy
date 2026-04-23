import torch

from src.metrics import calculate_prompt_metrics
from src.metrics.generation_tree import sequence_entropy_from_step_entropy

from .reward_filtering import filter_rollouts_by_mask
from .structures import PromptRolloutStats


@torch.inference_mode()
def compute_prompt_rollout_stats(
    prompt: str,
    step_conditional_entropy: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    *,
    normalized_embeddings: torch.Tensor | None,
    reward_scores: torch.Tensor,
    keep_mask: torch.Tensor,
    max_new_tokens: int,
) -> PromptRolloutStats:
    """
    Computes generation space metrics from M generated rollouts for one prompt.

    Args:
        prompt: Single input prompt string shared by the sampled generations.
        step_conditional_entropy: List of length M. The i-th tensor has shape
            [T_i] containing per-step entropies `H(Y^(i)_t | X, y^(i)_<t)`,
            where Y_t ∈ V including EOS.
        sequence_lengths: Tensor of shape [M], where i-th entry is generated
            token length `N^(i)`.
        normalized_embeddings: Tensor [M, D] of L2-normalized embeddings
            for the M generated responses, aligned with raw rollout order, or
            `None` to skip embedding-based metrics.
        reward_scores: Tensor [M] of reward scores aligned with raw rollout order.
        keep_mask: Boolean tensor [M] indicating retained rollouts.
        max_new_tokens: Generation length cap `T_max` used during sampling.

    Returns:
        `PromptRolloutStats` containing the reward scores, retained rollout
        mask, kept entropy tensors and lengths, and raw/kept prompt-level
        metrics.
    """
    sequence_lengths = sequence_lengths.to(dtype=torch.float32)

    raw_sequence_conditional_entropy = sequence_entropy_from_step_entropy(step_conditional_entropy)
    raw_metrics = calculate_prompt_metrics(
        sequence_entropy=raw_sequence_conditional_entropy,
        sequence_lengths=sequence_lengths,
        normalized_embeddings=normalized_embeddings,
        max_new_tokens=max_new_tokens,
    )

    kept_step_conditional_entropy, kept_lengths = filter_rollouts_by_mask(
        rollout_tensors=step_conditional_entropy,
        sequence_lengths=sequence_lengths,
        keep_mask=keep_mask,
    )

    kept_normalized_embeddings = (
        normalized_embeddings[keep_mask] if normalized_embeddings is not None else None
    )
    kept_sequence_conditional_entropy = sequence_entropy_from_step_entropy(kept_step_conditional_entropy)
    kept_metrics = calculate_prompt_metrics(
        sequence_entropy=kept_sequence_conditional_entropy,
        sequence_lengths=kept_lengths,
        normalized_embeddings=kept_normalized_embeddings,
        max_new_tokens=max_new_tokens,
    )

    return PromptRolloutStats(
        prompt=prompt,
        raw_sequence_lengths=sequence_lengths,
        reward_scores=reward_scores,
        keep_mask=keep_mask,
        raw_metrics=raw_metrics,
        raw_sequence_conditional_entropy=raw_sequence_conditional_entropy,
        kept_sequence_lengths=kept_lengths,
        kept_sequence_conditional_entropy=kept_sequence_conditional_entropy,
        kept_metrics=kept_metrics,
    )
