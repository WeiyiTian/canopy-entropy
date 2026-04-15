import math

import torch

from .structures import PromptRollouts


def build_reward_filter_mask(
    reward_scores: torch.Tensor,
    keep_fraction: float,
) -> torch.Tensor:
    """
    Selects the highest-reward rollouts.

    Args:
        reward_scores: Tensor [M] of reward scores aligned with raw rollout order.
        keep_fraction: Fraction of highest-reward rollouts to retain.

    Returns:
        Boolean tensor [M] with `True` for retained rollouts.
    """
    keep_count = math.ceil(keep_fraction * int(reward_scores.numel()))
    keep_indices = torch.topk(reward_scores, k=keep_count).indices
    keep_mask = torch.zeros_like(reward_scores, dtype=torch.bool)
    keep_mask[keep_indices] = True
    return keep_mask


def apply_filter_mask(
    sequence_step_logprobs: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Applies a Boolean mask to rollout tensors.

    Args:
        sequence_step_logprobs: List of M step logprob tensors.
        sequence_lengths: Tensor [M] with generated lengths per rollout.
        keep_mask: Boolean tensor [M] indicating retained rollouts.

    Returns:
        (kept_step_logprobs, kept_lengths): Tuple in original rollout order
        - kept_step_logprobs: List of step logprob tensors for retained rollouts.
        - kept_lengths: Tensor of generated lengths for retained rollouts.
    """
    kept_step_logprobs = [
        logprobs
        for logprobs, keep in zip(sequence_step_logprobs, keep_mask.tolist(), strict=True)
        if keep
    ]
    kept_lengths = sequence_lengths[keep_mask]

    return kept_step_logprobs, kept_lengths


def default_reward_filter(
    rollouts: PromptRollouts,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns zero reward scores and a mask that retains all rollouts.
    Used when no reward model is configured.

    Args:
        rollouts: Rollout statistics for a single prompt with M sampled sequences.

    Returns:
        (reward_scores, keep_mask):
        - reward_scores: Tensor [M] of zeros on the rollouts' device.
        - keep_mask: Boolean tensor [M] of `True` on the rollouts' device.
    """
    num_samples = len(rollouts.generated_texts)
    device = rollouts.sequence_lengths.device
    reward_scores = torch.zeros(num_samples, dtype=torch.float32, device=device)
    keep_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
    return reward_scores, keep_mask
