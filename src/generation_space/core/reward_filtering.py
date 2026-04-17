import math

import torch


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


def filter_rollouts_by_mask(
    rollout_tensors: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Applies a Boolean mask to a list of per-rollout tensors.

    Args:
        rollout_tensors: List of M per-rollout tensors.
        sequence_lengths: Tensor [M] with generated lengths per rollout.
        keep_mask: Boolean tensor [M] indicating retained rollouts.

    Returns:
        (kept_rollout_tensors, kept_lengths): Tuple in original rollout order
        - kept_rollout_tensors: List of per-rollout tensors for retained rollouts.
        - kept_lengths: Tensor of generated lengths for retained rollouts.
    """
    kept_rollout_tensors = [
        rollout_tensor
        for rollout_tensor, keep in zip(rollout_tensors, keep_mask.tolist(), strict=True)
        if keep
    ]
    kept_lengths = sequence_lengths[keep_mask]

    return kept_rollout_tensors, kept_lengths
