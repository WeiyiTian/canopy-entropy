import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


@torch.no_grad()
def aggregate_running_entropy_rate(
    step_conditional_entropy: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates cumulative entropy rate summed over all active rollouts at
    specified positions.

    For rollout i of length T_i, the rate at 1-based position t is
    `sum(i-th step_entropy[:t]) / t`. At each evaluated position, only rollouts
    with `T_i >= t` contribute to the returned sum and count.

    Args:
        step_conditional_entropy: List of M tensors; tensor i has shape [T_i]
            with per-step entropies for rollout i.
        sequence_lengths: Tensor of shape [M] containing each rollout length T_i.
        positions: Tensor of shape [B] containing 1-based token positions to evaluate.

    Returns:
        (entropy_rate_sum, active_rollout_count):
        - entropy_rate_sum: Tensor of shape [B]; b-th entry is the entropy rates 
            summed over active rollouts at positions[b].
        - active_rollout_count: Tensor of shape [B]; b-th entry is the number
            of rollouts with `T_i >= positions[b]`, i.e, still active at position b.
    """
    per_rollout_entropy = [
        entropy.to(torch.float64) for entropy in step_conditional_entropy
    ]
    padded_step_entropy = pad_sequence(
        per_rollout_entropy,
        batch_first=True,
        padding_value=0.0,
    ) # [M, T_max]

    # Leading zero so positions index directly as 1-based token counts.
    prefix_entropy = F.pad(
        padded_step_entropy.cumsum(dim=1), # [M, T_max]
        pad=(1, 0),
        value=0.0,
    ) # pad zero to the left: [M, T_max + 1]

    # clamp positions to stay <= T_max
    clamped_position_indices = positions.clamp_max(prefix_entropy.shape[1] - 1)
    prefix_entropy_at_positions = prefix_entropy.index_select(
        dim=1,
        index=clamped_position_indices, # [B]
    ) # [M, B]

    # [1, B] <= [M, 1] => [M, B]
    active_rollout_mask = positions[None, :] <= sequence_lengths[:, None]  # [M, B]
    # [M, B] / [1, B] => [M, B]
    entropy_rate_at_positions = prefix_entropy_at_positions / positions[None, :]
    # [M, B] => [B]
    entropy_rate_sum = (entropy_rate_at_positions * active_rollout_mask).sum(dim=0)
    # [M, B] => [B]
    active_rollout_count = active_rollout_mask.sum(dim=0)
    return entropy_rate_sum, active_rollout_count
