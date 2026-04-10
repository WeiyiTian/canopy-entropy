from dataclasses import dataclass
import math

import torch

from ..models import SkyworkRewardPipeline


@dataclass(slots=True)
class RewardFilterResult:
    """
    Reward filtering outputs for one prompt's rollouts.

    Attributes:
        reward_scores: Reward scores aligned with the raw rollout order.
        kept_mask: Boolean mask indicating which rollouts were retained.
        retained_step_logprobs: Token-level score tensors for retained rollouts in
            original rollout order.
        retained_lengths: Retained rollout lengths in original rollout order.
    """

    reward_scores: torch.Tensor
    kept_mask: torch.Tensor
    retained_step_logprobs: list[torch.Tensor]
    retained_lengths: torch.Tensor


def filter_rollouts_by_reward(
    prompt: str,
    generated_texts: list[str],
    sequence_step_logprobs: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    reward_model: SkyworkRewardPipeline | None,
    reward_keep_fraction: float,
) -> RewardFilterResult:
    """
    Scores one prompt's rollouts and returns the retained rollout view.

    Args:
        prompt: Source prompt shared by all sampled rollouts.
        generated_texts: List of M rollout texts for the prompt.
        sequence_step_logprobs: List of M token-level score tensors aligned with `generated_texts`.
        sequence_lengths: Tensor [M] with generated lengths per rollout.
        reward_model: Optional reward model used to score each rollout. If
            None, all rollouts are kept and assigned zero reward.
        reward_keep_fraction: Fraction of highest-reward rollouts to retain when
            reward_model is provided.

    Returns:
        `RewardFilterResult` containing per-rollout reward scores, the retention
        mask, and retained rollout tensors in original rollout order.
    """

    device = sequence_lengths.device
    rollout_count = len(generated_texts)

    if reward_model is None:
        reward_scores = torch.zeros(rollout_count, dtype=torch.float32, device=device)
        kept_mask = torch.ones(rollout_count, dtype=torch.bool, device=device)
    else:
        reward_scores = reward_model.score_batch(prompt, generated_texts).to(device=device)
        keep_count = math.ceil(reward_keep_fraction * rollout_count)
        keep_indices = torch.topk(reward_scores, k=keep_count).indices
        kept_mask = torch.zeros(rollout_count, dtype=torch.bool, device=device)
        kept_mask[keep_indices] = True

    retained_step_logprobs = [
        logprobs
        for logprobs, keep in zip(sequence_step_logprobs, kept_mask.tolist(), strict=True)
        if keep
    ]
    retained_lengths = sequence_lengths[kept_mask]

    return RewardFilterResult(
        reward_scores=reward_scores,
        kept_mask=kept_mask,
        retained_step_logprobs=retained_step_logprobs,
        retained_lengths=retained_lengths,
    )
