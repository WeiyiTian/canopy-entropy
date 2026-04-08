import math
from dataclasses import dataclass

import torch

from ...models.reward_pipeline import SkyworkRewardPipeline


@dataclass(slots=True)
class FilteredRollouts:
    """
    Reward-scored prompt rollouts with explicit retained indices.

    Attributes:
        generated_texts: Raw rollout texts in original rollout order.
        sequence_step_scores: Token-level score tensors aligned with `generated_texts`.
        sequence_lengths: Generated token lengths aligned with `generated_texts`.
        reward_scores: Reward scores aligned with the raw rollout order.
        keep_indices: Indices of retained rollouts in original rollout order.
    """

    generated_texts: list[str]
    sequence_step_scores: list[torch.Tensor]
    sequence_lengths: torch.Tensor
    reward_scores: torch.Tensor
    keep_indices: torch.Tensor


def filter_rollouts_by_reward(
    prompt: str,
    generated_texts: list[str],
    sequence_step_scores: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    reward_model: SkyworkRewardPipeline | None,
    reward_keep_fraction: float,
) -> FilteredRollouts:
    """
    Scores one prompt's rollouts and returns the retained rollout indices.

    Args:
        prompt: Source prompt shared by all sampled rollouts.
        generated_texts: List of M rollout texts for the prompt.
        sequence_step_scores: List of M token-level score tensors aligned with `generated_texts`.
        sequence_lengths: Tensor [M] with generated lengths per rollout.
        reward_model: Optional reward model used to score each rollout. If
            None, all rollouts are kept and assigned zero reward.
        reward_keep_fraction: Fraction of highest-reward rollouts to retain when
            reward_model is provided.

    Returns:
        `FilteredRollouts` containing the raw prompt rollouts, per-rollout
        reward scores, and retained rollout indices sorted in rollout order.
    """
    device = sequence_lengths.device

    if not reward_model:
        reward_scores = torch.zeros(len(generated_texts), dtype=torch.float32, device=device)
        keep_indices = torch.arange(len(generated_texts), dtype=torch.long, device=device)
    else:
        reward_scores = reward_model.score_batch(prompt, generated_texts).to(device=device)
        keep_count = math.ceil(reward_keep_fraction * len(reward_scores))
        keep_indices = torch.topk(reward_scores, k=keep_count).indices.sort().values

    return FilteredRollouts(
        generated_texts=generated_texts,
        sequence_step_scores=sequence_step_scores,
        sequence_lengths=sequence_lengths,
        reward_scores=reward_scores,
        keep_indices=keep_indices,
    )
