import math
import torch

from ...models.reward_pipeline import SkyworkRewardPipeline
from .generation_space_rollups import FilteredRollouts, RawRollouts


def filter_rollouts_by_reward(
    prompt: str,
    generated_texts: list[str],
    sequence_step_scores: list[torch.Tensor],
    sequence_lengths: torch.Tensor,
    reward_model: SkyworkRewardPipeline | None,
    reward_keep_fraction: float,
) -> FilteredRollouts:
    """
    Scores one prompt's rollouts and return both raw and retained rollout data.

    Args:
        prompt: Source prompt shared by all sampled rollouts.
        generated_texts: List of `M` rollout texts for the prompt.
        sequence_step_scores: List of `M` token-level score tensors aligned with
            `generated_texts`.
        sequence_lengths: Tensor `[M]` with generated lengths per rollout.
        reward_model: Optional reward model used to score each rollout. If
            `None`, all rollouts are kept and assigned zero reward.
        reward_keep_fraction: Fraction of highest-reward rollouts to retain when
            `reward_model` is provided.

    Returns:
        `FilteredRollouts` containing the raw rollouts, per-rollout reward
        scores, and a boolean mask indicating which rollouts are retained.
    """
    device = sequence_lengths.device

    if reward_model is None:
        reward_scores = torch.zeros(len(generated_texts), dtype=torch.float32, device=device)
        kept_mask = torch.ones(len(generated_texts), dtype=torch.bool, device=device)
    else:
        reward_scores = reward_model.score_batch(prompt, generated_texts).to(device=device)
        keep_count = math.ceil(reward_keep_fraction * len(reward_scores))
        kept_mask = torch.zeros(len(generated_texts), dtype=torch.bool, device=device)
        kept_mask[torch.topk(reward_scores, k=keep_count).indices] = True

    return FilteredRollouts(
        raw_rollouts=RawRollouts(
            generated_texts=generated_texts,
            sequence_step_scores=sequence_step_scores,
            sequence_lengths=sequence_lengths,
        ),
        reward_scores=reward_scores,
        kept_mask=kept_mask,
    )
