import torch

from src.metrics import (
    PromptMetrics,
    aggregate_prompt_controlled_correlation,
    calculate_tree_rollout_metrics,
    pool_bucketed_semantic_diversity,
)

from .structures import PromptRolloutStats


def compute_pooled_metrics(
    prompt_results: list[PromptRolloutStats],
) -> dict[str, dict[str, object]]:
    """
    Computes cross-prompt generation space metrics for raw and reward-filtered rollouts.

    Args:
        prompt_results: List of P `PromptRolloutStats` containing per-prompt stats and 
        metrics for both raw and reward-filtered rollouts.

    Returns:
        Dictionary with `"raw"` and `"kept"` entries. Each entry is a metric
        dictionary keyed by `GENERATION_SPACE_METRIC_KEYS`. Entries are scalar tensors,
        except `semantic_diversity_bucketed_mean`, which is a mapping from length
        bucket names to scalar tensors.
    
    Notes:
        Correlation and covariance metrics without the `_pooled` suffix are
        prompt-controlled aggregates of the per-prompt metrics. Other metrics are
        recomputed on all rollouts concatenated into one tensor.
    """
    return {
        "raw": _pool_per_prompt_metrics(
            per_prompt_sequence_lengths=[pr.raw_sequence_lengths for pr in prompt_results],
            per_prompt_sequence_entropy=[pr.raw_sequence_conditional_entropy for pr in prompt_results],
            per_prompt_metrics=[pr.raw_metrics for pr in prompt_results],
        ),
        "kept": _pool_per_prompt_metrics(
            per_prompt_sequence_lengths=[pr.kept_sequence_lengths for pr in prompt_results],
            per_prompt_sequence_entropy=[pr.kept_sequence_conditional_entropy for pr in prompt_results],
            per_prompt_metrics=[pr.kept_metrics for pr in prompt_results],
        ),
    }


def _pool_per_prompt_metrics(
    per_prompt_sequence_lengths: list[torch.Tensor],
    per_prompt_sequence_entropy: list[torch.Tensor],
    per_prompt_metrics: list[PromptMetrics],
) -> dict[str, object]:
    """
    Aggregates one aligned set of per-prompt rollout metrics across prompts
    into a generation-space metric dict.

    Args:
        per_prompt_sequence_lengths: List of length P. The p-th tensor has shape
            [n_p] with rollout lengths `N^(p,i)` for prompt p.
        per_prompt_sequence_entropy: List of length P. The p-th tensor has shape
            [n_p] with per-rollout summed step conditional entropies as 
            `H^(p,i)_sum = sum_t H(Y_t^(p,i) | X^p, y_<t^(p,i))`.
        per_prompt_metrics: List of P `PromptMetrics`. The p-th entry contains
            per-prompt metrics computed from prompt p's n_p rollouts.
        
    Returns:
        Metric dictionary keyed by `GENERATION_SPACE_METRIC_KEYS`. Entries are 
        scalar tensors, except `semantic_diversity_bucketed_mean`, which is a 
        mapping from length bucket names to scalar tensors.
    
    Notes:
        Correlation and covariance metrics without the `_pooled` suffix are
        prompt-controlled aggregates of the per-prompt metrics. Other metrics are
        recomputed on all rollouts concatenated into one tensor.
    """
    sequence_lengths = torch.cat(per_prompt_sequence_lengths, dim=0).to(dtype=torch.float32)
    sequence_entropy = torch.cat(per_prompt_sequence_entropy, dim=0)
    per_prompt_rollout_counts = [int(t.numel()) for t in per_prompt_sequence_lengths]

    (
        tm_star_max,
        gen_ppl,
        branching_factor,
        entropy_rate_length_correlation_pooled,
        entropy_rate_length_covariance_pooled,
    ) = calculate_tree_rollout_metrics(sequence_entropy, sequence_lengths)

    entropy_rate_length_correlation, entropy_rate_length_covariance = (
        aggregate_prompt_controlled_correlation(
            correlations=[m.entropy_rate_length_correlation for m in per_prompt_metrics],
            covariances=[m.entropy_rate_length_covariance for m in per_prompt_metrics],
            per_prompt_rollout_counts=per_prompt_rollout_counts,
        )
    )
    semantic_diversity_length_correlation, semantic_diversity_length_covariance = (
        aggregate_prompt_controlled_correlation(
            correlations=[m.semantic_diversity_length_correlation for m in per_prompt_metrics],
            covariances=[m.semantic_diversity_length_covariance for m in per_prompt_metrics],
            per_prompt_rollout_counts=per_prompt_rollout_counts,
        )
    )

    semantic_diversity = torch.nanmean(
            torch.stack([m.semantic_diversity for m in per_prompt_metrics])
    )
    semantic_diversity_bucketed_mean = pool_bucketed_semantic_diversity(
            [m.bucketed_semantic_diversity for m in per_prompt_metrics]
    )

    return {
        "tm_star_max": tm_star_max,
        "gen_ppl": gen_ppl,
        "branching_factor": branching_factor,
        "entropy_rate_length_correlation": entropy_rate_length_correlation,
        "entropy_rate_length_covariance": entropy_rate_length_covariance,
        "entropy_rate_length_correlation_pooled": entropy_rate_length_correlation_pooled,
        "entropy_rate_length_covariance_pooled": entropy_rate_length_covariance_pooled,
        "semantic_diversity_length_correlation": semantic_diversity_length_correlation,
        "semantic_diversity_length_covariance": semantic_diversity_length_covariance,
        "semantic_diversity": semantic_diversity,
        "semantic_diversity_bucketed_mean": semantic_diversity_bucketed_mean,
    }
