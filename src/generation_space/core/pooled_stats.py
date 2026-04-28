import torch

from src.metrics import PromptMetrics, aggregate_prompt_controlled_correlation
from src.metrics.semantic_metrics import pool_bucketed_semantic_diversity

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
        dictionary keyed by `GENERATION_SPACE_METRIC_KEYS`. The two
        `*_vs_length` entries are `Correlation`, `semantic_diversity_bucketed_mean`
        maps length bucket names to scalar tensors, and all others are scalar tensors.

    Notes:
        Pooled linear quantities are unweighted per-prompt means. `exp` is applied
        after averaging. Correlations and covariances use the prompt-controlled aggregate.
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
        Metric dictionary keyed by `GENERATION_SPACE_METRIC_KEYS`. The two
        `*_vs_length` entries are `Correlation`, `semantic_diversity_bucketed_mean`
        maps length bucket names to scalar tensors, and all others are scalar tensors.

    Notes:
        Pooled linear quantities are unweighted per-prompt means. `exp` is applied
        after averaging. Correlations and covariances use the prompt-controlled aggregate.
    """
    per_prompt_rollout_counts = [int(t.numel()) for t in per_prompt_sequence_lengths]

    tm_star_max = torch.nanmean(
        torch.stack([m.tm_star_max for m in per_prompt_metrics]) # [P]
    )
    mean_length = torch.nanmean(
        torch.stack([lengths.to(dtype=torch.float32).mean() for lengths in per_prompt_sequence_lengths]) # [P]
    )
    mean_entropy_rate = torch.nanmean(
        torch.stack([
            (entropy / lengths.to(dtype=torch.float32).clamp(min=1.0)).mean()
            for entropy, lengths in zip(
                per_prompt_sequence_entropy, per_prompt_sequence_lengths, strict=True
            ) # P pairs of tensors with shape [n_p]
        ]) # [P] of mean([n_p] of entropy / length)
    )
    gen_ppl = torch.exp(tm_star_max / mean_length)
    branching_factor = torch.exp(mean_entropy_rate)

    entropy_rate_vs_length = aggregate_prompt_controlled_correlation(
        correlations=[m.entropy_rate_vs_length for m in per_prompt_metrics],
        per_prompt_rollout_counts=per_prompt_rollout_counts,
    )
    semantic_diversity_vs_length = aggregate_prompt_controlled_correlation(
        correlations=[m.semantic_diversity_vs_length for m in per_prompt_metrics],
        per_prompt_rollout_counts=per_prompt_rollout_counts,
    )

    semantic_diversity = torch.nanmean(
        torch.stack([m.semantic_diversity for m in per_prompt_metrics]) # [P]
    )
    semantic_diversity_bucketed_mean = pool_bucketed_semantic_diversity(
        [m.bucketed_semantic_diversity for m in per_prompt_metrics] # list of P dicts
    ) # dict of {bucket_name: scalar mean over P prompts}
    truncation_rate = torch.nanmean(
        torch.stack([m.truncation_rate for m in per_prompt_metrics]) # [P]
    )

    return {
        "tm_star_max": tm_star_max,
        "gen_ppl": gen_ppl,
        "branching_factor": branching_factor,
        "entropy_rate_vs_length": entropy_rate_vs_length,
        "truncation_rate": truncation_rate,
        "semantic_diversity_vs_length": semantic_diversity_vs_length,
        "semantic_diversity": semantic_diversity,
        "semantic_diversity_bucketed_mean": semantic_diversity_bucketed_mean,
    }
