from dataclasses import dataclass

import numpy as np
import torch

from src.generation_space.core.structures import PromptRolloutStats
from src.metrics import aggregate_correlation_arrays


@dataclass(slots=True)
class BootstrapEstimate:
    """
    Bootstrap uncertainty summary for one point estimate.

    Attributes:
        value: Original point estimate computed on the full non-resampled prompt list.
        se: Sample standard deviation (ddof=1) of bootstrap replicates.
        ci_low: Lower endpoint of the percentile CI.
        ci_high: Upper endpoint of the percentile CI.
    """

    value: float
    se: float
    ci_low: float
    ci_high: float


@dataclass(slots=True)
class _PromptComponents:
    """
    Per-prompt scalar arrays needed to recompute the bootstrap resample.
    Each field is a Tensor of shape [P], aligned by prompt index.

    Attributes:
        ce_star_max: per-prompt mean trajectory entropy `(1/M) sum_i H^(i)_sum`.
        mean_length: per-prompt mean rollout length `(1/M) sum_i N^(i)`.
        entropy_rate: per-prompt mean rollout entropy rate `(1/M) sum_i r^(i)_N`.
        rollout_counts: per-prompt rollout count `n_p`, used for correlation weights.
        pearson, spearman, kendall, covariance: per-prompt entropy-rate-vs-length
            coefficients on their native scale (no Fisher-z applied yet).
    """

    ce_star_max: torch.Tensor
    mean_length: torch.Tensor
    entropy_rate: torch.Tensor
    rollout_counts: torch.Tensor
    pearson: torch.Tensor
    spearman: torch.Tensor
    kendall: torch.Tensor
    covariance: torch.Tensor


def compute_paired_bootstrap_comparison(
    base_stats: list[PromptRolloutStats],
    instruct_stats: list[PromptRolloutStats],
    metric_scale_types: dict[str, str],
    n_boot: int,
    seed: int,
    ci_level: float = 0.95,
) -> dict[str, dict[str, BootstrapEstimate]]:
    """
    Computes bootstrap summaries for two aligned model variants and their differences.

    Each replicate resamples prompts with replacement and keeps all rollouts for
    each sampled prompt. Base and instruct use the same sampled prompt indices,
    so delta and percent-delta uncertainty is estimated from paired replicates.

    Args:
        base_stats: List of P `PromptRolloutStats` from the base variant.
        instruct_stats: List of P `PromptRolloutStats` from the instruct variant,
            aligned with base_stats by prompt index.
        metric_scale_types: Mapping from metric keys to scale types.
            `signed` metrics compute absolute change `delta` only.
            `positive` metrics also compute a relative change `percent_delta`.
        n_boot: Number of bootstrap replicates.
        seed: Seed for the RNG used to draw prompt indices.
        ci_level: confidence interval level.

    Returns:
        Nested mapping from metric name to bootstrap summaries for the base value,
        instruct value, absolute delta, and, for positive-scale metrics, percent delta.

    """
    if len(base_stats) != len(instruct_stats):
        raise ValueError(
            f"paired bootstrap requires equal P; "
            f"got base={len(base_stats)} instruct={len(instruct_stats)}."
        )
    n_prompts = len(base_stats)

    base_components = _extract_components(base_stats)
    instruct_components = _extract_components(instruct_stats)

    rng = np.random.default_rng(seed)
    prompt_indices = torch.from_numpy(
        rng.integers(low=0, high=n_prompts, size=(n_boot, n_prompts))
    )

    base_replicates_by_metric = _aggregate(base_components, prompt_indices)
    instruct_replicates_by_metric = _aggregate(instruct_components, prompt_indices)

    full_indices = torch.arange(n_prompts).unsqueeze(0)  # [1, P]
    base_full_estimates = {
        key: float(metric_val[0])
        for key, metric_val in _aggregate(base_components, full_indices).items()
    }
    instruct_full_estimates = {
        key: float(metric_val[0])
        for key, metric_val in _aggregate(instruct_components, full_indices).items()
    }

    bootstrap_results: dict[str, dict[str, BootstrapEstimate]] = {}
    for metric, scale_type in metric_scale_types.items():
        base_full_estimate = base_full_estimates[metric]
        instruct_full_estimate = instruct_full_estimates[metric]
        base_replicates = base_replicates_by_metric[metric]
        instruct_replicates = instruct_replicates_by_metric[metric]
        entry: dict[str, BootstrapEstimate] = {
            "base": _compute_bootstrap_estimate(base_full_estimate, base_replicates, ci_level),
            "instruct": _compute_bootstrap_estimate(
                instruct_full_estimate, instruct_replicates, ci_level
            ),
            "delta": _compute_bootstrap_estimate(
                instruct_full_estimate - base_full_estimate,
                instruct_replicates - base_replicates,
                ci_level,
            ),
        }
        if scale_type == "positive":
            entry["percent_delta"] = _compute_bootstrap_estimate(
                100.0 * (instruct_full_estimate - base_full_estimate) / abs(base_full_estimate),
                100.0 * (instruct_replicates - base_replicates) / torch.abs(base_replicates),
                ci_level,
            )
        bootstrap_results[metric] = entry
    return bootstrap_results


def _extract_components(prompt_stats: list[PromptRolloutStats]) -> _PromptComponents:
    """Stacks per-prompt scalar metrics into prompt-aligned 1-D tensors."""
    metrics = [s.raw_metrics for s in prompt_stats]
    correlations = [m.entropy_rate_vs_length for m in metrics]
    return _PromptComponents(
        ce_star_max=torch.stack([m.ce_star_max for m in metrics]),
        mean_length=torch.stack([s.raw_sequence_lengths.mean() for s in prompt_stats]),
        entropy_rate=torch.stack([m.entropy_rate for m in metrics]),
        rollout_counts=torch.tensor(
            [s.raw_sequence_lengths.numel() for s in prompt_stats], dtype=torch.float32
        ),
        pearson=torch.stack([c.pearson for c in correlations]),
        spearman=torch.stack([c.spearman for c in correlations]),
        kendall=torch.stack([c.kendall for c in correlations]),
        covariance=torch.stack([c.covariance for c in correlations]),
    )


def _aggregate(components: _PromptComponents, prompt_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Recomputes pooled metrics for each bootstrap replicate, i.e, each row of `prompt_indices`.

    Args:
        components: Prompt-aligned scalar tensors, each field has shape [P].
        prompt_indices: Integer tensor of shape [B, P], where row b contains the
            prompt indices sampled with replacement for bootstrap replicate b.

    Returns:
        Mapping from each metric key to a tensor of shape [B] containing that
        metric's pooled value for each bootstrap replicate.
    """
    # [B, P] => [B]
    ce = torch.nanmean(components.ce_star_max[prompt_indices], dim=-1)
    mean_length = torch.nanmean(components.mean_length[prompt_indices], dim=-1)
    entropy_rate = torch.nanmean(components.entropy_rate[prompt_indices], dim=-1)
    correlation = aggregate_correlation_arrays(
        pearson=components.pearson[prompt_indices],
        spearman=components.spearman[prompt_indices],
        kendall=components.kendall[prompt_indices],
        covariance=components.covariance[prompt_indices],
        counts=components.rollout_counts[prompt_indices],
    )
    return {
        "ce_star_max": ce,
        "gen_ppl": torch.exp(ce / mean_length),
        "branching_factor": torch.exp(entropy_rate),
        "entropy_rate_vs_length.covariance": correlation.covariance,
        "entropy_rate_vs_length.pearson": correlation.pearson,
        "entropy_rate_vs_length.spearman": correlation.spearman,
        "entropy_rate_vs_length.kendall": correlation.kendall,
    }


def _compute_bootstrap_estimate(
    full_estimate: float, replicates: torch.Tensor, ci_level: float
) -> BootstrapEstimate:
    """
    Computes a bootstrap estimate from the deterministic estimate and its replicates.

    Args:
        full_estimate: Deterministic estimate computed from the full, non-resampled data.
        replicates: Tensor of shape [B] containing bootstrap replicate values for the estimate.
        ci_level: Confidence interval level.

    Returns:
        Bootstrap estimate containing the deterministic estimate, the replicate
        standard error, and percentile confidence interval endpoints.
    """
    alpha = (1.0 - ci_level) / 2.0
    ci_low, ci_high = torch.quantile(
        replicates,
        torch.tensor([alpha, 1.0 - alpha], dtype=replicates.dtype, device=replicates.device),
    )
    return BootstrapEstimate(
        value=full_estimate,
        se=float(replicates.std(unbiased=True)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )
