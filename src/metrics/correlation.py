from __future__ import annotations
from dataclasses import dataclass, fields

import torch
from torchmetrics.functional import kendall_rank_corrcoef, spearman_corrcoef


@dataclass(slots=True)
class Correlation:
    """
    Co-movement between two per-rollout values.

    Attributes:
        pearson: Pearson correlation `rho(a, b)`.
        spearman: Spearman rank correlation over `(a, b)`; rank-based, uses
            mid-ranks for ties.
        kendall: Kendall's tau-b over `(a, b)` with tie-adjusted concordance.
        covariance: Pearson-scale covariance `Cov(a, b)`.
    """

    pearson: torch.Tensor
    spearman: torch.Tensor
    kendall: torch.Tensor
    covariance: torch.Tensor

    def to_cpu(self) -> Correlation:
        """Moves all tensors in the Correlation to CPU and returns a new instance."""
        return Correlation(**{f.name: getattr(self, f.name).cpu() for f in fields(self)})


def calculate_correlation(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Correlation:
    """
    Measures co-movement between two per-rollout values.

    Args:
        a: Tensor of shape [M].
        b: Tensor of shape [M].

    Returns:
        `Correlation` containing Pearson/Spearman/Kendall coefficients and
        the Pearson covariance computed over the M paired observations.
    """
    # [2, M]
    pairs = torch.stack([a, b], dim=0)
    covariance = torch.cov(pairs, correction=1)[0, 1]
    pearson = torch.corrcoef(pairs)[0, 1]

    a_f32 = a.to(dtype=torch.float32)
    b_f32 = b.to(dtype=torch.float32)
    spearman = spearman_corrcoef(a_f32, b_f32)
    kendall = kendall_rank_corrcoef(a_f32, b_f32, variant="b")

    return Correlation(
        pearson=pearson,
        spearman=spearman,
        kendall=kendall,
        covariance=covariance,
    )


def aggregate_prompt_controlled_correlation(
    correlations: list[Correlation],
    per_prompt_rollout_counts: list[int],
) -> Correlation:
    """
    Aggregates per-prompt `Correlation` across prompts while controlling
    for prompt effects.

    Args:
        correlations: List of length P where p-th entry is `Correlation`
        containing 4 co-movement measures computed for prompt p.
        per_prompt_rollout_counts: List of length P where p-th entry is prompt
            rollout count `n_p` as an int.

    Returns:
        `Correlation` containing the prompt-controlled aggregates of the
        Pearson, Spearman, and Kendall coefficients and the Pearson-scale covariance.
    """
    pearson = torch.stack([c.pearson for c in correlations]) # [P]
    spearman = torch.stack([c.spearman for c in correlations]) # [P]
    kendall = torch.stack([c.kendall for c in correlations]) # [P]
    covariance = torch.stack([c.covariance for c in correlations]) # [P]
    counts = torch.as_tensor(
        per_prompt_rollout_counts,
        dtype=pearson.dtype,
        device=pearson.device,
    ) # [P]
    return aggregate_correlation_arrays(pearson, spearman, kendall, covariance, counts)


def aggregate_correlation_arrays(
    pearson: torch.Tensor,
    spearman: torch.Tensor,
    kendall: torch.Tensor,
    covariance: torch.Tensor,
    counts: torch.Tensor,
) -> Correlation:
    """
    Aggregates per-prompt correlation coefficients across prompts while controlling
    for prompt effects.

    For each prompt p, with rollout number i ranging from 1 to `n_p`, a per-prompt
    `Correlation` summarizes co-movement between two per-rollout values
    `a^(p) = {a^(p,i)}_{i=1}^{n_p}` and `b^(p) = {b^(p,i)}_{i=1}^{n_p}`.
    - Generation tree entropy rate: `a = N` (rollout length) and `b = r_N` (entropy rate),
        where `r^(p,i)_N = H^(p,i)_sum / N^(p,i)`.
    - Semantic diversity: `a = N` (rollout length) and `b = d` (list of mean dissimilarity from
        each rollout to other rollouts), `where d^(p,i) = 1 - (1/(M-1)) * sum_{j!=i} <e_(p,i), e_(p,j)>`.

    Per-coefficient aggregation:
    - Pearson and Spearman (bounded and non-linear): Fisher-z transform, weighted mean
        with weights `n_p - 3`, then transform back to the r scale.
    - Kendall tau-b (combinatorial): direct weighted mean on the τ scale with weights
        `n_p (n_p - 1) / 2` (pair count).
    - Covariance: weighted mean with weights `n_p - 1`.

    Args:
        pearson: Tensor of shape [..., P] containing per-prompt Pearson correlations.
        spearman: Tensor of shape [..., P] containing per-prompt Spearman correlations.
        kendall: Tensor of shape [..., P] containing per-prompt Kendall tau-b coefficients.
        covariance: Tensor of shape [..., P] containing per-prompt Pearson covariances.
        counts: Tensor of shape [..., P] with rollout counts `n_p` aligned with the coefficient tensors.

    Returns:
        `Correlation` containing the prompt-controlled aggregates over the last axis of
        Pearson, Spearman, and Kendall coefficients and the Pearson-scale covariance.
    """
    fisher_weights = (counts - 3.0).clamp_min(0.0)
    kendall_pair_weights = (counts * (counts - 1.0) / 2.0).clamp_min(0.0)
    covariance_weights = (counts - 1.0).clamp_min(0.0)

    return Correlation(
        pearson=_aggregate_via_fisher_z(pearson, fisher_weights),
        spearman=_aggregate_via_fisher_z(spearman, fisher_weights),
        kendall=_weighted_nanmean(kendall, kendall_pair_weights),
        covariance=_weighted_nanmean(covariance, covariance_weights),
    )


def _aggregate_via_fisher_z(correlations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted mean of correlations under the Fisher-z transformation along the last axis.

    Args:
        correlations: Tensor of shape [..., P] with per-prompt correlation
            coefficients on the Pearson scale (r in [-1, 1]).
        weights: Tensor of shape [..., P] with non-negative weights.

    Returns:
        Tensor of shape [...] equal to `tanh(weighted_nanmean(atanh(r), weights))`.
    """
    clipped_corr = correlations.clamp(min=-0.999999, max=0.999999)
    fisher_z = torch.atanh(clipped_corr)
    weighted_z = _weighted_nanmean(fisher_z, weights)
    return torch.tanh(weighted_z)


def _weighted_nanmean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes a weighted mean over the last axis, ignoring invalid entries.

    Args:
        values: Tensor of shape [..., P] containing values to aggregate.
        weights: Tensor of shape [..., P] containing non-negative weights.

    Returns:
        Tensor of shape [...] with the weighted mean over the last axis.
        Rows with no valid weights fall back to `torch.nanmean`.
    """
    valid_mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    safe_values = torch.where(valid_mask, values, torch.zeros_like(values))
    safe_weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

    total_weight = safe_weights.sum(dim=-1) # [...]
    weighted_sum = (safe_values * safe_weights).sum(dim=-1) # [...]
    has_weight = total_weight > 0 # [...]
    fallback_weights = torch.nanmean(values, dim=-1) # [...]
    return torch.where(
        has_weight,
        weighted_sum / torch.where(has_weight, total_weight, torch.ones_like(total_weight)),
        fallback_weights,
    )
