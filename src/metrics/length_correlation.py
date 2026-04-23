from __future__ import annotations
from dataclasses import dataclass, fields

import torch
from torchmetrics.functional import kendall_rank_corrcoef, spearman_corrcoef


@dataclass(slots=True)
class LengthCorrelation:
    """
    Co-movement between per-rollout generation lengths and a per-rollout value
    of interest.

    Attributes:
        pearson: Pearson correlation `rho(N, z)`.
        spearman: Spearman rank correlation over `(N, z)`; rank-based, uses
            mid-ranks for ties.
        kendall: Kendall's tau-b over `(N, z)` with tie-adjusted concordance.
        covariance: Pearson-scale covariance `Cov(N, z)`.
    """

    pearson: torch.Tensor
    spearman: torch.Tensor
    kendall: torch.Tensor
    covariance: torch.Tensor

    def to_cpu(self) -> LengthCorrelation:
        """Moves all tensors in the LengthCorrelation to CPU and returns a new instance."""
        return LengthCorrelation(**{f.name: getattr(self, f.name).cpu() for f in fields(self)})


def calculate_length_correlation(
    sequence_lengths: torch.Tensor,
    per_rollout_values: torch.Tensor,
) -> LengthCorrelation:
    """
    Measures co-movement between per-rollout generation lengths and a per-rollout
    value of interest.

    Args:
        sequence_lengths: Tensor of shape [M] with rollout lengths `N^(i)`.
        per_rollout_values: Tensor of shape [M] with rollout values `z^(i)`.

    Returns:
        `LengthCorrelation` containing Pearson/Spearman/Kendall coefficients and
        the Pearson covariance computed over the M rollouts.
    """
    # [2, M]
    pairs = torch.stack([sequence_lengths, per_rollout_values], dim=0)
    covariance = torch.cov(pairs, correction=1)[0, 1]
    pearson = torch.corrcoef(pairs)[0, 1]

    lengths_f32 = sequence_lengths.to(dtype=torch.float32)
    values_f32 = per_rollout_values.to(dtype=torch.float32)
    spearman = spearman_corrcoef(lengths_f32, values_f32)
    kendall = kendall_rank_corrcoef(lengths_f32, values_f32, variant="b")

    return LengthCorrelation(
        pearson=pearson,
        spearman=spearman,
        kendall=kendall,
        covariance=covariance,
    )


def aggregate_prompt_controlled_correlation(
    length_correlations: list[LengthCorrelation],
    per_prompt_rollout_counts: list[int],
) -> LengthCorrelation:
    """
    Aggregates per-prompt `LengthCorrelation` across prompts while controlling
    for prompt effects.

    For each prompt p, with rollout number i ranging from 1 to `n_p`:
    - `N^(p) = {N^(p,i)}_{i=1}^{n_p}` is a list of rollout lengths for prompt p.
    - `z^(p) = {z^(p,i)}_{i=1}^{n_p}` is a list of per-rollout values being correlated with length.
        - Generation tree entropy rate: a list of entropy rates per rollout, i.e.,
            `z^(p,i) = r^(p,i)_N = H^(p,i)_sum / N^(p,i)`.
        - Semantic diversity: a list of mean dissimilarity from each rollout to other rollouts,
            i.e., `z^(p,i) = d^(p,i) = 1 - (1/(M-1)) * sum_{j!=i} <e_(p,i), e_(p,j)>`

    Per-coefficient aggregation:
    - Pearson and Spearman (bounded and non-linear): Fisher-z transform, weighted mean
        with weights `n_p - 3`, then transform back to the r scale.
    - Kendall tau-b (combinatorial): direct weighted mean on the τ scale with weights
        `n_p (n_p - 1) / 2` (pair count).
    - Covariance: weighted mean with weights `n_p - 1`.

    Args:
        length_correlations: List of length P where p-th entry is `LengthCorrelation`
        containing 4 co-movement measures computed for prompt p.
        per_prompt_rollout_counts: List of length P where p-th entry is prompt
            rollout count `n_p` as an int.

    Returns:
        `LengthCorrelation` containing the prompt-controlled aggregates of the
        Pearson, Spearman, and Kendall coefficients and the Pearson-scale covariance.
    """
    pearson = torch.stack([lc.pearson for lc in length_correlations]) # [P]
    spearman = torch.stack([lc.spearman for lc in length_correlations]) # [P]
    kendall = torch.stack([lc.kendall for lc in length_correlations]) # [P]
    covariance = torch.stack([lc.covariance for lc in length_correlations]) # [P]

    counts = torch.as_tensor(
        per_prompt_rollout_counts,
        dtype=pearson.dtype,
        device=pearson.device,
    ) # [P]
    fisher_weights = (counts - 3.0).clamp_min(0.0)
    kendall_pair_weights = (counts * (counts - 1.0) / 2.0).clamp_min(0.0)
    covariance_weights = (counts - 1.0).clamp_min(0.0)

    return LengthCorrelation(
        pearson=_aggregate_via_fisher_z(pearson, fisher_weights),
        spearman=_aggregate_via_fisher_z(spearman, fisher_weights),
        kendall=_weighted_nanmean(kendall, kendall_pair_weights),
        covariance=_weighted_nanmean(covariance, covariance_weights),
    )


def _aggregate_via_fisher_z(correlations: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted mean of correlations under the Fisher-z transform.

    Args:
        correlations: Tensor of shape [P] with per-prompt correlation coefficients
            on the Pearson scale (r in [-1, 1]).
        weights: Tensor of shape [P] with non-negative weights.

    Returns:
        Scalar tensor `tanh(weighted_nanmean(atanh(r), weights))`.
    """
    # clipped to avoid infinities at the boundary
    clipped_corr = correlations.clamp(min=-0.999999, max=0.999999)
    fisher_z = torch.atanh(clipped_corr)
    weighted_z = _weighted_nanmean(fisher_z, weights)
    return torch.tanh(weighted_z)


def _weighted_nanmean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes a weighted mean while ignoring invalid entries.

    Args:
        values: Tensor of shape [P] containing values to aggregate.
        weights: Tensor of shape [P] containing non-negative weights.

    Returns:
        Scalar tensor equal to the weighted mean over valid entries where both
        value and weight are finite and weight > 0.

    Notes:
        If no valid weighted entry exists, falls back to `torch.nanmean(values)`.
    """
    valid_mask = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    safe_values = torch.where(valid_mask, values, torch.zeros_like(values))
    safe_weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

    total_weight = safe_weights.sum()
    if total_weight > 0:
        return (safe_values * safe_weights).sum() / total_weight

    return torch.nanmean(values)
