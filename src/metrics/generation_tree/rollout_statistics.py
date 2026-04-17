from __future__ import annotations
from dataclasses import dataclass

import torch

from .branching_factor import calculate_branching_factor, calculate_diversity_correlation
from .gen_ppl import calculate_gen_ppl, calculate_tm_star_max


@dataclass(slots=True)
class RolloutMetrics:
    """
    Scalar generation space metrics computed from M rollout-level tensors for one prompt.

    Attributes:
        tm_star_max: `(1/M) * sum_i H^(i)_sum`, 
            where `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        gen_ppl: `exp(tm_star_max / E[N])`, where N is the generation length.
        branching_factor: `exp((1/M) * sum_i r^(i)_N)`,
            where `r^(i)_N = H^(i)_sum / N^(i)`.
        diversity_correlation: Pearson correlation over `(N^(i), r^(i)_N)`.
        diversity_covariance: Covariance over `(N^(i), r^(i)_N)`.
    """

    tm_star_max: torch.Tensor
    gen_ppl: torch.Tensor
    branching_factor: torch.Tensor
    diversity_correlation: torch.Tensor
    diversity_covariance: torch.Tensor

    def to_cpu(self) -> RolloutMetrics:
        return RolloutMetrics(
            tm_star_max=self.tm_star_max.cpu(),
            gen_ppl=self.gen_ppl.cpu(),
            branching_factor=self.branching_factor.cpu(),
            diversity_correlation=self.diversity_correlation.cpu(),
            diversity_covariance=self.diversity_covariance.cpu(),
        )


def calculate_rollout_metrics(
    sequence_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> RolloutMetrics:
    """
    Computes scalar generation space metrics from M rollout-level tensors for one prompt.

    Args:
        sequence_entropy: Tensor of shape [M], where i-th entry is
            `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is generated
            token length `N^(i)`.
    
    Returns:
        `RolloutMetrics`: Scalar generation space metrics, including tm_star_max,
        gen_ppl, branching_factor, diversity_correlation, and diversity_covariance.
    """
    sequence_entropy = sequence_entropy.to(dtype=torch.float32)
    sequence_lengths = sequence_lengths.to(dtype=torch.float32)

    tm_star_max = calculate_tm_star_max(sequence_entropy)
    gen_ppl = calculate_gen_ppl(tm_star_max, sequence_lengths)
    branching_factor = calculate_branching_factor(sequence_entropy, sequence_lengths)
    diversity_correlation, diversity_covariance = calculate_diversity_correlation(
        sequence_entropy,
        sequence_lengths,
    )

    return RolloutMetrics(
        tm_star_max=tm_star_max,
        gen_ppl=gen_ppl,
        branching_factor=branching_factor,
        diversity_correlation=diversity_correlation,
        diversity_covariance=diversity_covariance,
    )


def calculate_prompt_controlled_diversity(
    correlations: list[torch.Tensor],
    covariances: list[torch.Tensor],
    sample_sizes: list[int],
) -> dict[str, torch.Tensor]:
    """
    Aggregates diversity metrics across prompts while controlling for prompt effects.
    For each prompt p:
    - `N^(p) = {N^(p,i)}_{i=1}^{n_p}` is a list of rollout lengths for prompt p.
    - `r^(p)_N = {r^(p,i)_N}_{i=1}^{n_p}` is a list of average rollout uncertainty rates,
      where `r^(p,i)_N = H^(p,i)_sum / N^(p,i)`.

    Then aggregate across prompts:
    - Correlation is bounded and non-linear, so it is averaged after the Fisher-z
      transformation with weights `n_p - 3`, and then transformed back to the
      Pearson coefficient scale.
    - Covariance is aggregated as a weighted mean with weights `n_p - 1`.

    Args:
        correlations: List of length P where p-th entry is a scalar tensor for
            per-prompt Pearson correlation `corr_p = rho(N^(p), r^(p)_N)`.
        covariances: List of length P where p-th entry is a scalar tensor for
            per-prompt covariance `cov_p = Cov(N^(p), r^(p)_N)`.
        sample_sizes: List of length P where p-th entry is prompt rollout count `n_p`.

    Returns:
        dict: Dictionary with keys paired to scalar tensor values
        - diversity_correlation: Prompt-controlled aggregate Pearson correlation.
        - diversity_covariance: Prompt-controlled aggregate covariance.
    """

    correlation_tensor = torch.stack(correlations) # [P]
    covariance_tensor = torch.stack(covariances) # [P]
    sample_size_tensor = torch.as_tensor(
        sample_sizes,
        dtype=correlation_tensor.dtype,
        device=correlation_tensor.device,
    ) # [P]

    fisher_weights = (sample_size_tensor - 3.0).clamp_min(0.0)
    clipped_correlation = correlation_tensor.clamp(min=-0.999999, max=0.999999)
    fisher_z = torch.atanh(clipped_correlation)
    weighted_fisher_z = _weighted_nanmean(fisher_z, fisher_weights)
    correlation_aggregate = torch.tanh(weighted_fisher_z)

    covariance_weights = (sample_size_tensor - 1.0).clamp_min(0.0)
    covariance_aggregate = _weighted_nanmean(covariance_tensor, covariance_weights)

    return {
        "diversity_correlation": correlation_aggregate,
        "diversity_covariance": covariance_aggregate,
    }


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