import torch


def aggregate_prompt_controlled_correlation(
    correlations: list[torch.Tensor],
    covariances: list[torch.Tensor],
    sample_sizes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates per-prompt correlations and covariances across prompts while
    controlling for prompt effects.

    For each prompt p, with rollout number i ranging from 1 to `n_p`:
    - `N^(p) = {N^(p,i)}_{i=1}^{n_p}` is a list of rollout lengths for prompt p.
    - `z^(p) = {z^(p,i)}_{i=1}^{n_p}` is a list of per-rollout values being correlated with length.
        - Generation tree entropy rate: a list of entropy rates per rollout, i.e.,
            `z^(p,i) = r^(p,i)_N = H^(p,i)_sum / N^(p,i)`.
        - Semantic diversity: a list of mean dissimilarity from each rollout to other rollouts,
            i.e., `z^(p,i) = d^(p,i) = 1 - (1/(M-1)) * sum_{j!=i} <e_(p,i), e_(p,j)>`

    Then aggregate across prompts:
    - Correlation is bounded and non-linear, so it is averaged after the Fisher-z
        transformation with weights `n_p - 3`, and then transformed back to the
        Pearson coefficient scale.
    - Covariance is aggregated as a weighted mean with weights `n_p - 1`.

    Args:
        correlations: List of length P where p-th entry is a scalar tensor for
            per-prompt Pearson correlation `corr_p = rho(N^(p), z^(p))`.
        covariances: List of length P where p-th entry is a scalar tensor for
            per-prompt covariance `cov_p = Cov(N^(p), z^(p))`.
        sample_sizes: List of length P where p-th entry is prompt rollout count `n_p`.

    Returns:
        (correlation_aggregate, covariance_aggregate): Tuple of scalar tensors
        - correlation_aggregate: Prompt-controlled aggregate Pearson correlation.
        - covariance_aggregate: Prompt-controlled aggregate covariance.
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

    return correlation_aggregate, covariance_aggregate


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
