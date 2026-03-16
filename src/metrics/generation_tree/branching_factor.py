import torch


def _calculate_rollout_average_uncertainty(
    sequence_conditional_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Computes per-rollout average uncertainty `r^(i)_N = H^(i)_sum / N^(i)` for each rollout, 
    where `N^(i)` is the i-th rollout length, and `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.

    Args:
        sequence_conditional_entropy: Tensor of shape [M] with summed step entropies per rollout,
            where i-th entry is `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M] with generated token lengths per rollout,
            where i-th entry is `N^(i)`.

    Returns:
        Tensor of shape [M] where i-th entry is sequence_conditional_entropy[i] / sequence_lengths[i],
        i.e., `r^(i)_N = H^(i)_sum / N^(i)`.
    """
    safe_lengths = sequence_lengths.clamp(min=1.0)
    return sequence_conditional_entropy / safe_lengths


def calculate_branching_factor(
    sequence_conditional_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Estimates branching factor from Monte Carlo rollout statistics.

    Args:
        sequence_conditional_entropy: Tensor of shape [M], where i-th entry is the sum of
            step entropies of rollout i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is the generated
            token length of rollout i, `N^(i)`.

    Returns:
        Scalar tensor BF = exp(E_N[r^(i)_N]) = exp((1/M) * sum_i r^(i)_N),
        where `r^(i)_N = H^(i)_sum / N^(i)`.
    """
    # [M], i-th entry is r^(i)_N = H^(i)_sum / N^(i)
    rollout_average_uncertainty = _calculate_rollout_average_uncertainty(
        sequence_conditional_entropy, sequence_lengths
    )

    expected_rollout_average_uncertainty = rollout_average_uncertainty.mean()
    return torch.exp(expected_rollout_average_uncertainty)


def calculate_diversity_correlation(
    sequence_conditional_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Measures co-movement between generation length and average trajectory entropy.

    Args:
        sequence_conditional_entropy: Tensor of shape [M], where i-th entry is the sum of
            step entropies of rollout i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is the generated
            token length N^(i) of rollout i.

    Returns:
        (pearson_correlation, covariance): Tuple that contains
        - pearson_correlation: Scalar tensor `rho(N, r_N)` over rollout pairs `(N^(i), r^(i)_N)`.
        - covariance: Scalar tensor `Cov(N, r_N)` over rollout pairs `(N^(i), r^(i)_N)`.
    """
    # [M], i-th entry is r^(i)_N = H^(i)_sum / N^(i)
    rollout_average_uncertainty = _calculate_rollout_average_uncertainty(
        sequence_conditional_entropy, sequence_lengths
    )

    #[2, M]
    rollout_pairs = torch.stack([sequence_lengths, rollout_average_uncertainty], dim=0)
    covariance = torch.cov(rollout_pairs, correction=1)[0, 1]
    pearson_correlation = torch.corrcoef(rollout_pairs)[0, 1]
    return pearson_correlation, covariance
