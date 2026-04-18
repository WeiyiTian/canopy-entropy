import torch


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
    rollout_entropy_rate = calculate_rollout_entropy_rate(
        sequence_conditional_entropy, sequence_lengths
    )

    expected_rollout_entropy_rate = rollout_entropy_rate.mean()
    return torch.exp(expected_rollout_entropy_rate)


def calculate_rollout_entropy_rate(
    sequence_conditional_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Computes per-rollout entropy rate `r^(i)_N = H^(i)_sum / N^(i)` for each rollout,
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