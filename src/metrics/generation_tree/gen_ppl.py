import torch


def calculate_tm_star_max(sequence_conditional_entropy: torch.Tensor) -> torch.Tensor:
    """
    Computes expected cumulative branching uncertainty `TM*_max` from rollout sequence entropy.

    Args:
        sequence_conditional_entropy: Tensor of shape [M], where i-th entry is
            `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.

    Returns:
        Scalar tensor `TM*_max = (1/M) * sum_i H^(i)_sum`.
    """
    return sequence_conditional_entropy.mean()


def calculate_gen_ppl(tm_star_max: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Estimates GenPPL from rollout-level uncertainty and generated lengths.

    Args:
        tm_star_max: Scalar tensor representing expected cumulative branching uncertainty, 
            i.e., `(1/M) * sum_i H^(i)_sum, where H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is the generated
            token length of rollout i, `N^(i)`.

    Returns:
        Scalar tensor representing `GenPPL_max = tm_star_max / mean_seq_len`.
    """
    expected_length = sequence_lengths.mean()

    entropy_rate_max = tm_star_max / expected_length
    gen_ppl_max = torch.exp(entropy_rate_max)

    return gen_ppl_max
