import torch


def step_conditional_entropy_from_logprobs(logprobs: torch.Tensor) -> torch.Tensor:
    """
    For a single rollout, computes per-step conditional entropy `H(Y_t | X, y_<t)`
    from candidate token log-probabilities.

    Args:
        logprobs: Candidate log-probabilities for a single rollout with shape [T, K],
            where K is the number of candidate tokens for each step.

    Returns:
        Tensor of shape [T] containing entropy values at each step.

    Notes:
        Log-probabilities are assumed to already be normalized over the last
        dimension before entropy is computed. Non-finite entries are ignored.
    """
    finite_mask = torch.isfinite(logprobs)
    finite_logprobs = torch.where(finite_mask, logprobs, torch.zeros_like(logprobs))
    finite_probs = torch.where(finite_mask, logprobs.exp(), torch.zeros_like(logprobs))
    return -(finite_probs * finite_logprobs).sum(dim=-1).to(dtype=torch.float32)


def sequence_entropy_from_step_entropy(
    step_conditional_entropy: list[torch.Tensor],
) -> torch.Tensor:
    """
    Sums per-step entropies into per-sequence total conditional entropies.

    Args:
        step_conditional_entropy: List of length M. The i-th tensor has shape
            [T_i] containing per-step entropies `H(Y^(i)_t | X, y^(i)_<t)`.

    Returns:
        Tensor of shape [M], where entry i is the sum of step entropies
            of sequence i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
    """
    return torch.stack(
        # H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)
        [step_entropy.sum() for step_entropy in step_conditional_entropy]
    ).to(dtype=torch.float32)
