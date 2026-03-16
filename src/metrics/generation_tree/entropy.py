import torch

from ...models import normalize_scores


def step_conditional_entropy_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    For a single rollout, computes per-step conditional entropy `H(Y_t | X, y_<t)`
    from candidate token scores.

    Args:
        scores: Candidate scores for a single rollout with shape [T, K], 
            where K is the number of candidate tokens for each step.

    Returns:
        Tensor of shape [T] containing entropy values at each step.

    Notes:
        Scores are normalized to log-probabilities over the last dimension
        before entropy is computed. Non-finite entries are ignored.
    """
    normalized_log_probs = normalize_scores(scores)

    finite_mask = torch.isfinite(normalized_log_probs)
    finite_log_probs = torch.where(
        finite_mask,
        normalized_log_probs,
        torch.zeros_like(normalized_log_probs),
    )

    finite_probs = torch.where(
        finite_mask,
        normalized_log_probs.exp(),
        torch.zeros_like(normalized_log_probs),
    )
    
    return -(finite_probs * finite_log_probs).sum(dim=-1).to(dtype=torch.float32)


def step_entropy_and_sequence_entropy(
    sequence_step_scores: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Computes per-step conditional entropy for each sequence and per-sequence total conditional entropy.

    Args:
        sequence_step_scores: List of length M. The i-th tensor has shape
            [T_i, K_i_max], where T_i is sequence i's generated length and
            K_i_max is the number of retained candidate scores per step
            padded with `-inf`.
            
    Returns:
        (step_entropy_per_sequence, sequence_entropy): Tuple that contains
        - step_entropy_per_sequence: List of length M. The i-th tensor has
            shape [T_i] with per-step entropies `H(Y^(i)_t | X, y^(i)_<t)`.
        - sequence_entropy: Tensor of shape [M], where entry i is the sum of
            step entropies of sequence i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
    """
    device = sequence_step_scores[0].device
    sequence_entropy = torch.empty(
        (len(sequence_step_scores),),
        dtype=torch.float32,
        device=device,
    )
    step_entropy_per_sequence: list[torch.Tensor] = []

    for sequence_idx, scores in enumerate(sequence_step_scores):
        step_entropy = step_conditional_entropy_from_scores(scores=scores)
        step_entropy_per_sequence.append(step_entropy)
        # H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)
        sequence_entropy[sequence_idx] = step_entropy.sum()

    return step_entropy_per_sequence, sequence_entropy
