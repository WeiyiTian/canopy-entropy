import torch


def _truncate_scores(masked_scores: torch.Tensor, pad_value=torch.tensor(float("-inf"))) -> torch.Tensor:
    """
    Truncates scores by removing -inf positions and right-padding with `pad_value`.
    
    Args:
        masked_scores: Tensor of shape [T, num_candidates] containing scores with -inf for invalid tokens.
    
    Returns:
        Tensor of shape [T, max_valid] containing scores of valid tokens, padded with `pad_value`.
    """
    num_candidates = masked_scores.size(-1)
    flat_scores = masked_scores.reshape(-1, num_candidates) # [T, num_candidates]

    # [T, num_candidates]
    valid_mask = ~torch.isneginf(flat_scores)
    valid_counts = valid_mask.sum(dim=-1) # [T]
    max_valid = int(valid_counts.max().item())

    pad = pad_value.to(masked_scores.device)
    # [T, max_valid]
    truncated = torch.full(
        (flat_scores.size(0), max_valid),
        pad,
        dtype=masked_scores.dtype,
        device=masked_scores.device,
    )

    if max_valid > 0:
        # valid tokens appeared before the current position: [T, num_candidates]
        valid_ranks = valid_mask.cumsum(dim=-1).sub_(1)
        # row indices for valid tokens: [total_valid_tokens]
        valid_rows, _ = valid_mask.nonzero(as_tuple=True)
        # 1-D tensor of True indices: [total_valid_tokens]
        # per-row packed column positions for kept token
        truncated[valid_rows, valid_ranks[valid_mask]] = flat_scores[valid_mask]

    # [T, max_valid]
    return truncated.view(*masked_scores.shape[:-1], max_valid)


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Normalizes scores along the last axis.
    p_i = e^x_i / sum_j e^x_j => log(p_i) = x_i - log(sum_j e^x_j)
    """
    return scores - torch.logsumexp(scores, dim=-1, keepdim=True)


def apply_temperature(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling of scores."""
    return scores if temperature == 1.0 else (scores / temperature)


def apply_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Applies top-k filtering."""
    vocab_dim = scores.size(-1)
    if top_k <= 0 or top_k >= vocab_dim:
        # [T, num_candidates]
        return scores

    # [T, top_k]
    topk_scores, _ = torch.topk(scores, k=top_k, dim=-1)
    # [T] => [T, 1]
    min_values = topk_scores[..., -1].unsqueeze(-1)
    masked_scores = torch.where(scores < min_values, torch.tensor(float('-inf')).to(scores.device), scores)
    return masked_scores


def apply_top_p(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Applies nucleus filtering."""
    if top_p >= 1.0:
        # [T, num_candidates]
        return scores
    
    # [T, num_candidates]
    sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # [T, num_candidates] bool indices in the space of sorted scores
    sorted_indices_to_remove = cumulative_probs > top_p
    # right shift to keep the first token that exceeds the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    mask = torch.zeros_like(scores, dtype=torch.bool)
    # write sorted_indices_to_remove to the original indices in mask
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    masked_scores = scores.clone()
    masked_scores[mask] = float('-inf')

    return masked_scores


def extract_valid_scores(
    scores: torch.Tensor,
    top_p: float | None = None,
    top_k: int | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Applies temperature / top-k / top-p to scores.
    
    Args:
        scores: Tensor of shape [T, num_candidates] containing the raw scores for each candidate token.
        top_p: Nucleus filtering probability.
        top_k: Top-k filtering number.
        temperature: Temperature for scaling scores.
    
    Returns:
        Tensor of shape [T, max_valid] containing processed and normalized scores of valid tokens.
    """
    processed_scores = apply_temperature(scores, temperature=temperature)

    if top_k is not None:
        processed_scores = apply_top_k(processed_scores, top_k=top_k)

    if top_p is not None:
        processed_scores = apply_top_p(processed_scores, top_p=top_p)

    final_scores = normalize_scores(processed_scores)
    truncated_scores = _truncate_scores(final_scores)
    return truncated_scores
