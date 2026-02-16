import torch


def _truncate_logits(masked_logits: torch.Tensor, pad_value=torch.tensor(float("-inf"))) -> torch.Tensor:
    """Truncate logits by removing -inf positions and right-padding with pad_value."""
    vocab_dim = masked_logits.size(-1)
    flat_logits = masked_logits.reshape(-1, vocab_dim) # [S, vocab_dim]

    # [S, vocab_dim]
    valid_mask = ~torch.isneginf(flat_logits)
    valid_counts = valid_mask.sum(dim=-1) # [S]
    max_valid = int(valid_counts.max().item())

    pad = pad_value.to(masked_logits.device)
    # [S, max_valid]
    truncated = torch.full(
        (flat_logits.size(0), max_valid),
        pad,
        dtype=masked_logits.dtype,
        device=masked_logits.device,
    )

    if max_valid > 0:
        # valid tokens appeared before the current position: [S, vocab_dim]
        valid_ranks = valid_mask.cumsum(dim=-1).sub_(1)
        # row indices for valid tokens: [total_valid_tokens]
        valid_rows, _ = valid_mask.nonzero(as_tuple=True)
        # 1-D tensor of True indices: [total_valid_tokens]
        # per-row packed column positions for kept token
        truncated[valid_rows, valid_ranks[valid_mask]] = flat_logits[valid_mask]

    # [S, max_valid]
    return truncated.view(*masked_logits.shape[:-1], max_valid)


def normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits along the last axis.
    p_i = e^x_i / sum_j e^x_j => log(p_i) = x_i - log(sum_j e^x_j)
    """
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling of logits."""
    return logits if temperature == 1.0 else (logits / temperature)


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Apply top-k filtering."""
    vocab_dim = logits.size(-1)
    if top_k <= 0 or top_k >= vocab_dim:
        # [S, num_candidates]
        return logits

    # [S, top_k]
    topk_logits, _ = torch.topk(logits, k=top_k, dim=-1)
    # [S] => [S, 1]
    min_values = topk_logits[..., -1].unsqueeze(-1)
    masked_logits = torch.where(logits < min_values, torch.tensor(float('-inf')).to(logits.device), logits)
    return masked_logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus filtering."""
    if top_p >= 1.0:
        # [S, num_candidates]
        return logits
    
    # [S, num_candidates]
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # [S, num_candidates] bool indices in the space of sorted logits
    sorted_indices_to_remove = cumulative_probs > top_p
    # right shift to keep the first token that exceeds the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    mask = torch.zeros_like(logits, dtype=torch.bool)
    # write sorted_indices_to_remove to the original indices in mask
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    masked_logits = logits.clone()
    masked_logits[mask] = float('-inf')

    return masked_logits


def extract_valid_logits(
    logits: torch.Tensor,
    top_p: float | None = None,
    top_k: int | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply temperature / top-k / top-p to logits.
    Input logits shape is expected to be (..., num_candidates).
    For vllm outputs, logits should be in shape (S, num_candidates)
    """
    processed_logits = apply_temperature(logits, temperature=temperature)

    if top_k is not None:
        processed_logits = apply_top_k(processed_logits, top_k=top_k)

    if top_p is not None:
        processed_logits = apply_top_p(processed_logits, top_p=top_p)

    final_logits = normalize_logits(processed_logits)
    truncated_logits = _truncate_logits(final_logits)
    return truncated_logits
