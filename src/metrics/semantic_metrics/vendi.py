import torch


def calculate_vendi_entropy(normalized_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes the Vendi entropy of a prompt's rollout embeddings.

    Args:
        normalized_embeddings: L2-normalized tensor of shape [M, D], one row per rollout.

    Returns:
        Scalar tensor containing the Vendi entropy, i.e., log of the Vendi Score H = log(VS).
        `H = -sum_i lambda_i * log(lambda_i)`, where `lambda_i` are the eigenvalues of the 
        normalized cosine kernel matrix `K / M`.

    Notes:
        The Vendi Score is `VS = exp(H)`, the effective number of distinct responses among
        the M rollouts, bounded in `[1, M]`. 

        For L2-normalized rows, `K = E @ E^T` is the cosine kernel with unit diagonal, so
        `tr(K / M) = 1` and the eigenvalues of `K / M` sum to 1.
    """
    num_embeddings = normalized_embeddings.shape[0] # M
    embeddings = normalized_embeddings.double()
    kernel = embeddings @ embeddings.T # smilarity matrix K = [M, M]

    eigenvalues = torch.linalg.eigvalsh(kernel / num_embeddings).clamp_min(0.0) # [M]
    eigenvalues = eigenvalues[eigenvalues > 0]

    entropy = -(eigenvalues * eigenvalues.log()).sum()
    return entropy.to(normalized_embeddings.dtype)
