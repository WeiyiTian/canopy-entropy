import torch
import torch.nn.functional as F


def average_pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean cosine similarity across all unique embedding pairs.

    Args:
        embeddings: Tensor of shape [N, D].

    Returns:
        Scalar tensor, average pairwise cosine similarity.
    """
    num_embeddings = embeddings.shape[0]

    normalized_embeddings = F.normalize(embeddings, p=2, dim=-1) # [N, D]
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T # [N, N]
    # [2, N*(N-1)/2]: indices of upper triangle (excluding diagonal)
    pair_indices = torch.triu_indices(
        row=num_embeddings,
        col=num_embeddings,
        offset=1, # exclude self-pairs
        device=embeddings.device,
    )
    # mean similarity across all unique pairs (i, j)
    return similarity_matrix[pair_indices[0], pair_indices[1]].mean()
