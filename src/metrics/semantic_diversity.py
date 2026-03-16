from collections.abc import Sequence

import torch
from sentence_transformers import SentenceTransformer

from .semantic_metrics import average_pairwise_cosine_similarity


@torch.inference_mode()
def calculate_semantic_diversity(
    responses: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Computes response semantic diversity from average pairwise cosine similarity,
    where diversity is defined as `1 - average_pairwise_similarity`.

    Args:
        responses: Model responses to compare.
        model: SentenceTransformer embedding model.
        batch_size: Embedding micro-batch size.
        device: Optional device override.

    Returns:
        Scalar tensor representing the semantic diversity.
    """
    resolved_device = str(device) if device is not None else None

    embeddings = model.encode(
        list(responses),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True, # pre-normalize for cosine similarity
        show_progress_bar=True,
        device=resolved_device,
    )
    average_similarity = average_pairwise_cosine_similarity(embeddings)
    semantic_diversity = 1.0 - average_similarity

    return semantic_diversity
