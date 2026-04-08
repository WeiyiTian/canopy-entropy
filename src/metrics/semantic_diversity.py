from collections.abc import Sequence

import torch
from sentence_transformers import SentenceTransformer

from .semantic_metrics import (
    BucketStats,
    calculate_bucketed_semantic_diversity,
)


@torch.inference_mode()
def calculate_semantic_diversity_from_responses(
    responses: Sequence[str],
    model: SentenceTransformer,
    sequence_lengths: torch.Tensor,
    batch_size: int = 16,
) -> dict[str, BucketStats]:
    """
    Encodes responses and computes bucketed semantic diversity.

    Args:
        responses: Model responses to compare.
        model: SentenceTransformer embedding model.
        sequence_lengths: Tensor of shape [N].
        batch_size: Embedding micro-batch size.

    Returns:
        Mapping from length bucket name to per-bucket `BucketStats` containing
        semantic similarity, diversity, counts, and observed length ranges.
    """
    normalized_embeddings = model.encode(
        responses,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True, # pre-normalize for cosine similarity
        show_progress_bar=True,
    )
    return calculate_bucketed_semantic_diversity(
        normalized_embeddings=normalized_embeddings,
        sequence_lengths=sequence_lengths,
    )
