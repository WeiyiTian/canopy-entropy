from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ...constants import LENGTH_BUCKET_NAMES


@dataclass(slots=True, frozen=True)
class BucketStats:
    """
    Semantic similarity summary for one response-length bucket.

    Attributes:
        average_similarity: Mean cosine similarity across unique response pairs.
        semantic_diversity: `1 - average_similarity` for the bucket.
        num_responses: Number of responses assigned to the bucket.
        min_length: Minimum response length observed in the bucket.
        max_length: Maximum response length observed in the bucket.
    """

    average_similarity: torch.Tensor
    semantic_diversity: torch.Tensor
    num_responses: torch.Tensor
    min_length: torch.Tensor
    max_length: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Serialize bucket statistics to a mapping of scalar tensors."""
        return {
            "average_similarity": self.average_similarity,
            "semantic_diversity": self.semantic_diversity,
            "num_responses": self.num_responses,
            "min_length": self.min_length,
            "max_length": self.max_length,
        }


def calculate_bucketed_semantic_diversity(
    normalized_embeddings: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> dict[str, BucketStats]:
    """
    Computes bucketed semantic diversity from pre-normalized response embeddings.

    Args:
        normalized_embeddings: L2-normalized tensor of shape [N, D].
        sequence_lengths: Response length tensor of shape [N].

    Returns:
        Dictionary from each entry in `LENGTH_BUCKET_NAMES` to its `BucketStats`.
    """
    sequence_lengths = sequence_lengths.to(device=normalized_embeddings.device)
    bucket_ids = _rank_partition_bucket_ids(sequence_lengths)

    return {
        bucket_name: _bucket_stats(
            normalized_embeddings=normalized_embeddings,
            sequence_lengths=sequence_lengths,
            # indices of all elements that belong to this bucket
            member_indices=(bucket_ids == bucket_id).nonzero(as_tuple=True)[0],
        )
        for bucket_id, bucket_name in enumerate(LENGTH_BUCKET_NAMES)
    }


def stack_semantic_diversity_results(
    results: Sequence[dict[str, BucketStats]],
) -> dict[str, BucketStats]:
    """
    Stacks prompt-level semantic diversity results into one pooled tensor view.

    Args:
        results: Sequence of bucketed semantic-diversity summaries.

    Returns:
        Dictionary from each entry in `LENGTH_BUCKET_NAMES` to its `BucketStats`,
        whose fields have shape [len(results)] after stacking.
    """
    return {
        bucket_name: BucketStats(
            average_similarity=torch.stack([result[bucket_name].average_similarity for result in results]),
            semantic_diversity=torch.stack([result[bucket_name].semantic_diversity for result in results]),
            num_responses=torch.stack([result[bucket_name].num_responses for result in results]),
            min_length=torch.stack([result[bucket_name].min_length for result in results]),
            max_length=torch.stack([result[bucket_name].max_length for result in results]),
        )
        for bucket_name in LENGTH_BUCKET_NAMES
    }


def _average_pairwise_cosine_similarity(normalized_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Returns mean cosine similarity across all unique embedding pairs.

    Args:
        normalized_embeddings: L2-normalized tensor of shape [N, D].

    Returns:
        Scalar tensor with the mean cosine similarity over unique embedding pairs.
    """
    num_embeddings = normalized_embeddings.shape[0]
    # [N, N]
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    # upper triangle (excluding diagonal)
    pair_sum = torch.triu(similarity_matrix, diagonal=1).sum()
    # N * (N-1) / 2
    num_pairs = num_embeddings * (num_embeddings - 1) // 2
    # mean similarity across all unique pairs (i, j)
    return pair_sum / num_pairs


def _rank_partition_bucket_ids(sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Assigns each response to an ordered length bucket by rank partitioning,
    The responses are sorted by length, then partitioned into contiguous rank 
    buckets of roughly equal size.

    Args:
        sequence_lengths: Tensor of shape [N] containing response lengths.

    Returns:
        Tensor of shape [N] with integer bucket ids in the range `[0, len(LENGTH_BUCKET_NAMES) - 1]`. 
    """
    num_lengths = int(sequence_lengths.numel())
    sorted_indices = torch.argsort(sequence_lengths, stable=True)
    ranks = torch.empty_like(sorted_indices)
    # ranks[i]: position of element i in the sorted order
    ranks[sorted_indices] = torch.arange(num_lengths, device=sequence_lengths.device)
    # rank → bucket index
    return torch.div(ranks * len(LENGTH_BUCKET_NAMES), num_lengths, rounding_mode="floor")


def _bucket_stats(
    normalized_embeddings: torch.Tensor,
    sequence_lengths: torch.Tensor,
    member_indices: torch.Tensor,
) -> BucketStats:
    """
    Computes semantic similarity statistics for one response-length bucket.

    Args:
        normalized_embeddings: L2-normalized tensor of shape [N, D].
        sequence_lengths: Tensor of shape [N] containing response lengths.
        member_indices: Long tensor of shape [N_bucket] containing indices assigned to the bucket.

    Returns:
        `BucketStats` containing information about the bucket.
    """
    device = normalized_embeddings.device
    nan_value = torch.tensor(torch.nan, device=device)

    bucket_lengths = sequence_lengths[member_indices]
    num_responses = int(member_indices.numel())
    num_pairs = num_responses * (num_responses - 1) // 2

    average_similarity = (
        _average_pairwise_cosine_similarity(normalized_embeddings[member_indices])
        if num_pairs > 0 else nan_value
    )
    min_length = bucket_lengths.min() if num_responses > 0 else nan_value
    max_length = bucket_lengths.max() if num_responses > 0 else nan_value

    return BucketStats(
        average_similarity=average_similarity,
        semantic_diversity=1.0 - average_similarity,
        num_responses=torch.tensor(num_responses, device=device),
        min_length=min_length,
        max_length=max_length,
    )
