from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

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

    def to_cpu(self) -> BucketStats:
        """Returns a new object with all tensors moved to CPU."""
        return BucketStats(**{f.name: getattr(self, f.name).cpu() for f in fields(self)})


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
        bucket_name: _calc_bucket_stats(
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


def calculate_rollout_semantic_diversity(
    normalized_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Computes per-rollout semantic diversity relative to the other rollouts.

    Args:
        normalized_embeddings: L2-normalized tensor of shape [M, D], one row per rollout.

    Returns:
        Tensor of shape [M]: i-th entry is the semantic diversity of rollout i 
        relative to the other M-1 rollouts, computed as `d^(i) = 1 - (1/(M-1)) * sum_{j!=i} <e_i, e_j>`,
        i.e., 1 - rollout i's mean cosine similarity to the other M-1 rollouts.

    Notes:
        `sum_{j!=i} <e_i, e_j> = sum_j <e_i, e_j> - <e_i, e_i> = sum_j <e_i, e_j> - 1`
        `sum_j <e_i, e_j> = <e_i, sum_j e_j> = <e_i, s>`
        `sum_{j!=i} <e_i, e_j> = <e_i, s> - 1 = e_i @ s - 1`
    """
    num_embeddings = normalized_embeddings.shape[0] # M
    embedding_sum = normalized_embeddings.sum(dim=0) # s = [D]
    self_excluded_dots = normalized_embeddings @ embedding_sum - 1.0 # [M]: sum_{j!=i} <e_i, e_j>
    return 1.0 - self_excluded_dots / (num_embeddings - 1)


def _average_pairwise_cosine_similarity(normalized_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Returns mean cosine similarity across all unique embedding pairs.

    Args:
        normalized_embeddings: L2-normalized tensor of shape [N, D].

    Returns:
        Scalar tensor with the mean cosine similarity over unique embedding pairs.
    
    Notes:
        ||sum_i{x_i}||^2 = sum_i{x_i ⋅ x_i} + 2 * sum_{i<j}{x_i ⋅ x_j} = N + 2 * sum_{i<j}{x_i ⋅ x_j}
        => sum_{i<j}{x_i ⋅ x_j} = (||sum_i{x_i}||^2 - N) / 2
    """
    num_embeddings = normalized_embeddings.shape[0]
    num_pairs = num_embeddings * (num_embeddings - 1) // 2

    embedding_sum = normalized_embeddings.sum(dim=0) # [D]: O(Nd)
    pair_sum = (embedding_sum.square().sum() - num_embeddings) / 2.0 # O(d)
    
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


def _calc_bucket_stats(
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
