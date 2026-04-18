from __future__ import annotations
from dataclasses import dataclass

import torch

from .generation_tree.branching_factor import calculate_branching_factor, calculate_rollout_entropy_rate
from .generation_tree.gen_ppl import calculate_gen_ppl, calculate_tm_star_max
from .semantic_metrics.cosine_similarity import (
    BucketStats,
    calculate_bucketed_semantic_diversity,
    calculate_rollout_semantic_diversity,
)


@dataclass(slots=True)
class PromptMetrics:
    """
    Generation space metrics computed from M rollout-level tensors for one prompt.

    Attributes:
        tm_star_max: `(1/M) * sum_i H^(i)_sum`,
            where `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        gen_ppl: `exp(tm_star_max / E[N])`, where N is the generation length.
        branching_factor: `exp((1/M) * sum_i r^(i)_N)`, where `r^(i)_N = H^(i)_sum / N^(i)`.
        entropy_rate_length_correlation: Pearson correlation `rho(N, r_N)` over `(N^(i), r^(i)_N)`.
        entropy_rate_length_covariance: Covariance `Cov(N, r_N)` over `(N^(i), r^(i)_N)`.
        semantic_diversity: 1 - average pairwise cosine similarity, equiavalent to
            `(1/M) * sum_i d^(i)`, where `d^(i) = 1 - (1/(M-1)) * sum_{j!=i} <e^(i), e^(j)>`
            is rollout i's mean dissimilarity to the other M-1 rollouts.
        semantic_diversity_length_correlation: Pearson correlation `rho(N, d)` over `(N^(i), d^(i))`.
        semantic_diversity_length_covariance: Covariance `Cov(N, d)` over `(N^(i), d^(i))`.
        bucketed_semantic_diversity: Mapping from each length-bucket name to `BucketStats`
            computed over the bucket's members. Buckets partition the M rollouts by
            sequence length.
    """

    tm_star_max: torch.Tensor
    gen_ppl: torch.Tensor
    branching_factor: torch.Tensor
    entropy_rate_length_correlation: torch.Tensor
    entropy_rate_length_covariance: torch.Tensor
    semantic_diversity: torch.Tensor
    semantic_diversity_length_correlation: torch.Tensor
    semantic_diversity_length_covariance: torch.Tensor
    bucketed_semantic_diversity: dict[str, BucketStats]

    def to_cpu(self) -> PromptMetrics:
        return PromptMetrics(
            tm_star_max=self.tm_star_max.cpu(),
            gen_ppl=self.gen_ppl.cpu(),
            branching_factor=self.branching_factor.cpu(),
            entropy_rate_length_correlation=self.entropy_rate_length_correlation.cpu(),
            entropy_rate_length_covariance=self.entropy_rate_length_covariance.cpu(),
            semantic_diversity=self.semantic_diversity.cpu(),
            semantic_diversity_length_correlation=self.semantic_diversity_length_correlation.cpu(),
            semantic_diversity_length_covariance=self.semantic_diversity_length_covariance.cpu(),
            bucketed_semantic_diversity={
                bucket_name: bucket_stats.to_cpu()
                for bucket_name, bucket_stats in self.bucketed_semantic_diversity.items()
            },
        )


def calculate_prompt_metrics(
    sequence_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
    normalized_embeddings: torch.Tensor,
) -> PromptMetrics:
    """
    Computes generation space metrics from M rollout-level tensors for one prompt.

    Args:
        sequence_entropy: Tensor of shape [M], where i-th entry is sum of step
            entropies of rollout i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is generated
            token length `N^(i)` of rollout i.
        normalized_embeddings: L2-normalized tensor of shape [M, D] aligned with
            rollout order.

    Returns:
        `PromptMetrics`: Generation space metrics, including tm_star_max,
        gen_ppl, branching_factor, entropy_rate_length_correlation/covariance,
        semantic_diversity, semantic_diversity_length_correlation/covariance,
        and bucketed_semantic_diversity.
    """
    sequence_lengths = sequence_lengths.to(dtype=torch.float32)

    (
        tm_star_max,
        gen_ppl,
        branching_factor,
        entropy_rate_length_correlation,
        entropy_rate_length_covariance,
    ) = calculate_tree_rollout_metrics(sequence_entropy, sequence_lengths)

    # [M], i-th entry is d^(i) = 1 - (1/(M-1)) * sum_{j!=i} <e^(i), e^(j)>
    rollout_semantic_diversity = calculate_rollout_semantic_diversity(normalized_embeddings)
    semantic_diversity = rollout_semantic_diversity.mean()
    semantic_diversity_length_correlation, semantic_diversity_length_covariance = (
        _length_correlation_covariance(sequence_lengths, rollout_semantic_diversity)
    )
    bucketed_semantic_diversity = calculate_bucketed_semantic_diversity(
        normalized_embeddings=normalized_embeddings,
        sequence_lengths=sequence_lengths,
    )

    return PromptMetrics(
        tm_star_max=tm_star_max,
        gen_ppl=gen_ppl,
        branching_factor=branching_factor,
        entropy_rate_length_correlation=entropy_rate_length_correlation,
        entropy_rate_length_covariance=entropy_rate_length_covariance,
        semantic_diversity=semantic_diversity,
        semantic_diversity_length_correlation=semantic_diversity_length_correlation,
        semantic_diversity_length_covariance=semantic_diversity_length_covariance,
        bucketed_semantic_diversity=bucketed_semantic_diversity,
    )


def calculate_tree_rollout_metrics(
    sequence_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the generation-tree scalar metrics from M rollout-level tensors.

    Args:
        sequence_entropy: Tensor of shape [M], where i-th entry is sum of step
            entropies of rollout i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is generated
            token length `N^(i)` of rollout i.

    Returns:
        Tuple of scalar tensors: (tm_star_max, gen_ppl, branching_factor, 
        entropy_rate_length_correlation, entropy_rate_length_covariance)
        - tm_star_max: `(1/M) * sum_i H^(i)_sum`, 
            where `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        - gen_ppl: `exp(tm_star_max / E[N])`, where N is the generation length.
        - branching_factor: `exp((1/M) * sum_i r^(i)_N)`, where `r^(i)_N = H^(i)_sum / N^(i)`.
        - entropy_rate_length_correlation: Pearson correlation `rho(N, r_N)` over `(N^(i), r^(i)_N)`.
        - entropy_rate_length_covariance: Covariance `Cov(N, r_N)` over `(N^(i), r^(i)_N)`.
    """
    sequence_lengths = sequence_lengths.to(dtype=torch.float32)

    tm_star_max = calculate_tm_star_max(sequence_entropy)
    gen_ppl = calculate_gen_ppl(tm_star_max, sequence_lengths)
    branching_factor = calculate_branching_factor(sequence_entropy, sequence_lengths)

    # [M], i-th entry is r^(i)_N = H^(i)_sum / N^(i)
    rollout_entropy_rate = calculate_rollout_entropy_rate(sequence_entropy, sequence_lengths)
    entropy_rate_length_correlation, entropy_rate_length_covariance = (
        _length_correlation_covariance(sequence_lengths, rollout_entropy_rate)
    )

    return (
        tm_star_max,
        gen_ppl,
        branching_factor,
        entropy_rate_length_correlation,
        entropy_rate_length_covariance,
    )


def _length_correlation_covariance(
    sequence_lengths: torch.Tensor,
    per_rollout_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Measures co-movement (correlation and covariance) between per-rollout generation 
    lengths and a per-rollout value of interest.

    Args:
        sequence_lengths: Tensor of shape [M] with rollout lengths `N^(i)`.
        per_rollout_values: Tensor of shape [M] with rollout values `z^(i)`.

    Returns:
        (correlation, covariance): Tuple of scalar tensors for `rho(N, z)` and
        `Cov(N, z)` over the M rollouts.
    """
    # [2, M]
    pairs = torch.stack([sequence_lengths, per_rollout_values], dim=0)
    covariance = torch.cov(pairs, correction=1)[0, 1]
    correlation = torch.corrcoef(pairs)[0, 1]

    return correlation, covariance