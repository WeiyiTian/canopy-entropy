from __future__ import annotations
from dataclasses import dataclass

import torch

from src.constants import LENGTH_BUCKET_NAMES

from .generation_tree.branching_factor import calculate_branching_factor, calculate_rollout_entropy_rate
from .generation_tree.gen_ppl import calculate_gen_ppl, calculate_tm_star_max
from .length_correlation import LengthCorrelation, calculate_length_correlation
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
        entropy_rate_vs_length: `LengthCorrelation` capturing co-movement between rollout
            length `N` and rollout entropy rate `r_N = H_sum / N`.
        truncation_rate: Fraction of rollouts with `N^(i) >= max_new_tokens`, i.e.
            the empirical `P(N >= T_max)`.
        semantic_diversity: 1 - average pairwise cosine similarity, equiavalent to
            `(1/M) * sum_i d^(i)`, where `d^(i) = 1 - (1/(M-1)) * sum_{j!=i} <e^(i), e^(j)>`
            is rollout i's mean dissimilarity to the other M-1 rollouts.
        semantic_diversity_vs_length: `LengthCorrelation` capturing co-movement between rollout
            length `N` and rollout semantic diversity `d`.
        bucketed_semantic_diversity: Mapping from each length-bucket name to `BucketStats`
            computed over the bucket's members. Buckets partition the M rollouts by
            sequence length.
    """

    tm_star_max: torch.Tensor
    gen_ppl: torch.Tensor
    branching_factor: torch.Tensor
    entropy_rate_vs_length: LengthCorrelation
    truncation_rate: torch.Tensor
    semantic_diversity: torch.Tensor
    semantic_diversity_vs_length: LengthCorrelation
    bucketed_semantic_diversity: dict[str, BucketStats]

    def to_cpu(self) -> PromptMetrics:
        return PromptMetrics(
            tm_star_max=self.tm_star_max.cpu(),
            gen_ppl=self.gen_ppl.cpu(),
            branching_factor=self.branching_factor.cpu(),
            entropy_rate_vs_length=self.entropy_rate_vs_length.to_cpu(),
            truncation_rate=self.truncation_rate.cpu(),
            semantic_diversity=self.semantic_diversity.cpu(),
            semantic_diversity_vs_length=self.semantic_diversity_vs_length.to_cpu(),
            bucketed_semantic_diversity={
                bucket_name: bucket_stats.to_cpu()
                for bucket_name, bucket_stats in self.bucketed_semantic_diversity.items()
            },
        )


def calculate_prompt_metrics(
    sequence_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
    normalized_embeddings: torch.Tensor | None,
    max_new_tokens: int,
) -> PromptMetrics:
    """
    Computes generation space metrics from M rollout-level tensors for one prompt.

    Args:
        sequence_entropy: Tensor of shape [M], where i-th entry is sum of step
            entropies of rollout i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is generated
            token length `N^(i)` of rollout i.
        normalized_embeddings: L2-normalized tensor of shape [M, D] aligned with
            rollout order, or `None` to skip embedding-based metrics (semantic
            diversity fields are populated with NaN).
        max_new_tokens: Generation length cap `T_max` used during sampling.

    Returns:
        `PromptMetrics`: Generation space metrics, including tm_star_max, gen_ppl,
        branching_factor, entropy_rate_vs_length co-movement, truncation_rate,
        semantic_diversity, semantic_diversity_vs_length co-movement, and
        bucketed_semantic_diversity.
    """
    tm_star_max, gen_ppl, branching_factor, entropy_rate_vs_length = calculate_tree_rollout_metrics(
        sequence_entropy, sequence_lengths
    )

    truncation_rate = (sequence_lengths >= max_new_tokens).to(torch.float32).mean()

    if normalized_embeddings is None:
        device = sequence_lengths.device
        nan = torch.tensor(torch.nan, device=device)
        semantic_diversity = nan
        semantic_diversity_vs_length = LengthCorrelation(
            pearson=nan, spearman=nan, kendall=nan, covariance=nan
        )
        bucketed_semantic_diversity = {
            bucket_name: BucketStats(
                average_similarity=nan,
                semantic_diversity=nan,
                num_responses=torch.tensor(0, device=device),
                min_length=nan,
                max_length=nan,
            )
            for bucket_name in LENGTH_BUCKET_NAMES
        }
    else:
        # [M], i-th entry is d^(i) = 1 - (1/(M-1)) * sum_{j!=i} <e^(i), e^(j)>
        rollout_semantic_diversity = calculate_rollout_semantic_diversity(normalized_embeddings)
        semantic_diversity = rollout_semantic_diversity.mean()
        semantic_diversity_vs_length = calculate_length_correlation(
            sequence_lengths, rollout_semantic_diversity
        )
        bucketed_semantic_diversity = calculate_bucketed_semantic_diversity(
            normalized_embeddings=normalized_embeddings,
            sequence_lengths=sequence_lengths,
        )

    return PromptMetrics(
        tm_star_max=tm_star_max,
        gen_ppl=gen_ppl,
        branching_factor=branching_factor,
        entropy_rate_vs_length=entropy_rate_vs_length,
        truncation_rate=truncation_rate,
        semantic_diversity=semantic_diversity,
        semantic_diversity_vs_length=semantic_diversity_vs_length,
        bucketed_semantic_diversity=bucketed_semantic_diversity,
    )


def calculate_tree_rollout_metrics(
    sequence_entropy: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LengthCorrelation]:
    """
    Computes the generation-tree scalar metrics from M rollout-level tensors.

    Args:
        sequence_entropy: Tensor of shape [M], where i-th entry is sum of step
            entropies of rollout i, i.e., `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        sequence_lengths: Tensor of shape [M], where i-th entry is generated
            token length `N^(i)` of rollout i.

    Returns:
        Tuple of (tm_star_max, gen_ppl, branching_factor, entropy_rate_vs_length):
        - tm_star_max: `(1/M) * sum_i H^(i)_sum`,
            where `H^(i)_sum = sum_t H(Y^(i)_t | X, y^(i)_<t)`.
        - gen_ppl: `exp(tm_star_max / E[N])`, where N is the generation length.
        - branching_factor: `exp((1/M) * sum_i r^(i)_N)`, where `r^(i)_N = H^(i)_sum / N^(i)`.
        - entropy_rate_vs_length: `LengthCorrelation` capturing co-movement between `N` and `r_N`.
    """
    tm_star_max = calculate_tm_star_max(sequence_entropy)
    gen_ppl = calculate_gen_ppl(tm_star_max, sequence_lengths)
    branching_factor = calculate_branching_factor(sequence_entropy, sequence_lengths)

    # [M], i-th entry is r^(i)_N = H^(i)_sum / N^(i)
    rollout_entropy_rate = calculate_rollout_entropy_rate(sequence_entropy, sequence_lengths)
    entropy_rate_vs_length = calculate_length_correlation(sequence_lengths, rollout_entropy_rate)

    return tm_star_max, gen_ppl, branching_factor, entropy_rate_vs_length
