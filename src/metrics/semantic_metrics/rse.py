from dataclasses import dataclass

import torch
from scipy.sparse.csgraph import connected_components


@dataclass(slots=True, frozen=True)
class RelaxedSemanticEntropyStats:
    """
    Relaxed Semantic Entropy (RSE) for one prompt's judged rollouts.

    Attributes:
        entropy: RSE `-sum_c (|c| / m) * log(|c| / m)` over clusters c.
        num_clusters: Number of semantic equivalence classes among the m rollouts.
        num_responses: Number of judged rollouts m.
    """

    entropy: torch.Tensor
    num_clusters: torch.Tensor
    num_responses: torch.Tensor

    @property
    def normalized_entropy(self) -> torch.Tensor:
        """
        Returns `entropy / log(m)`, bounded in `[0, 1]`.

        The ceiling `log(m)` depends on m, so raw `entropy` is only comparable across
        sets of equal size m.
        """
        return self.entropy / self.num_responses.log()


def calculate_relaxed_semantic_entropy(
    pairwise_judgments: torch.Tensor,
) -> RelaxedSemanticEntropyStats:
    """
    Computes RSE from a judge's pairwise similarity verdicts.

    Args:
        pairwise_judgments: Boolean-valued tensor of shape [m, m] where entry `[i, j]`
            is the judge's verdict on the ordered pair `(i, j)`. The diagonal is ignored.

    Returns:
        `RelaxedSemanticEntropyStats` with the entropy, the cluster count, and m.
    """
    cluster_labels = calculate_cluster_labels(pairwise_judgments)
    num_responses = cluster_labels.numel()

    cluster_sizes = torch.bincount(cluster_labels) # [num_clusters]
    cluster_probabilities = cluster_sizes / num_responses
    entropy = torch.special.entr(cluster_probabilities).sum() # entr(p) = -p * log(p)

    return RelaxedSemanticEntropyStats(
        entropy=entropy,
        num_clusters=torch.tensor(cluster_sizes.numel()),
        num_responses=torch.tensor(num_responses),
    )


def calculate_cluster_labels(pairwise_judgments: torch.Tensor) -> torch.Tensor:
    """
    Assigns each rollout to a semantic equivalence class.

    Args:
        pairwise_judgments: Boolean-valued tensor of shape [m, m] of ordered-pair verdicts.

    Returns:
        Tensor of shape [m] with cluster ids in `[0, num_clusters - 1]`.

    Notes:
        An edge requires agreement in both orderings, so the adjacency matrix is
        `J & J.T`. Unmatched rollouts form singletons.
    """
    if pairwise_judgments.ndim != 2 or pairwise_judgments.shape[0] != pairwise_judgments.shape[1]:
        raise ValueError(f"Expected a square [m, m] matrix, got {tuple(pairwise_judgments.shape)}.")

    verdicts = pairwise_judgments.bool().cpu().numpy()
    adjacency = verdicts & verdicts.T # reciprocal similarity only

    _, cluster_labels = connected_components(adjacency, directed=False)
    return torch.from_numpy(cluster_labels)
