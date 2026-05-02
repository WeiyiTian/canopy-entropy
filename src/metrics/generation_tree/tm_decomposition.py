import math
from dataclasses import dataclass


@dataclass(slots=True)
class TMDecomposition:
    """
    Scalar decomposition `TM*_max = E[N] * E[R] + Cov(N, R)`, where `N` is
    generation length and `R = H_sum / N` is per-token entropy rate.

    Attributes:
        tm_star_max: the expected cumulative branching uncertainty.
        length_driven: `E[N] * E[R]`, the part of `TM*_max` explained by
            treating length and entropy rate as independent.
        length_entropy_rate_covariance: `Cov(N, R) = TM*_max - E[N] * E[R]`. Negative values
            indicate longer rollouts have lower entropy rate; positive values
            indicate longer rollouts are more information-rich per token.
    """

    tm_star_max: float
    length_driven: float
    length_entropy_rate_covariance: float

    @property
    def length_driven_share(self) -> float:
        """Fraction of `TM*_max` explained by the independent length and entropy rate term."""
        return self.length_driven / self.tm_star_max

    @property
    def length_entropy_rate_cov_share(self) -> float:
        """Fraction of `TM*_max` due to length and entropy rate covariance."""
        return self.length_entropy_rate_covariance / self.tm_star_max


def tm_decomposition_from_pooled(
    tm_star_max: float,
    gen_ppl: float,
    branching_factor: float,
) -> TMDecomposition:
    """
    Reconstructs the model level TM* decomposition from the pooled metrics.
    - `gen_ppl = exp(TM*_max / E[N])` => `E[N] = TM*_max / log(gen_ppl)`.
    - `branching_factor = exp(E[R])` => `E[R] = log(branching_factor)`.

    Args:
        tm_star_max: Unweighted per-prompt mean `TM*_max = (1/P) * sum_p tm_star_max^(p)`.
        gen_ppl: `exp(TM*_max / E[N])` where `E[N]` is the unweighted per-prompt
            mean generation length.
        branching_factor: `exp(E[R])` where `E[R]` is the unweighted per-prompt
            mean per-token entropy rate.

    Returns:
        `TMDecomposition` with `tm_star_max`, `length_driven = E[N] * E[R]`,
        and `length_entropy_rate_covariance = tm_star_max - length_driven`.
    """
    mean_length = tm_star_max / math.log(gen_ppl)
    mean_entropy_rate = math.log(branching_factor)
    length_driven = mean_length * mean_entropy_rate
    length_entropy_rate_covariance = tm_star_max - length_driven
    return TMDecomposition(
        tm_star_max=tm_star_max,
        length_driven=length_driven,
        length_entropy_rate_covariance=length_entropy_rate_covariance,
    )
