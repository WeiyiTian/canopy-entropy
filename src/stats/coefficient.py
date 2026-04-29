from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class CoefficientStats:
    """
    Point estimate and inference for one regression coefficient.

    Attributes:
        estimate: Point estimate of the coefficient.
        stderr: Standard error of the estimate.
        pvalue: Two-sided p-value for `H_0: coefficient = 0`.
        ci_low: Lower bound of the 95% confidence interval.
        ci_high: Upper bound of the 95% confidence interval.
    """

    estimate: float
    stderr: float
    pvalue: float
    ci_low: float
    ci_high: float

    @classmethod
    def from_fit(cls, fit_result, term: str) -> CoefficientStats:
        """
        Extracts coefficient stats for `term` from a fitted statsmodels result.

        Args:
            fit_result: A fitted statsmodels regression results object exposing:
                - attributes `params`, `bse`, `pvalues`: pandas Series indexed by coefficient names
                - method `conf_int(alpha)`: returns a pandas DataFrame of confidence intervals
            term: Name of the coefficient (Patsy term) indexing the fitted model's parameters
        
        Returns:
            `CoefficientStats` with the estimate, standard error, p-value, and 95% confidence interval
        """
        conf_int = fit_result.conf_int(alpha=0.05).loc[term]
        return cls(
            estimate=float(fit_result.params[term]),
            stderr=float(fit_result.bse[term]),
            pvalue=float(fit_result.pvalues[term]),
            ci_low=float(conf_int[0]),
            ci_high=float(conf_int[1]),
        )
