from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from .coefficient import CoefficientStats


@dataclass(slots=True)
class QuadraticRobustnessResult:
    """
    Linearity robustness check for the interaction regression.
    Fits two ML mixed-effects models on a centered entropy rate `R_c = R_bar - mean(R_bar)`:

    Linear: D ~ R_c * 1{m=FT} + N_bar + C(task) + C(model_name) + (1|prompt)
    - R_c * 1{m=FT} = R_c + 1{m=FT} + R_c:1{m=FT}
    - base: D = intercept + beta * R_c + controls
    - FT: D = intercept + tau + (beta + eta) * R_c + controls

    Quadratic: D ~ R_c * 1{m=FT} + R_c^2 * 1{m=FT} + N_bar + C(task) + C(model_name) + (1|prompt)
    - R_c^2 * 1{m=FT} = R_c^2 + 1{m=FT} + R_c^2:1{m=FT}
    - base = intercept + beta * R_c + q * R_c^2 + controls
    - FT = intercept + tau + (beta + eta) * R_c + (q + k) * R_c^2 + controls

    The joint test H_0: coef(R_c^2) = 0 AND coef(R_c^2 * 1{m=FT}) = 0
    is reported as a likelihood-ratio test with 2 degrees of freedom. 

    Attributes:
        linear_beta: `R_c` slope from the linear ML fit.
        linear_eta: FT-specific change in `R_c` slope from the linear ML fit.
        quadratic_beta: `R_c` slope from the quadratic ML fit.
        quadratic_eta: FT-specific change in `R_c` slope from the quadratic ML fit.
        quadratic_q: `R_c^2` coefficient from the quadratic ML fit (curvature for base models).
        quadratic_k: `R_c^2 * 1{m=FT}` coefficient from the quadratic ML fit (extra curvature for FT models).
        lrt_statistic: 2 * (loglik_quadratic - loglik_linear).
        lrt_df: 2 (two added parameters: R_c^2 and R_c^2 * 1{m=FT}).
        lrt_pvalue: One-sided chi-square upper tail p-value.
        delta_beta: `quadratic_beta.estimate - linear_beta.estimate`.
        delta_eta: `quadratic_eta.estimate - linear_eta.estimate`.
        delta_beta_in_one_se: `|delta_beta| <= linear_beta.stderr`.
        delta_eta_in_one_se: `|delta_eta| <= linear_eta.stderr`.
        r_bar_mean: Mean of `R_bar` used to center `R_c`.
        n_obs: Number of (task, prompt, model, variant) panel rows used in both fits.
        n_prompts: Number of unique prompts (random intercept groups).
        linear_formula: Patsy formula string for the linear ML fit.
        quadratic_formula: Patsy formula string for the quadratic ML fit.
        linear_summary: Full statsmodels summary text for the linear ML fit.
        quadratic_summary: Full statsmodels summary text for the quadratic ML fit.
    
    Notes:
        - R is centered to `beta` interpretable as the slope at the mean R.
        - Inference uses ML instead of REML because REML log-likelihoods are not
            comparable across different fixed-effects structures.
    """

    linear_beta: CoefficientStats
    linear_eta: CoefficientStats
    quadratic_beta: CoefficientStats
    quadratic_eta: CoefficientStats
    quadratic_q: CoefficientStats
    quadratic_k: CoefficientStats
    lrt_statistic: float
    lrt_df: int
    lrt_pvalue: float
    delta_beta: float
    delta_eta: float
    delta_beta_in_one_se: bool
    delta_eta_in_one_se: bool
    r_bar_mean: float
    n_obs: int
    n_prompts: int
    linear_formula: str
    quadratic_formula: str
    linear_summary: str
    quadratic_summary: str


def fit_quadratic_robustness(
    panel: pd.DataFrame,
    groups_col: str = "prompt_uid",
    ft_variant: str = "instruct",
) -> QuadraticRobustnessResult:
    """
    Fits the linear and quadratic mixed-effects models for linearity robustness
    check and returns coefficients, LRT, and the linear-vs-quadratic deltas.

    - Linear: D ~ R_c * 1{m=FT} + N_bar + C(task) + C(model_name) + (1|prompt)
    - Quadratic: D ~ R_c * 1{m=FT} + R_c^2 * 1{m=FT} + N_bar + C(task) + C(model_name) + (1|prompt)

    Args:
        panel: DataFrame with one row per (model, variant, dataset, prompt) with columns
            `D`, `R_bar`, `N_bar`, `task`, `model_name`, `model_variant`, plus `groups_col`.
        groups_col: Column defining the random-intercept groups.
            Defaults to `prompt_uid` (globally unique prompt identifier across runs).
        ft_variant: The non-reference model_variant value identifying the fine-tuned variant.

    Returns:
        `QuadraticRobustnessResult` containing both fits, the joint LRT, and the deltas.
    """
    required_cols = {"D", "R_bar", "N_bar", "task", "model_name", "model_variant", groups_col}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    centered_panel = panel.copy()
    r_bar_mean = float(centered_panel["R_bar"].mean())
    centered_panel["R_c"] = centered_panel["R_bar"] - r_bar_mean

    # treatment coding
    ft_term = f"C(model_variant, Treatment(reference='base'))[T.{ft_variant}]"
    linear_formula = (
        f"D ~ R_c * C(model_variant, Treatment(reference='base'))"
        f" + N_bar + C(task) + C(model_name)"
    )
    quadratic_formula = (
        f"D ~ R_c * C(model_variant, Treatment(reference='base'))"
        f" + I(R_c**2) * C(model_variant, Treatment(reference='base'))"
        f" + N_bar + C(task) + C(model_name)"
    )

    groups = centered_panel[groups_col]
    linear_fit = smf.mixedlm(linear_formula, data=centered_panel, groups=groups).fit(reml=False)
    quadratic_fit = smf.mixedlm(quadratic_formula, data=centered_panel, groups=groups).fit(reml=False)

    lrt_statistic = float(2.0 * (quadratic_fit.llf - linear_fit.llf))
    lrt_df = 2
    # if the two curvature terms were actually useless, 
    # how often would we see an improvement this large or larger just from random noise
    lrt_pvalue = float(chi2.sf(max(lrt_statistic, 0.0), df=lrt_df))

    linear_beta = CoefficientStats.from_fit(linear_fit, "R_c")
    linear_eta = CoefficientStats.from_fit(linear_fit, f"R_c:{ft_term}")
    quadratic_beta = CoefficientStats.from_fit(quadratic_fit, "R_c")
    quadratic_eta = CoefficientStats.from_fit(quadratic_fit, f"R_c:{ft_term}")
    quadratic_q = CoefficientStats.from_fit(quadratic_fit, "I(R_c ** 2)")
    quadratic_k = CoefficientStats.from_fit(quadratic_fit, f"I(R_c ** 2):{ft_term}")

    delta_beta = quadratic_beta.estimate - linear_beta.estimate
    delta_eta = quadratic_eta.estimate - linear_eta.estimate

    return QuadraticRobustnessResult(
        linear_beta=linear_beta,
        linear_eta=linear_eta,
        quadratic_beta=quadratic_beta,
        quadratic_eta=quadratic_eta,
        quadratic_q=quadratic_q,
        quadratic_k=quadratic_k,
        lrt_statistic=lrt_statistic,
        lrt_df=lrt_df,
        lrt_pvalue=lrt_pvalue,
        delta_beta=float(delta_beta),
        delta_eta=float(delta_eta),
        delta_beta_in_one_se=bool(np.abs(delta_beta) <= linear_beta.stderr),
        delta_eta_in_one_se=bool(np.abs(delta_eta) <= linear_eta.stderr),
        r_bar_mean=r_bar_mean,
        n_obs=int(linear_fit.nobs),
        n_prompts=int(centered_panel[groups_col].nunique()),
        linear_formula=linear_formula,
        quadratic_formula=quadratic_formula,
        linear_summary=str(linear_fit.summary()),
        quadratic_summary=str(quadratic_fit.summary()),
    )
