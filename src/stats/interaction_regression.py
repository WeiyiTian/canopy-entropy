from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from .coefficient import CoefficientStats


@dataclass(slots=True)
class InteractionRegressionResult:
    """
    Estimated coefficients and inference for the interaction regression:
    D_{tmp} = alpha + beta * R_bar_{tmp} + tau * 1{m=FT}
        + eta * (R_bar_{tmp} * 1{m=FT}) + gamma * N_bar_{tmp}
        + lambda_t + delta_family + u_p + eps_{tmp}

    `lambda_t` is the task fixed effect, `delta_family` is the model-family
    fixed effect, and `u_p` is the prompt-level random intercept.

    `H_0: eta = 0`: Asks whether fine-tuning changes the slope of `D` on `R_bar`.
    Rejecting it means the FT slope `(beta + eta)` differs from the base slope `beta`.

    Attributes:
        beta: `CoefficientStats` for the base entropy rate (`R_bar`) coefficient `beta`.
        tau: `CoefficientStats` for the FT intercept offset `tau` relative to the base.
        eta: `CoefficientStats` for the change in entropy rate (`R_bar`) slope coefficient 
            from base to FT.
        gamma: `CoefficientStats` for the length (`N_bar`) coefficient `gamma`.
        n_obs: Number of (task, prompt, model, variant) panel rows used in the fit.
        n_prompts: Number of unique prompts.
        formula: Pasty formula string used for the fit.
        summary: Full statsmodels summary text for diagnostics.
    
    Notes:
        R_bar * C(model_variant)
            = R_bar + C(model_variant) + R_bar:C(model_variant)

        D ~ R_bar * C(model_variant) + N_bar + C(task) + C(model_name)
            = D ~ R_bar + C(model_variant) + R_bar:C(model_variant) + N_bar + C(task) + C(model_name)
        
        D = α + β · R_bar + τ * 1{m=FT} + η · R_bar * 1{m=FT} + γ · N_bar + λ_t + δ_family
        - 1{m=FT}=1: D = (α + τ) + (β + η) · R_bar + γ · N_bar + λ_t + δ_family
        - 1{m=FT}=0: D = α + β · R_bar + γ · N_bar + λ_t + δ_family
    """

    beta: CoefficientStats
    tau: CoefficientStats
    eta: CoefficientStats
    gamma: CoefficientStats
    n_obs: int
    n_prompts: int
    formula: str
    summary: str


def fit_interaction_regression(
    panel: pd.DataFrame,
    formula: str = (
        "D ~ R_bar * C(model_variant, Treatment(reference='base')) + N_bar + C(task) + C(model_name)"
    ),
    groups_col: str = "prompt_uid",
    ft_variant: str = "instruct",
) -> InteractionRegressionResult:
    """
    Fits the interaction regression as a REML mixed-effects model:
    D_{tmp} = alpha + beta * R_bar_{tmp} + tau * 1{m=FT}
        + eta * (R_bar_{tmp} * 1{m=FT}) + gamma * N_bar_{tmp}
        + lambda_t + delta_family + u_p + eps_{tmp}

    Args:
        panel: DataFrame with one row per (model, variant, dataset, prompt)
            with columns:`D`, `R_bar`, `N_bar`, `task`, `model_name`,
            `model_variant`, plus `prompt_uid`.
        formula: Patsy formula for the fixed-effects part. The default
            interacts `R_bar` with `C(model_variant, Treatment(reference='base'))`
            and adds a model family fixed effect `delta_family`.
        groups_col: Column defining the random-intercept groups `u_p`.
            Defaults to `prompt_uid` (globally unique prompt identifier across runs).
        ft_variant: The non-reference model_variant value identifying the fine-tuned variant.

    Returns:
        `InteractionRegressionResult` with the `beta`, `tau`, `eta`, and
        `gamma` coefficient stats, sample sizes, and the full statsmodels summary.
    """
    required_cols = {"D", "R_bar", "N_bar", "task", "model_name", "model_variant", groups_col}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    model = smf.mixedlm(formula, data=panel, groups=panel[groups_col])
    result = model.fit(reml=True)

    # treatment coding
    ft_indicator = f"C(model_variant, Treatment(reference='base'))[T.{ft_variant}]"
    return InteractionRegressionResult(
        beta=CoefficientStats.from_fit(result, "R_bar"),
        tau=CoefficientStats.from_fit(result, ft_indicator),
        eta=CoefficientStats.from_fit(result, f"R_bar:{ft_indicator}"),
        gamma=CoefficientStats.from_fit(result, "N_bar"),
        n_obs=int(result.nobs),
        n_prompts=int(panel[groups_col].nunique()),
        formula=formula,
        summary=str(result.summary()),
    )
