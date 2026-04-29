from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from .coefficient import CoefficientStats


@dataclass(slots=True)
class MainRegressionResult:
    """
    Estimated coefficients and inference for the main regression:
    D_{tmp} = alpha + beta * R_bar_{tmp} + gamma * N_bar_{tmp}
        + delta_m + lambda_t + u_p + eps_{tmp}

    `delta_m` is the model fixed effect (combining `model_name` and
    `model_variant`), `lambda_t` is the task fixed effect, and `u_p` is the
    prompt-level random intercept.

    `H_0: beta = 0`: Asks whether mean per-rollout entropy rate `R_bar` predicts
    semantic diversity `D` after controlling for length, task, and model identity
    (combined `model_name`/`model_variant`).

    Attributes:
        beta: `CoefficientStats` for the entropy rate (`R_bar`) coefficient `beta`.
        gamma: `CoefficientStats` for the length (`N_bar`) coefficient `gamma`.
        n_obs: Number of (task, prompt, model, variant) panel rows used in the fit.
        n_prompts: Number of unique prompts.
        formula: Pasty formula string used for the fit.
        summary: Full statsmodels summary text for diagnostics.
    """

    beta: CoefficientStats
    gamma: CoefficientStats
    n_obs: int
    n_prompts: int
    formula: str
    summary: str


def fit_main_regression(
    panel: pd.DataFrame,
    formula: str = "D ~ R_bar + N_bar + C(task) + C(model_name) * C(model_variant)",
    groups_col: str = "prompt_uid",
) -> MainRegressionResult:
    """
    Fits the main regression as a REML mixed-effects model:
    D_{tmp} = alpha + beta * R_bar_{tmp} + gamma * N_bar_{tmp}
        + delta_m + lambda_t + u_p + eps_{tmp}

    Args:
        panel: DataFrame with one row per (model, variant, dataset, prompt)
            with columns:`D`, `R_bar`, `N_bar`, `task`, `model_name`,
            `model_variant`, plus `prompt_uid`.
        formula: Patsy formula for the fixed-effects part. The default treats
            `(model_name, model_variant)` as a single composite identifier via
            their interaction, matching `delta_m`.
        groups_col: Column defining the random-intercept groups `u_p`.
            Defaults to `prompt_uid` (globally unique prompt identifier across runs).

    Returns:
        `MainRegressionResult` with the `beta` and `gamma` coefficient stats,
        sample sizes, and the full statsmodels summary.
    """
    required_cols = {"D", "R_bar", "N_bar", "task", "model_name", "model_variant", groups_col}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    model = smf.mixedlm(formula, data=panel, groups=panel[groups_col])
    result = model.fit(reml=True)

    return MainRegressionResult(
        beta=CoefficientStats.from_fit(result, "R_bar"),
        gamma=CoefficientStats.from_fit(result, "N_bar"),
        n_obs=int(result.nobs),
        n_prompts=int(panel[groups_col].nunique()),
        formula=formula,
        summary=str(result.summary()),
    )
