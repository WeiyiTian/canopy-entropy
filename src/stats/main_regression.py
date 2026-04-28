from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf


@dataclass(slots=True)
class MainRegressionResult:
    """
    Estimated coefficients and inference for the main regression:
    D_{tmp} = alpha + beta * R_bar_{tmp} + gamma * N_bar_{tmp}
        + delta_m + lambda_t + u_p + eps_{tmp}

    where `delta_m` is the model fixed effect (combining `model_name` and
    `model_variant`), `lambda_t` is the task fixed effect, and `u_p` is the
    prompt-level random intercept.

    Attributes:
        beta: Estimated entropy-rate coefficient `beta`.
        beta_stderr: Standard error of `beta`.
        beta_pvalue: Two-sided p-value for `H_0: beta = 0`.
        beta_ci_low: Lower bound of the 95% confidence interval for `beta`.
        beta_ci_high: Upper bound of the 95% confidence interval for `beta`.
        gamma: Estimated length coefficient `gamma`.
        gamma_stderr: Standard error of `gamma`.
        n_obs: Number of (task, prompt, model, variant) panel rows used in the fit.
        n_prompts: Number of unique prompts.
        formula: Pasty formula string used for the fit.
        summary: Full statsmodels summary text for diagnostics.
    """

    beta: float
    beta_stderr: float
    beta_pvalue: float
    beta_ci_low: float
    beta_ci_high: float
    gamma: float
    gamma_stderr: float
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
    Fits the main regression as a mixed-effects model:
    D_{tmp} = alpha + beta * R_bar_{tmp} + gamma * N_bar_{tmp}
        + delta_m + lambda_t + u_p + eps_{tmp}

    Args:
        panel: DataFrame with one row per (model, variant, dataset, prompt)
            with columns:`D`, `R_bar`, `N_bar`, `task`, `model_name`,
            `model_variant`, `prompt_id`.
        formula: Patsy formula for the fixed-effects part. The default treats
            `(model_name, model_variant)` as a single composite identifier via
            their interaction, matching `delta_m`.
        groups_col: Column whose values define the random-intercept groups
            `u_p`. Defaults to `prompt_uid`, the globally unique prompt identifier
            across runs.

    Returns:
        `MainRegressionResult` with the `beta` estimate, its inference, and
        the full statsmodels summary.
    """
    required_cols = {"D", "R_bar", "N_bar", "task", "model_name", "model_variant", groups_col}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    model = smf.mixedlm(formula, data=panel, groups=panel[groups_col])
    result = model.fit(reml=True)

    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    conf_int = result.conf_int(alpha=0.05)

    return MainRegressionResult(
        beta=float(params["R_bar"]),
        beta_stderr=float(bse["R_bar"]),
        beta_pvalue=float(pvalues["R_bar"]),
        beta_ci_low=float(conf_int.loc["R_bar", 0]),
        beta_ci_high=float(conf_int.loc["R_bar", 1]),
        gamma=float(params["N_bar"]),
        gamma_stderr=float(bse["N_bar"]),
        n_obs=int(result.nobs),
        n_prompts=int(panel[groups_col].nunique()),
        formula=formula,
        summary=str(result.summary()),
    )
