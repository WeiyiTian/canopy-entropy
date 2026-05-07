from dataclasses import dataclass

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


@dataclass(slots=True)
class BetaInteractionRegressionResult:
    """
    Estimated coefficients and inference for the beta interaction regression:
    logit(E[D_{tmp}]) = alpha + beta * R_sc_{tmp} + tau * 1{m=FT}
        + eta * (R_sc_{tmp} * 1{m=FT}) + f(invN_sc_{tmp}; m)
        + lambda_t + u_p + v_family + eps_{tmp}

    `R_sc` is the standardized entropy rate, `invN_sc` is standardized
    inverse length, and `f(invN_sc; m)` is a natural cubic spline interacted
    with model variant. `lambda_t` is the task fixed effect, `u_p` is the
    prompt-level random intercept, and `v_family` is the model-family random
    intercept.

    `H_0: eta = 0`: Asks whether fine-tuning changes the logit-scale slope of
    `D` on `R_bar`. Rejecting it means the FT slope `(beta + eta)` differs
    from the base slope `beta`.

    Attributes:
        coefficients: DataFrame indexed by R term name with columns
            `Estimate`, `Std. Error`, `z value`, and `Pr(>|z|)` from
            `summary(fit)$coefficients$cond`. Key rows are `R_sc` for `beta`,
            `model_variant<FT>` for `tau`, and `R_sc:model_variant<FT>` for `eta`.
        summary: Full `glmmTMB` summary text for diagnostics.
        n_obs: Panel rows used in the fit.
        n_prompts: Number of unique prompts.

    Notes:
        - `D` is clipped to `[1e-6, 1 - 1e-6]` before fitting so the beta
          likelihood is defined at the boundaries.
        - The beta precision uses its own log-link submodel:
          `model_variant * task + ns(invN_sc, df=3) + model_name`.
        - Length enters the mean through the spline term, so there is no single
          `gamma` coefficient as in the linear mixed-effects regressions.
    """

    coefficients: pd.DataFrame
    summary: str
    n_obs: int
    n_prompts: int


def fit_beta_interaction_regression(panel: pd.DataFrame) -> BetaInteractionRegressionResult:
    """
    Fits the beta interaction regression via `glmmTMB`:
    logit(E[D_{tmp}]) = alpha + beta * R_sc_{tmp} + tau * 1{m=FT}
        + eta * (R_sc_{tmp} * 1{m=FT}) + f(invN_sc_{tmp}; m)
        + lambda_t + u_p + v_family + eps_{tmp}

    Args:
        panel: DataFrame with one row per (model, variant, task, prompt)
            and columns `D`, `R_bar`, `N_bar`, `task`, `model_name`,
            `model_variant`, and `prompt_uid`.

    Returns:
        `BetaInteractionRegressionResult` with the conditional fixed-effect
        coefficient table, sample sizes, and the full `glmmTMB` summary.
    """
    ro.r("suppressPackageStartupMessages({ library(glmmTMB); library(splines) })")

    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df"] = ro.conversion.get_conversion().py2rpy(panel)

    ro.r("""
        df$D <- abs(df$D)
        df$task <- relevel(factor(df$task), ref = "completion")
        df$model_variant <- relevel(factor(df$model_variant), ref = "base")
        df$prompt_uid <- factor(df$prompt_uid)
        df$model_name <- factor(df$model_name)
        df$R_sc <- as.numeric(scale(df$R_bar))
        df$invN_sc <- as.numeric(scale(1 / df$N_bar))
        df$D_beta <- pmin(pmax(df$D, 1e-6), 1 - 1e-6)

        fit <- glmmTMB(
            D_beta ~ R_sc * model_variant + ns(invN_sc, df = 4) * model_variant
                + task + (1 | prompt_uid) + (1 | model_name),
            dispformula = ~ model_variant * task + ns(invN_sc, df = 3) + model_name,
            family = beta_family(link = "logit"),
            data = df
        )
    """)

    with (ro.default_converter + pandas2ri.converter).context():
        coef_df = ro.conversion.get_conversion().rpy2py(
            ro.r("as.data.frame(summary(fit)$coefficients$cond)")
        )

    return BetaInteractionRegressionResult(
        coefficients=coef_df,
        summary="\n".join(ro.r("capture.output(summary(fit))")),
        n_obs=int(ro.r("nobs(fit)")[0]),
        n_prompts=int(panel["prompt_uid"].nunique()),
    )
