library(glmmTMB)
library(DHARMa)
library(splines)
library(dplyr)

# df <- read.csv("./panel_vendi_score.csv")
df <- read.csv("./topk1000_panel.csv")

# ============================================================
# Preprocessing
# ============================================================

df$task <- factor(df$task)
df$prompt_uid <- factor(df$prompt_uid)
df$model_name <- factor(df$model_name)
df$model_variant <- factor(df$model_variant)

df$model_variant <- relevel(df$model_variant, ref = "base")
df$task <- relevel(df$task, ref = "completion")

df$R_sc <- as.numeric(scale(df$R_bar))
df$invN_sc <- as.numeric(scale(1 / df$N_bar))

# ============================================================
# Vendi Score preprocessing
# ============================================================

# Use raw VS if available; otherwise reconstruct from log_VS
if ("VS" %in% names(df)) {
  df$VS_raw <- df$VS
} else if ("vendi_score" %in% names(df)) {
  df$VS_raw <- df$vendi_score
} else if ("Vendi_score" %in% names(df)) {
  df$VS_raw <- df$Vendi_score
} else if ("log_VS" %in% names(df)) {
  df$VS_raw <- exp(df$log_VS)
} else {
  stop("Cannot find VS column. Expected one of: VS, vendi_score, Vendi_score, log_VS.")
}

summary(df$VS_raw)

# ============================================================
# Transform VS from [1, M] to (0, 1)
# ============================================================

M <- 100   # change this if your number of sampled rollouts is not 100

df$VS_01_raw <- (df$VS_raw - 1) / (M - 1)

eps <- 1e-6
df$VS_beta <- pmin(pmax(df$VS_01_raw, eps), 1 - eps)

summary(df$VS_beta)

# ============================================================
# Beta regression on scaled Vendi Score
# Random model_name version
# ============================================================

fit_vendi_beta <- glmmTMB(
  
  VS_beta ~
    R_sc * model_variant +
    ns(invN_sc, df = 4) * model_variant +
    task +
    (1 | prompt_uid) +
    (1 | model_name),
  
  dispformula = ~
    model_variant * task +
    ns(invN_sc, df = 4) * model_variant +
    R_sc * model_variant + 
    model_name,
  
  family = beta_family(link = "logit"),
  
  data = df
)

summary(fit_vendi_beta)

sim_vendi_beta <- simulateResiduals(
  fittedModel = fit_vendi_beta,
  n = 1000
)

plot(sim_vendi_beta)

testUniformity(sim_vendi_beta)
testDispersion(sim_vendi_beta)
testOutliers(sim_vendi_beta)