library(glmmTMB)
library(DHARMa)
library(splines)

df <- read.csv("./panel.csv")

# ============================================================
# Preprocessing
# ============================================================

df$D <- abs(df$D)

df$task <- factor(df$task)
df$prompt_uid <- factor(df$prompt_uid)
df$model_name <- factor(df$model_name)
df$model_variant <- factor(df$model_variant)

df$model_variant <- relevel(df$model_variant, ref = "base")

# Matched by prefix so few-shot panels (completion_fewshot, ...) keep the same reference task.
task_ref <- grep("^completion", levels(df$task), value = TRUE)
if (length(task_ref) != 1) {
  stop(sprintf(
    "expected exactly one task level starting with 'completion', found: %s",
    paste(levels(df$task), collapse = ", ")
  ))
}
df$task <- relevel(df$task, ref = task_ref)

df$R_sc <- as.numeric(scale(df$R_bar))
df$invN_sc <- as.numeric(scale(1 / df$N_bar))

eps <- 1e-6
df$D_beta <- pmin(pmax(df$D, eps), 1 - eps)

# new

fit_beta_disp_R_interaction <- glmmTMB(
  
  D_beta ~
    R_sc * model_variant +
    ns(invN_sc, df = 4) * model_variant +
    task +
    (1 | prompt_uid) +
    (1 | model_name),
  
  dispformula = ~
    model_variant * task +
    ns(invN_sc, df = 3) +
    R_sc * model_variant +
    model_name,
  
  family = beta_family(link = "logit"),
  
  data = df
)

summary(fit_beta_disp_R_interaction)

sim_beta_R_int <- simulateResiduals(
  fit_beta_disp_R_interaction,
  n = 1000
)

plot(sim_beta_R_int)

testUniformity(sim_beta_R_int)
testDispersion(sim_beta_R_int)
testOutliers(sim_beta_R_int)