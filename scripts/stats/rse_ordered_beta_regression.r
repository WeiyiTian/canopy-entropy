# ============================================================
# Packages
# ============================================================

library(glmmTMB)
library(DHARMa)
library(splines)
library(dplyr)
library(ggplot2)

set.seed(20260802)

# ============================================================
# Load data
# ============================================================

df <- read.csv("./rse_panel.csv")

# ============================================================
# Outcome variables
# ============================================================

# Model 1 outcome:
# 1 = more than one semantic cluster
# 0 = exactly one semantic cluster
df$multi_cluster <- as.integer(df$n_clusters > 1L)

# Model 2 outcome:
# normalized semantic entropy, including exact 0 and 1
stopifnot(all(df$RSE_norm >= 0))
stopifnot(all(df$RSE_norm <= 1))

# ============================================================
# Factor variables
# ============================================================

df$task <- factor(df$task)
df$prompt_uid <- factor(df$prompt_uid)
df$model_name <- factor(df$model_name)
df$model_variant <- factor(df$model_variant)

df$model_variant <- relevel(
  df$model_variant,
  ref = "base"
)

df$task <- relevel(
  df$task,
  ref = "completion"
)

# Inspect model-family reference
levels(df$model_name)

# ============================================================
# Standardize uncertainty and inverse length
# ============================================================

df$R_sc <- as.numeric(
  scale(df$R_bar)
)

df$invN_sc <- as.numeric(
  scale(1 / df$N_bar)
)

# ============================================================
# Within-between decomposition by prompt
# ============================================================

df <- df %>%
  group_by(prompt_uid) %>%
  mutate(
    # Prompt-level average uncertainty
    R_between = mean(R_sc, na.rm = TRUE),
    
    # Deviation from prompt-average uncertainty
    R_within = R_sc - R_between,
    
    # Prompt-level average inverse length
    invN_between = mean(invN_sc, na.rm = TRUE),
    
    # Deviation from prompt-average inverse length
    invN_within = invN_sc - invN_between
  ) %>%
  ungroup()
# Within components should average approximately zero
# inside every prompt

decomposition_check <- df %>%
  group_by(prompt_uid) %>%
  summarise(
    mean_R_within = mean(R_within),
    mean_invN_within = mean(invN_within),
    .groups = "drop"
  )

max(abs(decomposition_check$mean_R_within))
max(abs(decomposition_check$mean_invN_within))

# These should both be close to zero
# Binary outcome
table(df$multi_cluster)
prop.table(table(df$multi_cluster))

table(
  df$model_variant,
  df$multi_cluster
)

xtabs(
  ~ task + model_variant + multi_cluster,
  data = df
)

xtabs(
  ~ model_name + model_variant + multi_cluster,
  data = df
)

# Ordered-beta outcome
summary(df$RSE_norm)

c(
  zero_rate = mean(df$RSE_norm == 0),
  one_rate = mean(abs(df$RSE_norm - 1) < 1e-12)
)

# ============================================================
# MODEL 
# Ordered-beta mixed-effects regression
# ============================================================

fit_rse_ordbeta <- glmmTMB(
  
  RSE_norm ~
    R_within * model_variant +
    R_between * model_variant +
    ns(invN_within, df = 3) * model_variant +
    invN_between +
    task +
    model_name +
    (1 | prompt_uid),
  
  # Conditional spread may differ across variants and tasks
  dispformula = ~
    model_variant * task +
    model_name,
  
  family = ordbeta(link = "logit"),
  
  data = df
)

summary(fit_rse_ordbeta)

# ============================================================
# Verify ordered-beta simulation
# ============================================================

set.seed(20260802)

ordbeta_simulation_check <- simulate(
  fit_rse_ordbeta,
  nsim = 3
)

# Should be FALSE for every simulation
sapply(
  ordbeta_simulation_check,
  function(y_sim) {
    isTRUE(
      all.equal(
        as.numeric(y_sim),
        df$RSE_norm
      )
    )
  }
)

# Simulations should also differ from each other
isTRUE(
  all.equal(
    ordbeta_simulation_check[[1]],
    ordbeta_simulation_check[[2]]
  )
)

# Check response range
sapply(
  ordbeta_simulation_check,
  range
)

# ============================================================
# DHARMa residual diagnostics: ordered-beta model
# ============================================================

set.seed(20260802)

sim_ordbeta <- simulateResiduals(
  fittedModel = fit_rse_ordbeta,
  n = 2000,
  refit = FALSE
)

plot(
  sim_ordbeta,
  quantreg = TRUE
)

testUniformity(sim_ordbeta)
testDispersion(sim_ordbeta)

# testOutliers(
#   sim_ordbeta,
#   type = "bootstrap",
#   nBoot = 1000
# )