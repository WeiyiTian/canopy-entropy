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
df$task <- relevel(df$task, ref = "completion")

df$R_sc <- as.numeric(scale(df$R_bar))
df$invN_sc <- as.numeric(scale(1 / df$N_bar))

eps <- 1e-6
df$D_beta <- pmin(pmax(df$D, eps), 1 - eps)

# ============================================================
# Improved Beta GLMM
# ============================================================

fit_beta_improved <- glmmTMB(
  
  D_beta ~
    
    # mean structure
    R_sc * model_variant +
    
    ns(invN_sc, df = 4) * model_variant +
    
    task +
    
    (1 | prompt_uid) +
    (1 | model_name),
  
  # improved dispersion structure
  dispformula = ~
    R_sc * model_variant +
    model_variant * task +
    ns(invN_sc, df = 3) +
    model_name,
  
  family = beta_family(link = "logit"),
  
  data = df
)

summary(fit_beta_improved)

# ============================================================
# Diagnostics
# ============================================================

sim_beta2 <- simulateResiduals(
  fit_beta_improved,
  n = 1000
)

plot(sim_beta2)

print(testUniformity(sim_beta2))