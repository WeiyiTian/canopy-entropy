library(glmmTMB)
library(DHARMa)
library(splines)

df <- read.csv("./panel.csv")

df$D <- abs(df$D)

df$task <- factor(df$task)
df$prompt_uid <- factor(df$prompt_uid)
df$model_name <- factor(df$model_name)
df$model_variant <- factor(df$model_variant)

df$model_variant <- relevel(df$model_variant, ref = "base")

# One level per model family x variant x task: under few-shot prompting, base
# models continue the demonstration pattern after answering while instruct
# models stop, so each group's diversity level and spread are its own.
df$group <- interaction(df$model_name, df$model_variant, df$task, drop = TRUE)

df$R_sc <- as.numeric(scale(df$R_bar))
# log rather than inverse length: few-shot rollouts often run on for thousands
# of tokens, a range that 1/N compresses into a single point.
df$logN_sc <- as.numeric(scale(log(df$N_bar)))
# Within-prompt spread of rollout lengths: D is pairwise, and prompts whose
# rollouts mix response modes of very different lengths are more diverse.
df$sd_logN_sc <- as.numeric(scale(df$sd_logN))

eps <- 1e-6
df$D_beta <- pmin(pmax(df$D, eps), 1 - eps)

fit_beta_fewshot <- glmmTMB(

  # group already contains model_variant, so variant enters only through its
  # interactions; adding its main effect again would be rank-deficient.
  D_beta ~
    R_sc + R_sc:model_variant +
    ns(logN_sc, df = 4) + ns(logN_sc, df = 4):model_variant +
    group +
    sd_logN_sc +
    (1 | prompt_uid),

  dispformula = ~
    group +
    ns(logN_sc, df = 3) +
    R_sc + R_sc:model_variant +
    sd_logN_sc,

  family = beta_family(link = "logit"),

  data = df,

  control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))
)

summary(fit_beta_fewshot)

sim_beta_fewshot <- simulateResiduals(
  fit_beta_fewshot,
  n = 10000,
  seed = 93
)

plot(sim_beta_fewshot)

testUniformity(sim_beta_fewshot)
testDispersion(sim_beta_fewshot)
testOutliers(sim_beta_fewshot)
