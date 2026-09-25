# Mixed logistic models for the camera-ready version (run after camera_ready.py).
# correct ~ label + exam (fixed) + (1 | student) + (1 | item), Laplace approximation (lme4).
# Wald and profile 95% intervals for the label slope. Output: out/camera_ready_glmm.txt
suppressPackageStartupMessages(library(lme4))
d <- read.csv("out/responses_long.csv")
labs <- c(`Gemini self difficulty` = "llm_diff_i", `Fable instr. difficulty` = "F_diff_i_n",
          `Fable bare difficulty` = "F_diff_b_n", `Gemini cold difficulty` = "G_diff_i_n", `Rater A Bloom` = "A_bloom_i", `Rater B Bloom` = "B_bloom_i",
          `Gemini self Bloom` = "llm_bloom_i", `Fable instr. Bloom` = "F_bloom_i_n", `Fable bare Bloom` = "F_bloom_b_n", `Gemini cold Bloom` = "G_bloom_i_n")
sink("out/camera_ready_glmm.txt")
cat(sprintf("%d responses, %d students, %d items\n", nrow(d), length(unique(d$student)), length(unique(d$item))))
for (nm in names(labs)) {
  dd <- d[!is.na(d[[labs[[nm]]]]), ]
  dd$x <- dd[[labs[[nm]]]]
  m <- glmer(y ~ x + exam + (1 | student) + (1 | item), data = dd, family = binomial,
             control = glmerControl(optimizer = "bobyqa"))
  b <- fixef(m)[["x"]]; se <- sqrt(diag(vcov(m)))[["x"]]
  pr <- tryCatch(confint(m, parm = "x", method = "profile"), error = function(e) matrix(NA, 1, 2))
  vc <- as.data.frame(VarCorr(m))
  cat(sprintf("%-26s b = %.2f (SE %.2f), OR %.2f, Wald 95%% [%.2f, %.2f], profile OR [%.2f, %.2f], p = %.3f; SD student %.2f, item %.2f\n",
              nm, b, se, exp(b), exp(b - 1.96 * se), exp(b + 1.96 * se), exp(pr[1]), exp(pr[2]),
              2 * pnorm(-abs(b / se)), vc$sdcor[vc$grp == "student"], vc$sdcor[vc$grp == "item"]))
}
sink()
cat(readLines("out/camera_ready_glmm.txt"), sep = "\n")
