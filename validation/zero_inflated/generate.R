#!/usr/bin/env Rscript

arguments <- commandArgs(trailingOnly = TRUE)
fixture_dir <- if (length(arguments)) arguments[[1]] else "validation/zero_inflated"
suppressPackageStartupMessages(library(glmmTMB))
obs <- read.delim(file.path(fixture_dir, "observations.tsv"))

row <- function(case) {
    data <- obs[obs$dataset == case, ]
    data$group <- factor(data$group)
    intercept_only <- case == "intercept"
    form <- if (intercept_only) count ~ 1 + (1 | group) else count ~ x + offset(offset) + (1 | group)
    zero <- if (intercept_only) ~1 else ~z
    fit <- glmmTMB(
        form, ziformula = zero, data = data, family = poisson,
        control = glmmTMBControl(
            optimizer = optim, optArgs = list(method = "BFGS"),
            optCtrl = list(reltol = 1e-13, maxit = 500)
        )
    )
    stopifnot(fit$fit$convergence == 0, fit$sdr$pdHess)
    coef <- fixef(fit)
    trial <- if (intercept_only) c(0.70, -0.90, log(0.55)) else c(0.68, 0.4, -0.85, 0.65, log(0.55))
    expected_order <- c(rep("beta", length(coef$cond)), rep("betazi", length(coef$zi)), "theta")
    stopifnot(identical(names(fit$fit$par), expected_order))
    names(trial) <- names(fit$fit$par)
    values <- c(
        count_intercept = unname(coef$cond[1]), count_slope = unname(coef$cond[2]),
        zi_intercept = unname(coef$zi[1]), zi_slope = unname(coef$zi[2]),
        sd = attr(VarCorr(fit)$cond$group, "stddev")[[1]],
        log_likelihood = as.numeric(logLik(fit)),
        trial_log_likelihood = -fit$obj$fn(trial)
    )
    data.frame(id = case, as.list(setNames(sprintf("%.17g", values), names(values))))
}

rows <- lapply(unique(obs$dataset), row)
cat("# R ", as.character(getRversion()), ", glmmTMB ", as.character(packageVersion("glmmTMB")),
    ", TMB ", as.character(packageVersion("TMB")), "\n", sep = "")
write.table(do.call(rbind, rows), sep = "\t", row.names = FALSE, quote = FALSE)
