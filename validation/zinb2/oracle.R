args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
suppressPackageStartupMessages(library(glmmTMB))
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

evaluate <- function(case) {
    data <- as.data.frame(do.call(rbind, lapply(case$observations, unlist)))
    names(data) <- c("count", "group", "x", "z", "d", "offset")
    data$group <- factor(data$group)
    fit <- glmmTMB(
        if (isTRUE(case$varied)) count ~ x + offset(offset) + (1 | group) else count ~ (1 | group),
        ziformula = if (isTRUE(case$varied)) ~z else ~1,
        dispformula = if (isTRUE(case$varied)) ~d else ~1,
        data = data, family = nbinom2,
        control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"),
                                 optCtrl = list(reltol = 1e-13, maxit = 600))
    )
    stopifnot(fit$fit$convergence == 0, fit$sdr$pdHess)
    coefs <- fixef(fit)
    expected_order <- c(rep("beta", length(coefs$cond)), rep("betazi", length(coefs$zi)),
                        rep("betadisp", length(coefs$disp)), "theta")
    stopifnot(identical(names(fit$fit$par), expected_order))
    trial <- setNames(unlist(case$trial), expected_order)
    list(count = I(unname(coefs$cond)), inflation = I(unname(coefs$zi)),
         dispersion = I(unname(coefs$disp)), sd = attr(VarCorr(fit)$cond$group, "stddev")[[1]],
         log_likelihood = as.numeric(logLik(fit)), trial_log_likelihood = -fit$obj$fn(trial))
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    glmmTMB = as.character(packageVersion("glmmTMB")), TMB = as.character(packageVersion("TMB")),
    results = results), auto_unbox = TRUE, digits = NA, pretty = TRUE), "\n", sep = "")
