#!/usr/bin/env Rscript

arguments <- commandArgs(trailingOnly = TRUE)
fixture_dir <- if (length(arguments)) arguments[[1]] else "validation/meta_analysis"
observations <- read.delim(file.path(fixture_dir, "observations.tsv"))
scenarios <- read.delim(file.path(fixture_dir, "scenarios.tsv"))

result_row <- function(scenario) {
    data <- observations[observations$dataset == scenario$dataset, ]
    fit <- metafor::rma.uni(
        yi = data$effect, vi = data$variance, method = "REML", test = scenario$test,
        level = 100 * scenario$confidence_level, control = list(threshold = 1e-12)
    )
    predicted <- predict(fit, predtype = scenario$prediction)
    control <- list(tol = 1e-11)
    if (!is.na(scenario$tau2_upper_limit)) {
        control$tau2.max <- scenario$tau2_upper_limit
    }
    profiled <- confint(fit, type = "PL", control = control)
    values <- c(
        study_count = fit$k, mean_effect = as.numeric(fit$b), standard_error = fit$se,
        statistic = fit$zval, p_value = fit$pval, lower = fit$ci.lb, upper = fit$ci.ub,
        prediction_lower = predicted$pi.lb, prediction_upper = predicted$pi.ub,
        tau2 = fit$tau2, tau2_lower = profiled$random["tau^2", "ci.lb"],
        tau2_upper = profiled$random["tau^2", "ci.ub"],
        tau2_upper_limit_reached = as.numeric(profiled$ub.sign == ">"),
        i2_percent = fit$I2, i2_lower = profiled$random["I^2(%)", "ci.lb"],
        i2_upper = profiled$random["I^2(%)", "ci.ub"], h2 = fit$H2,
        h2_lower = profiled$random["H^2", "ci.lb"], h2_upper = profiled$random["H^2", "ci.ub"]
    )
    data.frame(
        id = scenario$id,
        as.list(setNames(sprintf("%.17g", values), names(values))),
        check.names = FALSE
    )
}

rows <- lapply(seq_len(nrow(scenarios)), function(i) result_row(scenarios[i, ]))
cat("# R ", as.character(getRversion()), ", metafor ", as.character(packageVersion("metafor")), "\n", sep = "")
write.table(do.call(rbind, rows), sep = "\t", row.names = FALSE, quote = FALSE)
