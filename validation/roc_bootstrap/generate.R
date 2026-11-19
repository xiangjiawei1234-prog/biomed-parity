#!/usr/bin/env Rscript

arguments <- commandArgs(trailingOnly = TRUE)
fixture_dir <- if (length(arguments)) arguments[[1]] else "validation/roc_bootstrap"
data <- read.delim(file.path(fixture_dir, "observations.tsv"))
seed <- 70421
n <- 61
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)
controls <- which(data$label == 0)
cases <- which(data$label == 1)
draws <- t(replicate(n, c(sample(controls, replace = TRUE), sample(cases, replace = TRUE))))
colnames(draws) <- paste0("i", seq_len(ncol(draws)))
if ("--draws" %in% arguments) {
    write.table(draws - 1L, sep = "\t", row.names = FALSE, quote = FALSE)
    quit(save = "no")
}
suppressPackageStartupMessages(library(pROC))
scenarios <- read.delim(file.path(fixture_dir, "scenarios.tsv"))

result_row <- function(scenario) {
    spec <- list(
        partial.auc = if (is.na(scenario$low)) FALSE else c(scenario$low, scenario$high),
        partial.auc.focus = scenario$focus, partial.auc.correct = scenario$standardize
    )
    curve <- function(score, chosen = seq_len(nrow(data)), direction = scenario$direction, below = FALSE) {
        args <- c(list(
            response = data$label[chosen], predictor = data[[score]][chosen], direction = direction,
            levels = c(0, 1), percent = scenario$percent, quiet = TRUE, auc = TRUE,
            allow.invalid.partial.auc.correct = below
        ), spec)
        do.call(pROC::roc, args)
    }
    first <- curve(scenario$first, below = scenario$allow_below_chance)
    values <- c(
        auc_1 = as.numeric(first$auc), auc_2 = NA_real_, difference = NA_real_,
        standard_error = NA_real_, statistic = NA_real_, p_value = NA_real_,
        lower = NA_real_, median = NA_real_, upper = NA_real_, n_resamples = NA_real_
    )
    if (scenario$method != "auc") {
        resample_auc <- function(score, direction) {
            apply(draws, 1L, function(idx) as.numeric(curve(score, idx, direction, below = TRUE)$auc))
        }
        first_draws <- resample_auc(scenario$first, first$direction)
        if (scenario$method == "ci") {
            set.seed(seed)
            ci <- pROC::ci.auc(
                first, method = "bootstrap", conf.level = scenario$confidence_level,
                boot.n = n, boot.stratified = TRUE
            )
            tail <- (1 - scenario$confidence_level) / 2
            manual <- quantile(first_draws, c(tail, 0.5, 1 - tail), na.rm = TRUE)
            stopifnot(isTRUE(all.equal(as.numeric(ci), as.numeric(manual), tolerance = 1e-13)))
            values[c("lower", "median", "upper")] <- as.numeric(ci)
            values["n_resamples"] <- sum(is.finite(first_draws))
        } else {
            second <- curve(scenario$second, below = scenario$allow_below_chance)
            second_draws <- resample_auc(scenario$second, second$direction)
            differences <- first_draws - second_draws
            set.seed(seed)
            result <- pROC::roc.test(
                first, second, method = "bootstrap", paired = TRUE,
                alternative = scenario$alternative, boot.n = n, boot.stratified = TRUE
            )
            difference <- as.numeric(first$auc - second$auc)
            se <- sd(differences, na.rm = TRUE)
            statistic <- if (se == 0 && difference == 0 && all(differences == 0)) 0 else difference / se
            stopifnot(isTRUE(all.equal(as.numeric(result$statistic), statistic, tolerance = 1e-13)))
            values[c("auc_2", "difference", "standard_error", "statistic", "p_value", "n_resamples")] <-
                c(as.numeric(second$auc), difference, se, result$statistic, result$p.value, sum(is.finite(differences)))
        }
    }
    # Use NaN for oracle values reported as missing. Empty cells mark unused fields.
    numbers <- sprintf("%.17g", values)
    used <- if (scenario$method == "auc") "auc_1" else if (scenario$method == "ci") {
        c("auc_1", "lower", "median", "upper", "n_resamples")
    } else {
        c("auc_1", "auc_2", "difference", "standard_error", "statistic", "p_value", "n_resamples")
    }
    numbers[names(values) %in% used & is.na(values)] <- "NaN"
    data.frame(id = scenario$id, as.list(setNames(numbers, names(values))))
}

result <- lapply(seq_len(nrow(scenarios)), function(i) suppressWarnings(result_row(scenarios[i, ])))
cat("# R ", as.character(getRversion()), ", pROC ", as.character(packageVersion("pROC")), "\n", sep = "")
write.table(do.call(rbind, result), sep = "\t", row.names = FALSE, quote = FALSE)
