#!/usr/bin/env Rscript

arguments <- commandArgs(trailingOnly = TRUE)
fixture_dir <- if (length(arguments)) arguments[[1]] else "validation/roc"

observations <- read.delim(file.path(fixture_dir, "observations.tsv"))
scenarios <- read.delim(file.path(fixture_dir, "scenarios.tsv"))
fmt <- function(value) sprintf("%.17g", value)

result_row <- function(scenario) {
    data <- observations[observations$dataset == scenario$dataset, ]
    first <- data[data$cohort == 1, ]
    second <- data[data$cohort == 2, ]
    make_roc <- function(labels, scores) {
        pROC::roc(labels, scores, levels = c(0, 1), direction = scenario$direction, quiet = TRUE)
    }
    auc_1 <- make_roc(first$label, first$score_1)
    variance <- function(roc) pROC::var(roc, method = "delong")
    results <- c(
        auc_1 = as.numeric(auc_1$auc), auc_2 = NA_real_,
        variance_1 = variance(auc_1), variance_2 = NA_real_, covariance = NA_real_,
        difference = NA_real_, statistic = NA_real_, degrees_of_freedom = NA_real_,
        p_value = NA_real_, lower = NA_real_, upper = NA_real_
    )
    if (scenario$method == "auc") {
        interval <- pROC::ci.auc(auc_1, conf.level = scenario$confidence_level, method = "delong")
        results[c("lower", "upper")] <- interval[c(1, 3)]
    } else {
        paired <- scenario$method == "paired"
        auc_2 <- if (paired) {
            make_roc(first$label, first$score_2)
        } else {
            make_roc(second$label, second$score_1)
        }
        results[c("auc_2", "variance_2", "difference")] <- c(
            as.numeric(auc_2$auc), variance(auc_2), as.numeric(auc_1$auc - auc_2$auc)
        )
        comparison <- pROC::roc.test(
            auc_1, auc_2, method = "delong", paired = paired, alternative = scenario$alternative,
            conf.level = if (paired) scenario$confidence_level else 0.95
        )
        results[c("statistic", "p_value")] <- c(comparison$statistic, comparison$p.value)
        if (paired) {
            results["covariance"] <- pROC::cov(auc_1, auc_2, method = "delong")
            results[c("lower", "upper")] <- comparison$conf.int
        } else {
            results["degrees_of_freedom"] <- comparison$parameter
        }
    }
    data.frame(
        id = scenario$id,
        as.list(setNames(fmt(results), names(results))),
        check.names = FALSE
    )
}

rows <- lapply(seq_len(nrow(scenarios)), function(i) {
    # Some fixtures deliberately exercise degenerate outputs such as AUC = 1.
    suppressWarnings(result_row(scenarios[i, ]))
})
cat("# R ", as.character(getRversion()), ", pROC ", as.character(packageVersion("pROC")), "\n", sep = "")
write.table(do.call(rbind, rows), sep = "\t", row.names = FALSE, quote = FALSE)
