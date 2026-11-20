#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survey))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected cases.json and output JSON path")
options(survey.replicates.mse = FALSE, na.action = "na.omit")
cases <- jsonlite::read_json(args[[1]], simplifyVector = FALSE)
scalar <- jsonlite::unbox

as_matrix <- function(rows) {
    do.call(rbind, lapply(rows, function(row) {
        vapply(row, function(x) if (is.null(x)) NA_real_ else as.numeric(x), numeric(1))
    }))
}

r_names <- c(combined_weights = "combined.weights", bootstrap_average = "bootstrap.average")
results <- lapply(cases, function(case) {
    values <- as_matrix(case$values)
    repweights <- as_matrix(case$repweights)
    pweights <- unlist(case$weights)
    settings <- lapply(case$options, unlist)
    for (name in intersect(names(settings), names(r_names))) {
        names(settings)[names(settings) == name] <- r_names[[name]]
    }
    design <- do.call(svrepdesign, c(list(
        repweights = repweights, weights = pweights, data = as.data.frame(values)
    ), settings))
    if (!is.null(case$subset)) {
        selected <- unlist(case$subset)
        design <- design[selected, ]
        values <- values[selected, , drop = FALSE]
    }
    na_rm <- isTRUE(case$na_rm)
    summarize <- function(fit) {
        variance <- vcov(fit)
        list(estimate = unname(as.numeric(coef(fit))), covariance = unname(variance),
            standard_errors = unname(as.numeric(SE(fit))),
            replicate_estimates = matrix(fit$replicates, nrow = ncol(repweights), ncol = ncol(values)),
            variance_center = unname(as.numeric(attr(attr(fit[[1]], "var"), "means"))),
            design_df = scalar(degf(if (na_rm) design[complete.cases(values), ] else design)))
    }
    list(
        scale = scalar(design$scale), rscales = rep(design$rscales, length.out = ncol(repweights)),
        mse = scalar(design$mse), design_df = scalar(degf(design)),
        sampling_weights = unname(weights(design, type = "sampling")),
        replicate_weights = unname(weights(design, type = "replication")),
        analysis_weights = unname(weights(design, type = "analysis")),
        mean = summarize(svymean(values, design, na.rm = na_rm, return.replicates = TRUE)),
        total = summarize(svytotal(values, design, na.rm = na_rm, return.replicates = TRUE))
    )
})
names(results) <- vapply(cases, function(case) case$name, character(1))
jsonlite::write_json(list(
    r_version = scalar(as.character(getRversion())),
    survey_version = scalar(as.character(packageVersion("survey"))),
    results = results
), args[[2]], digits = 17, pretty = TRUE, na = "null")
