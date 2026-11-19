args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

matrix_rows <- function(matrix) {
    lapply(seq_len(nrow(matrix)), function(i) as.list(unname(matrix[i, ])))
}

results <- lapply(cases, function(case) {
    x <- if (is.list(case$covariates[[1]])) {
        do.call(rbind, lapply(case$covariates, unlist))
    } else {
        as.matrix(unlist(case$covariates))
    }
    colnames(x) <- paste0("x", seq_len(ncol(x)))
    data <- data.frame(time = unlist(case$time), status = unlist(case$status),
        w = unlist(case$weights), x)
    if (!is.null(case$psu)) data$psu <- unlist(case$psu)
    if (!is.null(case$strata)) data$sampling_strata <- unlist(case$strata)
    if (!is.null(case$model_strata)) data$model_strata <- unlist(case$model_strata)
    if (!is.null(case$fpc)) data$fpc <- unlist(case$fpc)

    d <- survey::svydesign(
        ids = if (is.null(case$psu)) ~1 else ~psu,
        strata = if (is.null(case$strata)) NULL else ~sampling_strata,
        fpc = if (is.null(case$fpc)) NULL else ~fpc,
        weights = ~w, data = data, nest = TRUE
    )
    Surv <- survival::Surv
    strata <- survival::strata
    terms <- colnames(x)
    if (!is.null(case$model_strata)) terms <- c(terms, "strata(model_strata)")
    formula <- reformulate(terms, response = "Surv(time,status)")
    ties <- if (is.null(case$ties)) "efron" else case$ties
    level <- if (is.null(case$confidence_level)) .95 else case$confidence_level
    fit <- do.call(survey::svycoxph, list(formula, design = d, method = ties))
    invisible(capture.output(table <- summary(fit)$coefficients))
    ci <- exp(confint(fit, level = level))
    list(
        coefficients = as.list(unname(coef(fit))),
        covariance = matrix_rows(vcov(fit)),
        naive_covariance = matrix_rows(fit$inv.info),
        standard_errors = as.list(unname(sqrt(diag(vcov(fit))))),
        hazard_ratios = as.list(unname(exp(coef(fit)))),
        lower_hazard_ratios = as.list(unname(ci[, 1])),
        upper_hazard_ratios = as.list(unname(ci[, 2])),
        pvalues = as.list(unname(table[, "Pr(>|z|)"])),
        log_likelihood = fit$ll[[2]],
        null_log_likelihood = fit$ll[[1]],
        design_df = survey::degf(d),
        residual_df = fit$degf.resid
    )
})
names(results) <- vapply(cases, function(case) case$id, character(1))
versions <- as.list(sapply(c("survey", "survival"), function(pkg) {
    as.character(utils::packageVersion(pkg))
}))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_versions = versions, results = results), auto_unbox = TRUE,
    digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
