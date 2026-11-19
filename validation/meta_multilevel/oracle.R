args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
source <- jsonlite::read_json(args[[1]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value
keep <- function(x) I(as.numeric(x))
matrix_rows <- function(x) as.matrix(x)
has_cr2 <- requireNamespace("clubSandwich", quietly = TRUE)
capture <- function(x) list(
    covariance = matrix_rows(x$vb), standard_errors = keep(x$se),
    statistics = keep(x$zval), p_values = keep(x$pval),
    confidence_intervals = cbind(x$ci.lb, x$ci.ub),
    degrees_of_freedom = keep(x$ddf),
    omnibus = list(statistic = unname(x$QM), numerator_df = x$QMdf[1],
                   denominator_df = x$QMdf[2], p_value = unname(x$QMp))
)
results <- lapply(source$cases, function(case) {
    data <- source$datasets[[case$dataset]]
    yi <- unlist(data$effects)
    v <- unlist(data$variances)
    group <- unlist(data$sampling_clusters)
    V <- data$sampling_correlation * outer(sqrt(v),sqrt(v)) * outer(group, group, '==')
    diag(V) <- v
    X <- if (is.list(data$moderators[[1L]])) {
        do.call(rbind, lapply(data$moderators, unlist))
    } else as.matrix(unlist(data$moderators))
    data_frame <- data.frame(yi = yi)
    grouping <- default(case$groups, data$groups)
    term_keys <- names(grouping)
    for (key in term_keys) {
        value <- grouping[[key]]
        data_frame[[key]] <- if (is.list(value[[1L]])) {
            vapply(value, function(row) paste(unlist(row), collapse = ":"), character(1))
        } else unlist(value)
    }
    random <- lapply(term_keys, function(key) as.formula(paste("~1|", key), env = environment()))
    fix <- case$fixed_variances
    sigma2 <- vapply(term_keys, function(key) default(fix[[key]], NA_real_), numeric(1))
    R <- lapply(names(case$known_correlations), function(key) {
        C <- do.call(rbind, lapply(case$known_correlations[[key]], unlist))
        dimnames(C) <- rep(list(sort(unique(data_frame[[key]]))), 2L)
        C
    })
    names(R) <- names(case$known_correlations)
    options <- list(
        yi = yi, V = V, mods = X, intercept = default(case$intercept, TRUE),
        random = random, R = R, Rscale = "none", sigma2 = sigma2, data = data_frame,
        method = default(case$method, "REML"), test = default(case$test, "z"),
        dfs = default(unlist(case$dfs), "residual"),
        level = 100 * default(case$confidence_level, .95),
        control = list(rel.tol = 1e-10, x.tol = 1e-8, eval.max = 1000L, iter.max = 500L)
    )
    if (!is.null(case$tested_coefficients)) options$btt <- unlist(case$tested_coefficients) + 1L
    if (!length(R)) options <- options[!names(options) %in% c("R", "Rscale")]
    if (!length(random)) options <- options[!names(options) %in% c("random","R","Rscale","sigma2")]
    fit <- suppressWarnings(do.call(metafor::rma.mv, options))
    ans <- c(capture(fit), list(
        coefficients = keep(coef(fit)), marginal_covariance = matrix_rows(fit$M),
        residuals = keep(residuals(fit)), log_likelihood = as.numeric(logLik(fit)),
        variance_components = as.list(setNames(fit$sigma2[seq_along(term_keys)], term_keys)),
        heterogeneity = list(statistic = fit$QE, numerator_df = fit$k - fit$p,
                             denominator_df = NA_integer_, p_value = fit$QEp)
    ))
    if (isTRUE(case$profile)) {
        free <- which(is.na(sigma2))
        intervals <- lapply(free, function(j) {
            unname(confint(fit, sigma2 = j, control = list(tol = 1e-8))$random[1, 2:3])
        })
        ans$variance_intervals <- setNames(intervals, term_keys[free])
    }
    for (correction in unlist(case$corrections)) {
        if (correction == "CR2" && !has_cr2) next
        ans$robust[[correction]] <- capture(metafor::robust(
            fit, cluster = unlist(data$robust_clusters), adjust = correction == "CR1",
            clubSandwich = correction == "CR2"
        ))
    }
    ans
})
names(results) <- vapply(source$cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("metafor")),
    clubSandwich_version = if (has_cr2) as.character(packageVersion("clubSandwich")) else NULL,
    results = results
), auto_unbox = TRUE, digits = NA, na = "null"), "\n", sep = "")
