args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
suppressPackageStartupMessages(library(rms))

default <- function(value, fallback) if (is.null(value)) fallback else value

fit_case <- function(case) {
    x <- as.matrix(case$x)
    y <- case$y
    n <- length(y)
    method <- default(case$method, "boot")
    confidence <- if ("confidence" %in% names(case)) default(case$confidence, 0) else 0.95
    fit <- lrm(y ~ x, x = TRUE, y = TRUE)
    options <- list(fit = fit, method = method, B = case$B, conf.int = confidence)
    if (!is.null(case$predy)) options$predy <- case$predy
    if (any(c("span", "iterations", "delta", "digits") %in% names(case))) {
        options$smoother <- function(x, y) {
            # rms 8.2-1 does not forward its formal digits argument to the smoother.
            if (!is.null(case$digits)) x <- round(x, case$digits)
            lowess(x, y, f = default(case$span, 2 / 3), iter = default(case$iterations, 0L),
                delta = default(case$delta, 0.01 * diff(range(x))))
        }
    }

    RNGkind("Mersenne-Twister", "Inversion", "Rejection")
    set.seed(case$r_seed)
    if (method == "crossvalidation") {
        selection <- list(fold_order = sample(n, replace = FALSE) - 1L)
    } else if (method == "randomization") {
        selection <- list(permutation_indices = t(replicate(case$B, sample(n, replace = FALSE))) - 1L)
    } else {
        selection <- list(bootstrap_indices = t(replicate(case$B, sample(n, replace = TRUE))) - 1L)
    }
    set.seed(case$r_seed)
    result <- do.call(calibrate, options)
    extract <- function(column) as.list(as.numeric(result[, column]))
    c(selection, list(
        id = case$id,
        predy = extract("predy"),
        fitted_probabilities = as.list(attr(result, "predicted")),
        apparent = extract("calibrated.orig"),
        corrected = extract("calibrated.corrected"),
        training_error = extract("training"),
        test_error = extract("test"),
        optimism = extract("optimism"),
        lower = if (confidence == 0) NULL else as.list(result[, "Lower"] + result[, "predy"]),
        upper = if (confidence == 0) NULL else as.list(result[, "Upper"] + result[, "predy"]),
        counts = extract("n")
    ))
}

cases <- jsonlite::fromJSON(args[[1]], simplifyDataFrame = FALSE)
output <- list(
    versions = list(
        R = as.character(getRversion()), rms = packageDescription("rms", fields = "Version"),
        Hmisc = packageDescription("Hmisc", fields = "Version")
    ),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
