args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
library(gsDesign)

spending_function <- function(name) {
    if (identical(name, "ldof")) sfLDOF else sfHSD
}

fit_two_boundaries <- function(case, alpha, spending, parameter) {
    timing <- case$information_fractions
    kind <- if (identical(case$design, "symmetric")) 2L else if (isTRUE(case$binding)) 3L else 4L
    beta <- if (is.null(case$beta)) 0.1 else case$beta
    futility_spending <- spending_function(case$futility_spending)
    futility_parameter <- case$futility_parameter
    if (identical(futility_spending, sfHSD) && is.null(futility_parameter)) futility_parameter <- -2
    fit <- gsDesign(
        k = length(timing), test.type = kind, alpha = alpha, beta = beta,
        timing = timing, sfu = spending, sfupar = parameter,
        sfl = futility_spending, sflpar = futility_parameter, r = 80, tol = 1e-7
    )
    exposure <- sqrt(tail(fit$n.I, 1))
    compare <- function(effect, respect_lower = TRUE, look = NULL, z = NULL) {
        reference <- fit
        # gsProbability uses 20 as an effective no-stop bound in its C implementation.
        if (!respect_lower) reference$lower$bound[] <- -20
        if (is.null(look)) {
            result <- gsProbability(d = reference, theta = effect / exposure)
        } else {
            result <- gsCP(reference, theta = effect / exposure, i = look + 1L, zi = z, r = 80)
        }
        list(
            effect = effect, respect_lower = respect_lower, look = look, z = z,
            upper = as.list(as.numeric(result$upper$prob)),
            lower = as.list(as.numeric(result$lower$prob))
        )
    }
    effect <- fit$delta * exposure
    list(
        id = case$id, information_fractions = as.list(fit$timing),
        upper_boundaries = as.list(fit$upper$bound), lower_boundaries = as.list(fit$lower$bound),
        cumulative_alpha = as.list(cumsum(fit$upper$spend)),
        cumulative_beta = if (kind == 2L) NULL else as.list(cumsum(fit$lower$spend)),
        alternative_effect = if (kind == 2L) NULL else effect,
        evaluations = list(
            compare(0), compare(effect), compare(0, FALSE),
            compare(effect, TRUE, look = 0, z = case$interim_z),
            compare(0, FALSE, look = length(timing) - 2L, z = case$interim_z)
        )
    )
}

fit_case <- function(case) {
    timing <- case$information_fractions
    alpha <- if (is.null(case$alpha)) 0.025 else case$alpha
    spending <- spending_function(case$spending)
    parameter <- case$parameter
    if (identical(spending, sfHSD) && is.null(parameter)) parameter <- -4
    if (!is.null(case$design)) return(fit_two_boundaries(case, alpha, spending, parameter))
    fit <- gsDesign(
        k = length(timing), test.type = 1, alpha = alpha, timing = timing,
        sfu = spending, sfupar = parameter, r = 80, tol = 1e-10
    )
    list(
        id = case$id, information_fractions = as.list(fit$timing),
        z_boundaries = as.list(fit$upper$bound),
        nominal_pvalues = as.list(pnorm(fit$upper$bound, lower.tail = FALSE)),
        cumulative_alpha = as.list(cumsum(fit$upper$spend)),
        incremental_alpha = as.list(fit$upper$spend),
        null_crossing_probabilities = as.list(fit$upper$prob[, 1])
    )
}

cases <- jsonlite::fromJSON(args[[1]], simplifyDataFrame = FALSE)
output <- list(
    versions = list(R = as.character(getRversion()), gsDesign = as.character(packageVersion("gsDesign"))),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
