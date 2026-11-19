args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
library(gsDesign)

fit_case <- function(case) {
    timing <- case$information_fractions
    alpha <- if (is.null(case$alpha)) 0.025 else case$alpha
    spending <- if (identical(case$spending, "ldof")) sfLDOF else sfHSD
    parameter <- case$parameter
    if (identical(spending, sfHSD) && is.null(parameter)) parameter <- -4
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
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, pretty = TRUE), "\n", sep = "")
