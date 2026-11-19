# Run with Rscript --vanilla validation/survival/oracle.R cases.json.
# The oracle uses survival's exported API and prints JSON to stdout.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
library(survival)
invisible(Sys.setlocale("LC_COLLATE", "C"))

fit_case <- function(case) {
    rho <- if (is.null(case$rho)) 0 else case$rho
    if (!is.null(case$survival_probability)) {
        reference <- data.frame(
            time = seq_along(case$event),
            event = case$event,
            probability = case$survival_probability
        )
        fit <- survdiff(Surv(time, event) ~ offset(probability), reference, rho = rho)
        return(list(
            id = case$id, n = fit$n, observed = fit$obs, expected = fit$exp,
            variance = fit$var, statistic = fit$chisq, df = 1L, pvalue = fit$pvalue
        ))
    }

    group_levels <- if (is.null(case$group_order)) sort(unique(case$group)) else case$group_order
    reference <- data.frame(
        time = case$time, event = case$event,
        group = factor(case$group, levels = group_levels)
    )
    if (is.null(case$strata)) {
        stratum_levels <- NULL
        formula <- Surv(time, event) ~ group
    } else {
        stratum_levels <- if (is.null(case$strata_order)) sort(unique(case$strata)) else case$strata_order
        reference$stratum <- factor(case$strata, levels = stratum_levels)
        formula <- Surv(time, event) ~ group + strata(stratum)
    }
    run_survdiff <- survdiff
    # survival 3.8.6 forwards an explicit timefix into model.frame.
    # Change only the copied wrapper's default to exercise its FALSE branch.
    if (identical(case$timefix, FALSE)) formals(run_survdiff)$timefix <- FALSE
    fit <- tryCatch(
        suppressWarnings(run_survdiff(formula, reference, rho = rho)),
        error = function(error) {
            if (!grepl("singular", conditionMessage(error), fixed = TRUE)) stop(error)
            NULL
        }
    )
    if (is.null(fit)) return(list(id = case$id, error = "singular covariance"))
    expected <- if (is.matrix(fit$exp)) rowSums(fit$exp) else fit$exp
    list(
        id = case$id, groups = as.list(group_levels), counts = as.list(as.numeric(fit$n)),
        observed = if (is.matrix(fit$obs)) fit$obs else as.list(fit$obs),
        expected = if (is.matrix(fit$exp)) fit$exp else as.list(fit$exp),
        variance = fit$var, statistic = fit$chisq, df = sum(expected > 0) - 1L,
        pvalue = fit$pvalue,
        strata = if (is.null(stratum_levels)) NULL else as.list(stratum_levels),
        strata_counts = if (is.null(fit$strata)) NULL else as.list(as.numeric(fit$strata))
    )
}

cases <- jsonlite::fromJSON(args[[1]], simplifyDataFrame = FALSE)
results <- list(
    versions = list(R = as.character(getRversion()), survival = as.character(packageVersion("survival"))),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(results, auto_unbox = TRUE, digits = NA, na = "string", null = "null", pretty = TRUE), "\n", sep = "")
