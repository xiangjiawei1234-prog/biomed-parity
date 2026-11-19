args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

default <- function(value, fallback) if (is.null(value)) fallback else value

results <- lapply(cases, function(case) {
    result <- mice::pool.scalar(
        unlist(case$q), unlist(case$u), n = default(case$n, Inf),
        k = default(case$k, 1), rule = default(case$rule, "rubin1987")
    )
    se <- sqrt(result$t)
    statistic <- (result$qbar - default(case$null, 0)) / se
    alpha <- 1 - default(case$confidence_level, .95)
    margin <- suppressWarnings(qt(alpha / 2, result$df, lower.tail = FALSE)) * se
    result$standard_error <- se
    result$statistic <- statistic
    result$pvalue <- suppressWarnings(2 * pt(abs(statistic), result$df, lower.tail = FALSE))
    result$lower <- result$qbar - margin
    result$upper <- result$qbar + margin
    result
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("mice")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
