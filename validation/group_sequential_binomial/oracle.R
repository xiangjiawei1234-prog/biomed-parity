args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(gsDesign))

default <- function(x, value) if (is.null(x)) value else x

observed <- function(fit, origin = 0) {
  stopped <- fit$lower$prob + fit$upper$prob
  remaining <- 1 - vapply(seq_len(ncol(stopped)), function(j) cumsum(stopped[, j]), numeric(fit$k))
  list(
    sample_sizes = fit$n.I + origin, lower = unname(fit$lower$prob), upper = unname(fit$upper$prob),
    continuing = matrix(remaining, nrow = fit$k), expected_sample_size = fit$en + origin
  )
}

calibrate <- function(case) {
  counts <- unlist(case$sample_sizes)
  p <- case$options
  alpha <- default(p$alpha, .025)
  null <- case$null_probability
  reflect <- identical(p$direction, "upper")
  if (reflect) null <- 1 - null
  spending <- if (identical(p$spending, "ldof")) sfLDOF else sfHSD
  parameter <- default(p$parameter, if (identical(p$spending, "ldof")) 1 else -4)
  x <- gsSurv(
    k = length(counts), test.type = 1, timing = counts / max(counts),
    ratio = null / (1 - null), alpha = alpha, sfu = spending, sfupar = parameter
  )
  # Keep the intended analysis fractions at the prescribed integer event counts.
  x$n.I <- counts
  x$maxn.IPlan <- max(counts)
  times <- unlist(p$spending_times)
  fit <- toBinomialExact(x, usTime = times)
  bound <- fit$lower$bound
  list(
    sample_sizes = counts,
    lower_boundaries = if (reflect) rep(-1, length(counts)) else bound,
    upper_boundaries = if (reflect) counts - bound else counts + 1,
    cumulative_error = cumsum(unname(fit$lower$prob[, 1L])),
    cumulative_alpha = spending(alpha, default(times, counts / max(counts)), parameter)$spend
  )
}

results <- lapply(cases, function(case) {
  n <- unlist(case$sample_sizes)
  theta <- unlist(case$probabilities)
  origin <- 0
  if (case$kind == "calibration") {
    return(c(list(id = case$id), calibrate(case)))
  }
  if (case$kind == "sprt") {
    p <- case$options
    fit <- binomialSPRT(
      p0 = default(p$p0, .05), p1 = default(p$p1, .25), alpha = default(p$alpha, .1), beta = default(p$beta, .15),
      minn = default(p$min_sample_size, 10), maxn = default(p$max_sample_size, 35)
    )
    return(c(list(
      id = case$id, lower_boundaries = fit$lower$bound, upper_boundaries = fit$upper$bound,
      upper_intercept = fit$intercept[1L], lower_intercept = fit$intercept[2L], slope = fit$slope
    ), observed(fit)))
  }
  if (case$kind == "harm") {
    fit <- gsDesign:::gsBinomialExactHarm(
      theta, n, unlist(case$efficacy), unlist(case$futility), unlist(case$harm),
      unlist(case$test_futility), unlist(case$test_harm)
    )
    return(c(list(id = case$id, harm = unname(fit$harm$prob), futility = unname(fit$futility$prob)), observed(fit)))
  }
  a <- unlist(case$lower)
  b <- unlist(case$upper)
  if (identical(case$respect_lower, FALSE)) a <- rep(-1, length(n))
  if (case$kind == "conditional") {
    look <- case$look + 1L
    future <- seq.int(look + 1L, length(n))
    origin <- n[look]
    n <- n[future] - origin
    a <- pmax(-1, a[future] - case$successes)
    b <- b[future] - case$successes
  }
  fit <- gsBinomialExact(k = length(n), theta = theta, n.I = n, a = a, b = b)
  c(list(id = case$id), observed(fit, origin))
})

cat(jsonlite::toJSON(list(
  versions = list(R = as.character(getRversion()), gsDesign = as.character(utils::packageVersion("gsDesign"))),
  cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
