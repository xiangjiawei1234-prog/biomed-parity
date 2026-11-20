args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)

default <- function(x, value) if (is.null(x)) value else x

rate <- function(x) {
  if (length(x) > 0L && is.list(x[[1]])) {
    return(do.call(rbind, lapply(x, unlist)))
  }
  unlist(x)
}

trial_args <- function(x) {
  list(
    lambdaC = rate(default(x$control_hazards, log(2) / 6)),
    hr = default(x$hazard_ratio, .6),
    hr0 = default(x$null_hazard_ratio, 1),
    eta = rate(default(x$control_dropout, 0)),
    etaE = rate(default(x$experimental_dropout, default(x$control_dropout, 0))),
    gamma = rate(default(x$accrual_rates, 1)),
    R = unlist(default(x$accrual_durations, 12)),
    S = unlist(x$hazard_durations),
    ratio = default(x$allocation_ratio, 1)
  )
}

fixed_args <- function(trial, x) {
  solve <- default(x$solve, "accrual_rate")
  followup <- default(x$minimum_followup, 6)
  if (!is.null(x$final_time) && is.null(x$minimum_followup)) {
    followup <- max(0, x$final_time - sum(trial$R))
  }
  list(
    T = if (solve == "accrual_rate") default(x$final_time, sum(trial$R) + followup),
    minfup = if (solve != "followup_duration") followup,
    alpha = default(x$alpha, .025),
    beta = if ("beta" %in% names(x)) x$beta else .1,
    sided = default(x$sided, 1),
    method = default(x$method, "LachinFoulkes"),
    tol = 1e-8
  )
}

spending <- function(x) {
  if (identical(x, "ldof")) gsDesign::sfLDOF else gsDesign::sfHSD
}

arm_counts <- function(x) {
  list(
    control_events = x$eDC,
    experimental_events = x$eDE,
    control_enrollment = x$eNC,
    experimental_enrollment = x$eNE
  )
}

calendar <- function(trial, times, fraction) {
  check_time <- function(time) {
    qe <- trial$ratio / (1 + trial$ratio)
    control <- gsDesign::eEvents(
      lambda = trial$lambdaC, eta = trial$eta, gamma = trial$gamma * (1 - qe),
      R = trial$R, S = trial$S, T = time
    )
    experimental <- gsDesign::eEvents(
      lambda = trial$lambdaC * trial$hr, eta = trial$etaE, gamma = trial$gamma * qe,
      R = trial$R, S = trial$S, T = time
    )
    list(eDC = control$d, eDE = experimental$d, eNC = control$n, eNE = experimental$n)
  }
  events <- function(time) {
    x <- check_time(time)
    sum(x$eDC, x$eDE)
  }
  end <- tail(times, 1L)
  target <- fraction * events(end)
  root <- stats::uniroot(function(t) events(t) - target, c(1e-6, end), tol = 1e-8)$root
  list(counts = lapply(times, function(t) arm_counts(check_time(t))), event_time = root)
}

results <- lapply(cases, function(case) {
  trial <- trial_args(case$trial)
  if (case$kind == "calendar") {
    return(c(list(id = case$id), calendar(trial, unlist(case$times), case$event_fraction)))
  }
  params <- c(trial, fixed_args(trial, case$options))
  if (case$kind == "fixed") {
    fit <- do.call(gsDesign::nSurv, params)
    return(list(
      id = case$id, counts = arm_counts(fit), analysis_time = fit$T,
      minimum_followup = fit$minfup, power = fit$power, accrual_durations = fit$R,
      accrual_rates = unname(fit$gamma)
    ))
  }
  k <- length(case$fractions)
  p <- case$options
  upper_parameter <- if (identical(p$spending, "ldof")) 1 else -4
  lower_parameter <- if (identical(p$futility_spending, "ldof")) 1 else -2
  design <- do.call(gsDesign::gsSurv, c(params, list(
    k = k, timing = unlist(case$fractions), test.type = default(p$test_type, 4),
    sfu = spending(p$spending), sfupar = default(p$parameter, upper_parameter),
    sfl = spending(p$futility_spending), sflpar = default(p$futility_parameter, lower_parameter)
  )))
  list(
    id = case$id, counts = arm_counts(design), analysis_times = design$T,
    minimum_followup = design$minfup, power = sum(design$upper$prob[, 2]),
    accrual_rates = unname(design$gamma), accrual_durations = design$R,
    event_targets = design$n.I, fixed_events = design$n.fix,
    upper_boundaries = design$upper$bound, lower_boundaries = design$lower$bound,
    alternative_effect = design$theta[2] * sqrt(tail(design$n.I, 1L))
  )
})

cat(jsonlite::toJSON(list(
  versions = list(R = as.character(getRversion()), gsDesign = as.character(utils::packageVersion("gsDesign"))),
  cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
