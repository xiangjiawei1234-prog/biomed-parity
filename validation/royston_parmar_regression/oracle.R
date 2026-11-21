# Reference fit, likelihood, and prediction values for the Royston-Parmar
# regression in biomed_parity.royston_parmar_regression.
#
# Usage: Rscript --vanilla oracle.R cases.json
#
# Parameter mapping between flexsurv and the Python module:
#   * cases.json stores the complete knot vector (boundary knots first and
#     last). flexsurvspline() expects interior knots in `knots` and the two
#     boundary knots in `bknots`; both are on the scale selected by
#     `timescale` (log time by default), as in the Python module.
#   * `gamma` follows flexsurv's ordering: intercept, slope, then one
#     coefficient per interior knot. The main design multiplies `gamma0`
#     (flexsurv's "location" parameter); an ancillary design keyed by
#     index j multiplies `gamma_j` through flexsurvreg(anc = ...).
#   * The Weibull special case (no interior knots) is fitted both as a
#     zero-knot spline (same gamma parameterisation) and with
#     flexsurvreg(dist = "weibull"). flexsurv stores the covariance on the
#     log shape / log scale estimation scale, so the weibull covariance is
#     mapped to the gamma parameterisation with the delta method:
#     gamma0 = -shape * log(scale), gamma1 = shape.
#   * `predict` blocks are evaluated with summary(fit, type = ...) for
#     survival, cumulative hazard, and hazard when no entry time is
#     supplied, and with the exported flexsurv spline distribution functions
#     otherwise. Density and probability come from dsurvspline() and
#     psurvspline(); entry times condition on survival to entry.
#   * The likelihood-only cases evaluate the documented likelihood directly
#     from dsurvspline()/psurvspline() at the supplied gamma, so they do not
#     depend on the optimiser.
#
# The synthetic data in cases.json were generated from Weibull,
# log-logistic, log-normal, and Royston-Parmar survival models; they are
# stored explicitly so R and Python fit exactly the same values.
suppressPackageStartupMessages(library(flexsurv))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
cases <- input$cases

`%||%` <- function(value, fallback) if (is.null(value)) fallback else value
num <- function(value) {
    vapply(value, function(item) if (is.null(item)) Inf else as.numeric(item), numeric(1))
}
maybe <- function(value) if (is.null(value)) NULL else num(value)

# Weighted log likelihood for exact, left-, right-, and interval-censored
# observations, optionally conditioned on survival past an entry time.
manual_loglik <- function(left, right, entry, weights, gamma, knots, scale, timescale) {
    n <- length(left)
    gamma <- if (is.matrix(gamma)) gamma else matrix(gamma, n, length(gamma), byrow = TRUE)
    knots <- matrix(knots, n, length(knots), byrow = TRUE)
    entry <- entry %||% rep(0, n)
    weights <- weights %||% rep(1, n)
    exact <- left == right & left > 0
    left_censored <- left == 0 & is.finite(right)
    right_censored <- is.finite(left) & is.infinite(right)
    interval <- left > 0 & is.finite(right) & right > left
    log_probability <- numeric(n)
    if (any(exact)) {
        log_probability[exact] <- dsurvspline(left[exact], gamma[exact, , drop = FALSE],
            knots = knots[exact, , drop = FALSE], scale = scale, timescale = timescale,
            log = TRUE)
    }
    if (any(left_censored)) {
        log_probability[left_censored] <- psurvspline(right[left_censored],
            gamma[left_censored, , drop = FALSE], knots = knots[left_censored, , drop = FALSE],
            scale = scale, timescale = timescale, log.p = TRUE)
    }
    if (any(right_censored)) {
        log_probability[right_censored] <- psurvspline(left[right_censored],
            gamma[right_censored, , drop = FALSE], knots = knots[right_censored, , drop = FALSE],
            scale = scale, timescale = timescale, lower.tail = FALSE, log.p = TRUE)
    }
    if (any(interval)) {
        upper <- psurvspline(left[interval], gamma[interval, , drop = FALSE],
            knots = knots[interval, , drop = FALSE], scale = scale, timescale = timescale,
            lower.tail = FALSE, log.p = TRUE)
        lower <- psurvspline(right[interval], gamma[interval, , drop = FALSE],
            knots = knots[interval, , drop = FALSE], scale = scale, timescale = timescale,
            lower.tail = FALSE, log.p = TRUE)
        log_probability[interval] <- upper + log1p(-exp(lower - upper))
    }
    log_entry <- psurvspline(entry, gamma, knots = knots, scale = scale,
        timescale = timescale, lower.tail = FALSE, log.p = TRUE)
    sum(weights * (log_probability - log_entry))
}

# Build the flexsurv response and covariates for one case.
case_data <- function(case) {
    left <- num(case$left)
    right <- num(case$right)
    data <- data.frame(left = left, right = right)
    response <- case$response %||% "right"
    if (response == "counting") {
        data$entry <- num(case$entry)
        data$stop <- left
        data$event <- as.numeric(is.finite(right))
        formula <- Surv(entry, stop, event) ~ 1
    } else if (response == "interval2") {
        data$l <- left
        data$r <- right
        formula <- Surv(l, r, type = "interval2") ~ 1
    } else {
        data$time <- left
        data$event <- as.numeric(is.finite(right))
        formula <- Surv(time, event) ~ 1
    }
    if (!is.null(case$design)) data$x <- num(case$design)
    if (!is.null(case$ancillary)) data$z <- num(case$ancillary[["1"]])
    if (!is.null(case$design)) formula <- update(formula, . ~ . + x)
    list(data = data, formula = formula)
}

# Fit with a tight optimiser tolerance. The spline distribution has no
# analytic derivative for scale = "normal" (and interval-censored terms are
# evaluated through probability differences), so the finite-difference step
# is reduced to make R's stopping point match Python's converged optimum.
fit_reference <- function(case) {
    prepared <- case_data(case)
    knots <- num(case$knots)
    n_parameters <- length(knots) + (!is.null(case$design)) + (!is.null(case$ancillary))
    arguments <- list(formula = prepared$formula, data = prepared$data,
        knots = knots[-c(1L, length(knots))], bknots = knots[c(1L, length(knots))],
        scale = case$scale %||% "hazard", timescale = case$timescale %||% "log",
        control = list(reltol = 1e-14, maxit = 5000,
            ndeps = rep(1e-6, n_parameters)))
    if (!is.null(case$ancillary)) arguments$anc <- list(gamma1 = ~z)
    if (!is.null(case$weights)) arguments$weights <- num(case$weights)
    if (!is.null(case$inits)) arguments$inits <- num(case$inits)
    fit <- do.call(flexsurvspline, arguments)
    est <- unname(fit$res[, "est"])
    names(est) <- rownames(fit$res)
    list(fit = fit, est = est, vcov = unname(vcov(fit)),
        loglik = as.numeric(logLik(fit)))
}

# Per-observation gamma implied by the fitted baseline and the regression
# coefficients.
parameter_gamma <- function(case, est) {
    left <- num(case$left)
    knots <- num(case$knots)
    gamma <- matrix(est[seq_along(knots)], length(left), length(knots), byrow = TRUE)
    if (!is.null(case$design)) gamma[, 1L] <- gamma[, 1L] + num(case$design) * est[["x"]]
    ancillary_names <- grep("^gamma\\d+\\(", names(est), value = TRUE)
    for (name in ancillary_names) {
        index <- as.integer(sub("^gamma(\\d+)\\(.*\\)$", "\\1", name))
        gamma[, index + 1L] <- gamma[, index + 1L] +
            num(case$ancillary[[as.character(index)]]) * est[[name]]
    }
    gamma
}

spline_values <- function(times, gamma, knots, scale, timescale) {
    basic <- list(gamma = gamma, knots = knots, scale = scale, timescale = timescale)
    list(survival = do.call(psurvspline, c(list(times), basic, list(lower.tail = FALSE))),
        cumulative_hazard = do.call(Hsurvspline, c(list(times), basic)),
        hazard = do.call(hsurvspline, c(list(times), basic)),
        density = do.call(dsurvspline, c(list(times), basic)),
        cdf = do.call(psurvspline, c(list(times), basic)))
}

# Gamma for one prediction profile: baseline plus the supplied design and
# ancillary values.
profile_gamma <- function(case, reference, profile) {
    knots <- num(case$knots)
    gamma <- reference$est[seq_along(knots)]
    if (!is.null(case$predict_design)) {
        gamma[1L] <- gamma[1L] + sum(num(case$predict_design[[profile]]) * reference$est[["x"]])
    }
    for (name in grep("^gamma\\d+\\(", names(reference$est), value = TRUE)) {
        index <- as.integer(sub("^gamma(\\d+)\\(.*\\)$", "\\1", name))
        values <- num(case$predict_ancillary[[as.character(index)]][[profile]])
        gamma[index + 1L] <- gamma[index + 1L] + sum(values * reference$est[[name]])
    }
    gamma
}

prediction_spec <- function(case, reference, spec) {
    times <- num(spec$time)
    knots <- num(case$knots)
    scale <- case$scale %||% "hazard"
    timescale <- case$timescale %||% "log"
    entry <- maybe(spec$entry)
    profiles <- max(length(spec$design) %||% 1L, length(entry) %||% 1L, 1L)
    case$predict_design <- spec$design
    case$predict_ancillary <- spec$ancillary
    rows <- list()
    for (profile in seq_len(profiles)) {
        gamma <- profile_gamma(case, reference, profile)
        values <- spline_values(times, gamma, knots, scale, timescale)
        if (!is.null(entry)) {
            # Condition on survival to the entry time: f(t)/S(entry),
            # S(t)/S(entry), h(t), and zero values before entry.
            initial <- spline_values(entry[profile], gamma, knots, scale, timescale)$survival
            before <- times < entry[profile]
            values$survival <- ifelse(before, 1, values$survival / initial)
            values$cumulative_hazard <- ifelse(before, 0, -log(values$survival))
            values$hazard <- ifelse(before, 0, values$hazard)
            values$density <- ifelse(before, 0, values$density / initial)
            values$cdf <- ifelse(before, 0, 1 - values$survival)
        }
        rows[[profile]] <- values
    }
    pick <- function(field) do.call(rbind, lapply(rows, `[[`, field))
    prediction <- list(survival = pick("survival"),
        cumulative_hazard = pick("cumulative_hazard"), hazard = pick("hazard"),
        density = pick("density"), cdf = pick("cdf"))
    if (is.null(entry)) {
        newdata <- NULL
        if (!is.null(spec$design) || !is.null(spec$ancillary)) {
            newdata <- data.frame(row = seq_len(profiles))
            if (!is.null(spec$design)) newdata$x <- vapply(seq_len(profiles),
                function(profile) num(spec$design[[profile]]), numeric(1))
            if (!is.null(spec$ancillary)) newdata$z <- vapply(seq_len(profiles),
                function(profile) num(spec$ancillary[["1"]][[profile]]), numeric(1))
        }
        # Canonical summary.flexsurvreg output for the three distribution
        # quantities that it supports.
        for (type in c("survival", "cumhaz", "hazard")) {
            summary_fit <- summary(reference$fit, type = type, t = times, newdata = newdata)
            prediction[[paste0(type, "_summary")]] <- do.call(rbind,
                lapply(summary_fit, `[[`, "est"))
        }
    }
    prediction
}

predict_reference <- function(case, reference) {
    lapply(case$predict, function(spec) prediction_spec(case, reference, spec))
}

# Built-in Weibull reference for the zero-interior-knot special case.
weibull_reference <- function(case) {
    data <- case_data(case)$data
    fit <- flexsurvreg(Surv(time, event) ~ 1, data = data, dist = "weibull")
    shape <- fit$res["shape", "est"]
    scale_estimate <- fit$res["scale", "est"]
    # flexsurv's covariance is on the log shape / log scale estimation scale.
    jacobian <- matrix(c(-shape * log(scale_estimate), -shape, shape, 0), 2L, 2L, byrow = TRUE)
    list(shape = shape, scale = scale_estimate,
        gamma = c(-shape * log(scale_estimate), shape),
        loglik = as.numeric(logLik(fit)),
        vcov = unname(jacobian %*% fit$cov %*% t(jacobian)))
}

evaluate <- function(case) {
    if (identical(case$kind, "loglik")) {
        gamma <- case$gamma
        per_row <- is.list(gamma) && length(gamma) > 0L && is.list(gamma[[1L]])
        gamma <- if (per_row) do.call(rbind, lapply(gamma, num)) else num(gamma)
        return(list(log_likelihood = manual_loglik(num(case$left), num(case$right),
            maybe(case$entry), maybe(case$weights), gamma, num(case$knots),
            case$scale %||% "hazard", case$timescale %||% "log")))
    }
    reference <- fit_reference(case)
    gamma <- parameter_gamma(case, reference$est)
    result <- list(est = reference$est, names = names(reference$est),
        loglik = reference$loglik, vcov = reference$vcov,
        loglik_at_est = manual_loglik(num(case$left), num(case$right),
            maybe(case$entry), maybe(case$weights), gamma, num(case$knots),
            case$scale %||% "hazard", case$timescale %||% "log"),
        predictions = predict_reference(case, reference))
    if (isTRUE(case$weibull)) result$weibull <- weibull_reference(case)
    result
}

results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1L))
versions <- list(R = as.character(getRversion()),
    flexsurv = as.character(packageVersion("flexsurv")),
    survival = as.character(packageVersion("survival")))
cat(jsonlite::toJSON(list(versions = versions, results = results), auto_unbox = TRUE,
    digits = NA, null = "null", na = "null", pretty = TRUE), "\n", sep = "")
