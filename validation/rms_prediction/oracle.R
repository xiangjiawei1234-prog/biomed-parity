args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(rms))

default <- function(x, value) if (is.null(x)) value else x

simultaneous_critical <- function(object, level) {
  correlation <- cov2cor(vcov(object))
  size <- nrow(correlation)
  df <- object$df
  upper_prob <- (1 - level) / (2 * size)
  bound <- if (df > 0) qt(upper_prob, df, lower.tail = FALSE) else qnorm(upper_prob, lower.tail = FALSE)
  # Keep the randomized integration deterministic across root evaluations.
  # Singular prediction correlation matrices need a larger integration budget.
  coverage <- function(critical) {
    set.seed(809)
    probability <- mvtnorm::pmvt(rep(-critical, size), rep(critical, size), df = df, corr = correlation,
                                algorithm = mvtnorm::GenzBretz(maxpts = 2000000, abseps = 2e-5))
    as.numeric(probability) - level
  }
  uniroot(coverage, c(0, bound), tol = 1e-7, extendInt = "upX")$root
}

column <- function(x) {
  missing <- if (any(vapply(x, is.character, TRUE))) NA_character_ else NA_real_
  vapply(x, function(value) default(value, missing), missing)
}

summarize <- function(case) {
  d <- if (identical(case$preset, "varying_n")) {
    data.frame(x = seq_len(300), z = c(seq_len(225)^2 / 100, rep(NA_real_, 75)))
  } else {
    as.data.frame(lapply(case$data, column), stringsAsFactors = FALSE)
  }
  for (name in names(case$factor_levels)) d[[name]] <- factor(d[[name]], levels = unlist(case$factor_levels[[name]]))
  p <- case$options
  settings <- list(
    q.effect = unlist(default(p$q_effect, c(.25, .75))),
    adjto.cat = default(p$categorical_adjustment, "mode"),
    n.unique = default(p$n_unique, 10)
  )
  if (!is.null(p$q_display)) settings$q.display <- unlist(p$q_display)
  fit <- do.call(datadist, c(list(d), settings))
  limits <- lapply(fit$limits, function(x) if (is.factor(x)) as.character(x) else unname(x))
  list(id = case$id, limits = limits, values = fit$values)
}

i <- seq_len(88)
predictors <- data.frame(
  x = seq(-2.6, 2.6, length.out = length(i)),
  arm = factor(rep(c("standard", "standard", "targeted", "watch"), length.out = length(i)),
               levels = c("watch", "standard", "targeted")),
  z = (i %% 17L - 8L) / 5
)
knots <- c(-2, -0.5, 0.55, 2)
make_predictions <- function(case) {
  d <- predictors
  p <- case$options
  dd <- datadist(d)
  for (name in names(p$adjust_to)) dd$limits[[name]][2] <- p$adjust_to[[name]]
  old <- options(datadist = dd)
  on.exit(options(old))
  lp <- 0.35 * d$x - 0.12 * d$x^2 + 0.6 * (d$arm == "standard") + 0.25 * d$z
  set.seed(177)
  if (case$model == "ols") {
    d$y <- 1 + lp + d$z + sin(i * 1.83) + .7 * cos(i * .87)
    fit <- ols(y ~ rcs(x, knots) * arm + z, data = d)
  } else if (case$model == "lrm") {
    d$y <- rbinom(nrow(d), 1, plogis(lp))
    fit <- lrm(y ~ rcs(x, knots) + arm + z, data = d)
  } else {
    stop <- rexp(nrow(d), exp(lp))
    censor <- rexp(nrow(d), .45)
    d$time <- pmin(stop, censor)
    d$status <- as.integer(stop <= censor)
    fit <- cph(survival::Surv(time, status) ~ rcs(x, knots) + arm + z, data = d)
  }
  fun <- switch(default(p$transform, "identity"), exp = exp, logistic = plogis, identity = identity)
  inverse <- switch(default(p$transform, "identity"), exp = log, logistic = qlogis, identity = identity)
  ref <- default(p$ref_zero, FALSE)
  level <- default(p$confidence_level, .95)
  type <- default(p$confidence_type, "mean")
  # rms 8.2-1 leaves the intercept column undefined for simultaneous ref.zero bands.
  direct_contrast <- ref && type == "simultaneous" && case$model != "cph"
  settings <- lapply(case$vary, function(value) if (is.null(value)) NA else unlist(value))
  set.seed(809)
  pred <- Predict(
    fit, factors = settings, np = default(p$points, 200), ref.zero = ref,
    conf.int = if (direct_contrast) FALSE else level, conf.type = type, fun = fun
  )
  linear <- predictrms(fit, newdata = as.data.frame(pred), type = "lp", kint = 1L, ref.zero = ref, se.fit = TRUE)
  rms_critical <- NULL
  if (type == "simultaneous") {
    if (!direct_contrast) {
      j <- which.max(linear$se.fit)
      rms_critical <- (inverse(pred$upper[j]) - inverse(pred$lower[j])) / (2 * linear$se.fit[j])
    }
    x <- predictrms(fit, newdata = as.data.frame(pred), type = "x")
    if (ref) {
      x <- sweep(x, 2L, as.vector(predictrms(fit, type = "adjto")))
    } else if (case$model == "cph") {
      x <- sweep(x, 2L, fit$means)
    }
    contrasts <- if (case$model == "cph") x else cbind(if (ref) 0 else 1, x)
    nonzero <- linear$se.fit > 0
    bounds <- confint(multcomp::glht(fit, contrasts[nonzero, , drop = FALSE], df = default(fit$df.residual, 0)),
                      level = level, calpha = simultaneous_critical)$confint
    pred$lower <- pred$upper <- pred$yhat
    pred$lower[nonzero] <- fun(bounds[, "lwr"])
    pred$upper[nonzero] <- fun(bounds[, "upr"])
  }
  result <- list(
    id = case$id, coefficients = unname(coef(fit)), covariance = unname(vcov(fit)),
    center = if (case$model == "cph") unname(fit$means), residual_df = fit$df.residual,
    grid = lapply(pred[names(pred) %in% names(predictors)], unname),
    linear_predictor = unname(linear$linear.predictors), standard_error = unname(linear$se.fit),
    estimate = pred$yhat, lower = pred$lower, upper = pred$upper
  )
  if (type == "simultaneous") result$rms_critical <- rms_critical
  result
}

results <- lapply(cases, function(case) if (case$kind == "summary") summarize(case) else make_predictions(case))
cat(jsonlite::toJSON(list(
  versions = list(R = as.character(getRversion()), rms = as.character(utils::packageVersion("rms")),
                  multcomp = as.character(utils::packageVersion("multcomp")),
                  mvtnorm = as.character(utils::packageVersion("mvtnorm"))),
  predictors = lapply(predictors, unname), factor_levels = list(arm = levels(predictors$arm)),
  knots = knots, cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
