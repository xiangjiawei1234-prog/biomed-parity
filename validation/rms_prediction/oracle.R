args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(rms))

default <- function(x, value) if (is.null(x)) value else x
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
  ref <- default(p$ref_zero, FALSE)
  settings <- lapply(case$vary, function(value) if (is.null(value)) NA else unlist(value))
  pred <- Predict(
    fit, factors = settings, np = default(p$points, 200), ref.zero = ref,
    conf.int = default(p$confidence_level, .95), fun = fun
  )
  linear <- predictrms(fit, newdata = as.data.frame(pred), type = "lp", kint = 1L, ref.zero = ref, se.fit = TRUE)
  list(
    id = case$id, coefficients = unname(coef(fit)), covariance = unname(vcov(fit)),
    center = if (case$model == "cph") unname(fit$means), residual_df = fit$df.residual,
    grid = lapply(pred[names(pred) %in% names(predictors)], unname),
    linear_predictor = unname(linear$linear.predictors), standard_error = unname(linear$se.fit),
    estimate = pred$yhat, lower = pred$lower, upper = pred$upper
  )
}

results <- lapply(cases, function(case) if (case$kind == "summary") summarize(case) else make_predictions(case))
cat(jsonlite::toJSON(list(
  versions = list(R = as.character(getRversion()), rms = as.character(utils::packageVersion("rms"))),
  predictors = lapply(predictors, unname), factor_levels = list(arm = levels(predictors$arm)),
  knots = knots, cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
