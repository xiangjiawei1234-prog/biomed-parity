args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(rms))

fit_case <- function(case) {
  i <- seq_len(84)
  d <- data.frame(x = seq(-2.2, 2.8, length.out = length(i)), z = sin(i * .8),
                  arm = factor(rep(c("A", "C", "B", "C", "A", "B", "C"), length.out = length(i))))
  knots <- c(-1.4, -.25, .8, 2.2)
  set.seed(277)
  d$y <- rbinom(nrow(d), 1, plogis(-.3 + .6 * d$x - .18 * d$x^2 + .5 * d$z + .8 * (d$arm == "B")))
  quadratic <- function(x) {
    basis <- cbind(x, quadratic = x^2)
    attr(basis, "nonlinear") <- 2L
    basis
  }
  formula <- switch(case$design, linear = y ~ gTrans(x, quadratic) + z,
                    factor_spline = y ~ rcs(x, knots) * arm + z,
                    separated = y ~ x, collinear = y ~ x + zz)
  if (case$design == "separated") d$y <- as.integer(d$x > .2)
  if (case$design == "collinear") d$zz <- 2 * d$x + 1

  arguments <- list(formula, data = d, x = TRUE, y = TRUE, gradtol = 1e-10, eps = 1e-12)
  if (!is.null(case$r_penalty)) {
    arguments$penalty <- case$r_penalty
  } else if (!is.null(case$penalty)) {
    arguments$penalty <- unlist(case$penalty)
  }
  if (!is.null(case$penalty_matrix)) arguments$penalty.matrix <- do.call(rbind, lapply(case$penalty_matrix, unlist))
  fit <- do.call(lrm, arguments)
  stopifnot(!fit$fail)
  p <- ncol(fit$x)
  penalty <- if (is.null(fit$penalty.matrix)) matrix(0, p, p) else fit$penalty.matrix
  effective <- if (is.null(fit$effective.df.diagonal)) rep(1, p + 1) else fit$effective.df.diagonal
  lr <- unname(fit$stats["Model L.R."])
  list(
    id = case$id, x = unname(fit$x), y = unname(fit$y),
    coefficients = unname(coef(fit)), covariance = unname(as.matrix(vcov(fit))),
    penalty_matrix = unname(penalty), linear_predictor = unname(fit$linear.predictors),
    probabilities = unname(plogis(fit$linear.predictors)), effective_df_diagonal = unname(effective),
    effective_df = unname(fit$stats["d.f."]), null_deviance = fit$deviance[1],
    deviance = fit$deviance[1] - lr, penalized_deviance = tail(fit$deviance, 1),
    likelihood_ratio = lr, pvalue = unname(fit$stats["P"])
  )
}

result <- list(
  versions = list(R = as.character(getRversion()), rms = as.character(packageVersion("rms"))),
  cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
