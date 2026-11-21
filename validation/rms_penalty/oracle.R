args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(rms))

`%||%` <- function(x, default) if (is.null(x)) default else x

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
  formula <- switch(case$design, quadratic = y ~ gTrans(x, quadratic) + z,
                    factor_spline = y ~ rcs(x, knots) * arm + z,
                    factor_interaction = y ~ x * arm)
  fit <- lrm(formula, data = d, x = TRUE, y = TRUE, gradtol = 1e-9, eps = 1e-11)
  stopifnot(!fit$fail)
  penalties <- if (is.null(names(case$penalties))) unlist(case$penalties) else lapply(case$penalties, unlist)
  criterion <- case$criterion %||% "aicc"
  which <- if (criterion == "aicc") "aic.c" else criterion
  call <- list(fit, penalty = penalties, which = which, keep.coef = TRUE,
               complex.more = case$complex_more %||% TRUE, noaddzero = !(case$include_zero %||% TRUE),
               maxit = 50, gradtol = 1e-9, eps = 1e-11)
  if (!is.null(case$penalty_matrix)) call$penalty.matrix <- do.call(rbind, lapply(case$penalty_matrix, unlist))
  trace <- do.call(pentrace, call)
  results <- trace$results.all
  penalty_names <- if (is.null(names(case$penalties))) "penalty" else names(case$penalties)
  best <- trace$fit
  covariance <- as.matrix(infoMxop(best$info.matrix, invert = TRUE))
  effective_diagonal <- if (is.null(best$penalty.matrix)) rep(1, length(coef(best))) else trace$diag
  list(
    id = case$id, x = unname(fit$x), y = unname(fit$y), penalty_names = as.list(penalty_names),
    penalties = unname(as.matrix(results[penalty_names])), effective_df = results$df,
    likelihood_ratio = results$aic + 2 * results$df, aic = results$aic, bic = results$bic, aicc = results$aic.c,
    criterion = criterion, best_index = which.max(results[[which]]) - 1L,
    objective = unname(trace$objective), coefficients = unname(trace$Coefficients),
    best_coefficients = unname(coef(best)), best_covariance = unname(covariance),
    best_effective_df_diagonal = unname(effective_diagonal),
    adjusted_covariance = unname(as.matrix(trace$var.adj))
  )
}

result <- list(
  versions = list(R = as.character(getRversion()), rms = as.character(packageVersion("rms"))),
  cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
