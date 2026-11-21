args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(rms))

fit_case <- function(case) {
  i <- seq_len(32)
  d <- data.frame(x = seq(-2, 2, length.out = length(i)), z = sin(i), y = as.integer((i * 5) %% 9 >= 4))
  weights <- switch(case$weights %||% "none", frequency = i %% 5, fractional = (i %% 6) / 4, none = NULL)
  normwt <- isTRUE(case$normwt)
  offset <- if (identical(case$offset, "varying")) cos(i / 3) / 2 else case$offset %||% 0
  d$offset_value <- offset
  controls <- list(gradtol = 1e-9, eps = 1e-11)
  null <- do.call(lrm.fit, c(list(y = d$y, offset = offset, weights = weights, normwt = normwt), controls))
  stopifnot(!null$fail)

  if (isTRUE(case$intercept_only)) {
    fit <- null
    x <- matrix(numeric(0), nrow(d), 0)
  } else {
    formula <- if (any(offset != 0)) y ~ x + z + offset(offset_value) else y ~ x + z
    arguments <- c(list(formula, data = d, x = TRUE, weights = weights, normwt = normwt), controls)
    if (!is.null(case$penalty)) arguments$penalty <- case$penalty
    fit <- suppressWarnings(do.call(lrm, arguments))
    stopifnot(!fit$fail)
    x <- fit$x
  }
  covariance <- as.matrix(infoMxop(fit$info.matrix, invert = TRUE))
  p <- ncol(x)
  penalty <- fit$penalty.matrix %||% matrix(0, p, p)
  # lrm's post-fit adjustment drops weights and compares against a no-offset
  # null. Evaluate the weighted unpenalized likelihood at the fitted estimates.
  evaluated <- lrm.fit(x, d$y, offset = offset, weights = weights, normwt = normwt,
                       initial = coef(fit), maxit = 1, compstats = FALSE)
  df <- diag(as.matrix(infoMxop(evaluated$info.matrix)) %*% covariance)
  null_deviance <- tail(null$deviance, 1)
  deviance <- tail(evaluated$deviance, 1)
  lr <- if (p) null_deviance - deviance else 0
  effective <- sum(df[-1])
  # lrm.fit stores X beta without the offset for slope models and omits
  # the training linear predictors for intercept-only models.
  linear <- drop(coef(fit)[1] + x %*% coef(fit)[-1] + offset)
  list(
    id = case$id, x = if (p) unname(x), y = d$y, input_weights = weights, input_offset = offset,
    coefficients = as.list(coef(fit) |> unname()), covariance = unname(covariance),
    penalty_matrix = unname(penalty), linear_predictor = unname(linear), probabilities = unname(plogis(linear)),
    effective_df_diagonal = as.list(unname(df)), effective_df = effective,
    null_deviance = null_deviance, deviance = deviance, penalized_deviance = tail(fit$deviance, 1),
    likelihood_ratio = lr, pvalue = if (p) pchisq(lr, effective, lower.tail = FALSE) else 1
  )
}

`%||%` <- function(x, default) if (is.null(x)) default else x
result <- list(
  versions = list(R = as.character(getRversion()), rms = as.character(packageVersion("rms"))),
  cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
