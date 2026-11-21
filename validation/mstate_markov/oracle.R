args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)

default <- function(x, value) if (is.null(x)) value else unlist(x)
rows <- function(x) do.call(rbind, lapply(x, unlist))

# Reuse MarkovTest with the offset and dimension-wise covariance centering.
full_cox_test <- mstate::MarkovTest
repairs <- c(offset = 0L, row = 0L, state = 0L)
repair_cox_trace <- function(node) {
  if (is.call(node) && identical(node[[1L]], as.name("if")) &&
      identical(node[[2L]], quote(!is.null(progfit$offset)))) {
    repairs["offset"] <<- repairs["offset"] + 1L
    return(quote({
      offset <- stats::model.offset(stats::model.frame(form, data = rdata))
      if (is.null(offset)) offset <- rep(0, nrow(rdata))
    }))
  }
  if (identical(node, quote(Zmat[v, ] - Zbar0))) {
    repairs["row"] <<- repairs["row"] + 1L
    return(quote(-sweep(Zbar0, 2L, Zmat[v, ], "-")))
  }
  if (identical(node, quote(Z3mat[v, k, indx2] - Zbar[, k, indx1]))) {
    repairs["state"] <<- repairs["state"] + 1L
    return(quote(Z3mat[v, k, indx2] - Zbar[, k, indx2]))
  }
  if (is.call(node)) as.call(lapply(as.list(node), repair_cox_trace)) else node
}
body(full_cox_test) <- repair_cox_trace(body(full_cox_test))
stopifnot(all(repairs == c(1L, 2L, 1L)))

results <- lapply(cases, function(case) {
  time <- rows(case$time)
  status <- rows(case$status)
  edges <- rows(case$transitions) + 1L
  transition_matrix <- matrix(NA_integer_, ncol(time), ncol(time))
  transition_matrix[edges] <- seq_len(nrow(edges))
  id <- default(case$subject_ids, seq_len(nrow(time)))
  start <- list(
    state = default(case$start_state, rep(0, nrow(time))) + 1L,
    time = default(case$start_time, rep(0, nrow(time)))
  )
  prep <- list(time = time, status = status, trans = transition_matrix, id = id, start = start)
  if (!is.null(case$covariate_data)) prep$keep <- as.data.frame(lapply(case$covariate_data, unlist))
  records <- do.call(mstate::msprep, prep)
  tr <- case$transition + 1L
  grid <- unlist(case$grid)
  min_time <- default(case$min_time, 0)
  history <- mstate::msdata2etm(records)
  weight_states <- sort(unique(history$from))
  history <- history[history$from == edges[tr, 1L], ]
  observed <- history$to == edges[tr, 2L]

  # Match the stream consumed by MarkovTest, then retain the event columns.
  set.seed(case$seed)
  count <- case$replicates * nrow(history)
  draws <- if (case$distribution == "poisson") rpois(count, 1) - 1 else rnorm(count)
  multipliers <- matrix(draws, nrow = case$replicates, byrow = TRUE)[, observed, drop = FALSE]
  set.seed(case$seed)
  fit <- mstate::MarkovTest(
    records, transition = tr, grid = grid, B = case$replicates, dist = case$distribution
  )

  r_functions <- mstate::optimal_weights_matrix(records, grid = grid, transition = tr, min_time = min_time)
  state_weights <- attr(r_functions, "weights")
  overall_weights <- mstate::optimal_weights_multiple(records, grid = grid, transition = tr, min_time = min_time)
  overall_fn <- function(x) weighted.mean(x, w = overall_weights)
  widths <- diff(c(min_time, grid))
  state_functions <- lapply(match(fit$qualset, weight_states), function(col) {
    w <- state_weights[, col] * widths
    list(function(x) weighted.mean(abs(x), w = w))
  })
  weighted_fit <- function(fn) {
    set.seed(case$seed)
    mstate::MarkovTest(
      records, transition = tr, grid = grid, B = case$replicates, dist = case$distribution,
      fn = fn, fn2 = overall_fn
    )
  }
  summaries <- function(fit, summary_fn = overall_fn) list(
    state_statistics = as.numeric(fit$orig_stat), overall_statistic = fit$orig_ch_stat,
    bootstrap_state_statistics = matrix(fit$b_stat_wb, nrow = case$replicates),
    bootstrap_overall_statistics = apply(fit$nch_wb_trace, 1L, summary_fn),
    state_pvalues = as.numeric(fit$p_stat_wb), overall_pvalue = fit$p_ch_stat_wb
  )
  traces <- function(fit) {
    variance <- vapply(seq_along(fit$qualset), function(i) fit$est_cov[, i, i], numeric(length(grid)))
    c(list(
      score_trace = fit$zbar * sqrt(variance + (variance == 0)), z_trace = fit$zbar,
      covariance = fit$est_cov, chi_squared_trace = fit$obs_chisq_trace,
      bootstrap_z_trace = fit$n_wb_trace, bootstrap_chi_squared_trace = fit$nch_wb_trace,
      z_quantiles = fit$est_quant
    ), summaries(fit, mean))
  }
  adjustments <- lapply(case$adjustments, function(adjustment) {
    test <- if (identical(adjustment$cox_mode, "full")) full_cox_test else mstate::MarkovTest
    terms <- c(unlist(adjustment$covariates))
    if (!is.null(adjustment$offset)) terms <- c(terms, sprintf("offset(%s)", adjustment$offset))
    set.seed(case$seed)
    fit <- test(
      records, transition = tr, grid = grid, B = case$replicates, dist = case$distribution,
      formula = paste(terms, collapse = " + ")
    )
    c(list(
      id = adjustment$id,
      cox_fit = list(
        covariates = I(default(adjustment$covariates, character())), coefficients = I(as.numeric(stats::coef(fit$coxfit))),
        covariance = if (length(stats::coef(fit$coxfit))) stats::vcov(fit$coxfit) else list(),
        log_likelihood = tail(fit$coxfit$loglik, 1L), null_log_likelihood = fit$coxfit$loglik[[1L]]
      )
    ), traces(fit))
  })
  c(list(
    id = case$id, qualifying_states = as.integer(fit$qualset - 1L), n_events_after = fit$nobs_grid, n_subjects = fit$Nsub,
    event_subject_ids = history$id[observed], event_times = history$exit[observed], multipliers = multipliers
  ), traces(fit), list(
    weights = list(
      state = state_weights, overall = overall_weights,
      optimal = summaries(weighted_fit(state_functions)),
      r_functions = summaries(weighted_fit(r_functions))
    ),
    adjustments = adjustments
  ))
})

cat(jsonlite::toJSON(list(
  versions = list(
    R = as.character(getRversion()), mstate = as.character(utils::packageVersion("mstate")),
    survival = as.character(utils::packageVersion("survival"))
  ),
  cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
