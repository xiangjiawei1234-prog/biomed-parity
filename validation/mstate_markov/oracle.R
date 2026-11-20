args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)

default <- function(x, value) if (is.null(x)) value else unlist(x)
rows <- function(x) do.call(rbind, lapply(x, unlist))

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
  records <- mstate::msprep(time = time, status = status, trans = transition_matrix, id = id, start = start)
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
  variance <- vapply(seq_along(fit$qualset), function(i) fit$est_cov[, i, i], numeric(length(grid)))

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
  summaries <- function(fit) list(
    state_statistics = as.numeric(fit$orig_stat), overall_statistic = fit$orig_ch_stat,
    bootstrap_state_statistics = matrix(fit$b_stat_wb, nrow = case$replicates),
    bootstrap_overall_statistics = apply(fit$nch_wb_trace, 1L, overall_fn),
    state_pvalues = as.numeric(fit$p_stat_wb), overall_pvalue = fit$p_ch_stat_wb
  )
  list(
    id = case$id, qualifying_states = as.integer(fit$qualset - 1L), n_events_after = fit$nobs_grid, n_subjects = fit$Nsub,
    event_subject_ids = history$id[observed], event_times = history$exit[observed], multipliers = multipliers,
    score_trace = fit$zbar * sqrt(variance + (variance == 0)), z_trace = fit$zbar,
    covariance = fit$est_cov, chi_squared_trace = fit$obs_chisq_trace,
    state_statistics = c(fit$orig_stat), overall_statistic = fit$orig_ch_stat,
    bootstrap_z_trace = fit$n_wb_trace, bootstrap_chi_squared_trace = fit$nch_wb_trace,
    bootstrap_state_statistics = matrix(fit$b_stat_wb, nrow = case$replicates),
    bootstrap_overall_statistics = rowMeans(fit$nch_wb_trace), z_quantiles = fit$est_quant,
    state_pvalues = c(fit$p_stat_wb), overall_pvalue = fit$p_ch_stat_wb,
    weights = list(
      state = state_weights, overall = overall_weights,
      optimal = summaries(weighted_fit(state_functions)),
      r_functions = summaries(weighted_fit(r_functions))
    )
  )
})

cat(jsonlite::toJSON(list(
  versions = list(R = as.character(getRversion()), mstate = as.character(utils::packageVersion("mstate"))),
  cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
