args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
suppressPackageStartupMessages(library(mstate))

fit_case <- function(case) {
    keep <- !duplicated(case$times, fromLast = TRUE)
    timing <- case$times[keep]
    predt <- if (is.null(case$predt)) 0 else case$predt
    direction <- if (is.null(case$direction)) "forward" else case$direction
    method <- if (is.null(case$method)) "aalen" else case$method
    transitions <- as.matrix(case$transitions) + 1L
    n <- if (is.null(case$n_states)) max(transitions) else case$n_states
    m <- nrow(transitions)
    trans <- matrix(NA_integer_, n, n)
    trans[transitions] <- seq_len(m)
    hazard <- as.matrix(case$cumulative_hazards)[keep, , drop = FALSE]
    hazard <- data.frame(
        time = rep(timing, m), Haz = as.numeric(hazard),
        trans = rep(seq_len(m), each = length(timing))
    )
    covariance <- NULL
    if (!is.null(case$cumulative_covariance)) {
        values <- case$cumulative_covariance[keep, , , drop = FALSE]
        pairs <- which(upper.tri(matrix(FALSE, m, m), diag = TRUE), arr.ind = TRUE)
        covariance <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(i) {
            data.frame(
                time = timing, varHaz = values[, pairs[i, 1], pairs[i, 2]],
                trans1 = unname(pairs[i, 1]), trans2 = unname(pairs[i, 2])
            )
        }))
    }
    full_covariance <- isTRUE(case$covariance)
    fit <- probtrans(
        structure(list(Haz = hazard, varHaz = covariance, trans = trans), class = "msfit"),
        predt = predt, direction = direction, method = method,
        variance = !is.null(covariance), covariance = full_covariance
    )
    grid <- fit[[1L]]$time
    state_tables <- fit[seq_len(n)]
    extract <- function(columns) lapply(seq_along(grid), function(t) {
        do.call(rbind, lapply(state_tables, function(x) unlist(x[t, columns])))
    })
    aligned_covariance <- NULL
    if (full_covariance) {
        if (direction == "forward") {
            aligned_covariance <- lapply(seq_along(grid), function(t) fit$varMatrix[, , t])
        } else {
            # The v0.3.3 backward recursion writes index t+1 for the SE row t.
            # Restore the matching time labels and append zero at the horizon.
            steps <- sum(timing <= predt)
            aligned_covariance <- lapply(seq_along(grid), function(t) {
                if (t <= steps) fit$varMatrix[, , t + 1L] else matrix(0, n^2, n^2)
            })
        }
    }
    list(
        id = case$id, times = as.list(grid),
        probabilities = extract(paste0("pstate", seq_len(n))),
        standard_errors = if (is.null(covariance)) NULL else extract(paste0("se", seq_len(n))),
        covariance = aligned_covariance
    )
}

cases <- jsonlite::fromJSON(args[[1]], simplifyDataFrame = FALSE)
output <- list(
    versions = list(R = as.character(getRversion()), mstate = as.character(packageVersion("mstate"))),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
