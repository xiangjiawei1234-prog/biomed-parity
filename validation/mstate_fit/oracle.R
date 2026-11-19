suppressPackageStartupMessages(library(mstate))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
fallback <- function(value, default) if (is.null(value)) default else value
parse <- function(cells) unlist(lapply(cells, function(x) if (is.null(x)) NA else x), use.names = FALSE)
matrix_rows <- function(rows) do.call(rbind, lapply(rows, parse))

evaluate <- function(case) {
    dataset <- input$datasets[[case$dataset]]
    endpoints <- matrix_rows(dataset$transitions) + 1L
    size <- nrow(endpoints)
    states <- max(endpoints)
    transitions <- matrix(NA_integer_, states, states)
    transitions[endpoints] <- seq_len(size)
    if (!is.null(dataset$time)) {
        dat <- msprep(matrix_rows(dataset$time), matrix_rows(dataset$status), trans = transitions,
            keep = as.data.frame(lapply(dataset$keep, parse)))
        design <- if (isTRUE(case$adjust_wide)) as.matrix(dat["x"]) else matrix(numeric(), nrow(dat), 0L)
    } else {
        dat <- data.frame(Tstart = parse(dataset$t_start), Tstop = parse(dataset$t_stop),
            status = parse(dataset$status), trans = parse(dataset$transition) + 1L)
        design <- matrix_rows(dataset$covariates)
    }
    names <- if (ncol(design)) paste0("x", seq_len(ncol(design))) else character()
    for (i in seq_along(names)) dat[[names[i]]] <- design[, i]
    formula <- reformulate(c(names, "strata(trans)"), response = "Surv(Tstart, Tstop, status)")
    call <- list(formula, data = dat, ties = "breslow", x = TRUE,
        control = coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 50))
    if (!is.null(case$coefficients)) {
        call$init <- parse(case$coefficients)
        call$control <- coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 0)
    }
    fit <- do.call(coxph, call)
    beta_covariance <- if (!length(names)) matrix(numeric(), 0L, 0L) else fit$var
    if (!is.null(case$coefficients)) {
        beta_covariance <- if (is.null(case$coefficient_covariance)) {
            matrix(0, length(names), length(names))
        } else matrix_rows(case$coefficient_covariance)
        fit$var <- beta_covariance
    }
    vartype <- fallback(case$vartype, "aalen")
    variance <- fallback(case$variance, TRUE)
    args <- list(fit, variance = variance, vartype = vartype, trans = transitions)
    if (length(names)) {
        values <- case$new_covariates
        new_design <- if (is.list(values[[1L]])) matrix_rows(values) else {
            matrix(rep(parse(values), each = size), size, length(names))
        }
        new_data <- data.frame(strata = seq_len(size))
        for (i in seq_along(names)) new_data[[names[i]]] <- new_design[, i]
        args$newdata <- new_data
    }
    result <- do.call(msfit, args)
    times <- sort(unique(result$Haz$time))
    hazards <- vapply(seq_len(size), function(k) result$Haz$Haz[result$Haz$trans == k], numeric(length(times)))
    variance_array <- if (variance) {
        values <- result$varHaz
        lapply(times, function(time) {
            at_time <- values[values$time == time, ]
            mat <- matrix(0, size, size)
            mat[cbind(at_time$trans1, at_time$trans2)] <- at_time$varHaz
            mat[cbind(at_time$trans2, at_time$trans1)] <- at_time$varHaz
            mat
        })
    } else NULL

    prediction <- NULL
    if (!is.null(case$predict)) {
        pred <- probtrans(result, predt = case$predict$predt, covariance = variance,
            variance = variance, method = vartype)
        grid <- pred[[1L]]$time
        state_tables <- pred[seq_len(states)]
        extract <- function(columns) lapply(seq_along(grid), function(i) {
            do.call(rbind, lapply(state_tables, function(tbl) unlist(tbl[i, columns])))
        })
        prediction <- list(
            times = I(grid),
            probabilities = extract(paste0("pstate", seq_len(states))),
            standard_errors = if (variance) extract(paste0("se", seq_len(states))) else NULL,
            covariance = if (variance) lapply(seq_along(grid), function(i) pred$varMatrix[, , i]) else NULL
        )
    }
    list(
        times = I(times), cumulative_hazards = hazards, cumulative_covariance = variance_array,
        coefficients = I(as.numeric(coef(fit))), coefficient_covariance = beta_covariance,
        log_likelihood = tail(fit$loglik, 1), prediction = prediction
    )
}
results <- lapply(input$cases, evaluate)
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), mstate = as.character(packageVersion("mstate")),
    survival = as.character(packageVersion("survival")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
