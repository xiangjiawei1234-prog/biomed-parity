args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]])
default <- function(value, fallback) if (is.null(value)) fallback else value
parameter <- function(value) {
    if (is.list(value[[1L]])) do.call(rbind, lapply(value, unlist)) else unlist(value)
}

evaluate <- function(case) {
    options <- list(gamma = parameter(case$gamma), knots = parameter(default(case$knots, list(-10, 10))),
        scale = default(case$scale, "hazard"), timescale = default(case$timescale, "log"))
    x <- unlist(case$time)
    evaluate_at <- function(fun, ...) do.call(fun, c(list(x), options, list(...)))
    cumulative <- evaluate_at(flexsurv::Hsurvspline)
    hazard <- evaluate_at(flexsurv::hsurvspline)
    result <- list(survival = I(evaluate_at(flexsurv::psurvspline, lower.tail = FALSE)),
        cumulative_hazard = I(cumulative), hazard = I(hazard), density = I(evaluate_at(flexsurv::dsurvspline)),
        cdf = I(evaluate_at(flexsurv::psurvspline)), log_survival = I(-cumulative), log_hazard = I(log(hazard)),
        log_density = I(evaluate_at(flexsurv::dsurvspline, log = TRUE)))
    if (!is.null(case$probability)) {
        result$quantile <- I(do.call(flexsurv::qsurvspline, c(list(unlist(case$probability)), options,
            list(lower.tail = default(case$lower_tail, TRUE), log.p = isTRUE(case$log_p)))))
    }
    result
}

results <- lapply(input$cases, evaluate)
names(results) <- vapply(input$cases, function(case) case$id, character(1L))
versions <- list(R = as.character(getRversion()), flexsurv = as.character(packageVersion("flexsurv")))
cat(jsonlite::toJSON(list(versions = versions, results = results), auto_unbox = TRUE,
    digits = NA, na = "string", pretty = TRUE), "\n", sep = "")
