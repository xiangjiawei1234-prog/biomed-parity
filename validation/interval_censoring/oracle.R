args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
evaluate <- function(case) {
    left <- unlist(case$left)
    right <- vapply(case$right, function(value) if (is.null(value)) Inf else value, 0.0)
    bounds <- if (is.null(case$bounds)) c(0, 1) else unlist(case$bounds)
    weights <- if (is.null(case$weights)) rep(1, length(left)) else unlist(case$weights)
    fit <- icenReg::ic_np(cbind(left, right), B = bounds, weights = weights, tol = 1e-13, maxIter = 4000)
    curve <- icenReg::getSCurves(fit)
    intervals <- curve$Tbull_ints[-1, , drop = FALSE]
    mass <- -diff(curve$S_curves$baseline)
    left_open <- (bounds[1] == 0) & (left != right)
    right_open <- (bounds[2] == 0) & (left != right)
    contained <- outer(left + left_open * 1e-10, intervals[, 1], "<=") &
        outer(right - right_open * 1e-10, intervals[, 2], ">=")
    list(intervals = unname(intervals), mass = I(mass),
         log_likelihood = sum(weights * log(as.vector(contained %*% mass))))
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("icenReg")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
