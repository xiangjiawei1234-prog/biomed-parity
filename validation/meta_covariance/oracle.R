args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
parse_vector <- function(values) unlist(lapply(values, function(value) {
    if (is.null(value)) NA_real_ else value
}), use.names = FALSE)
parse_matrix <- function(rows) do.call(rbind, lapply(rows, parse_vector))

evaluate <- function(case) {
    settings <- lapply(case[setdiff(names(case), c("id", "rvars", "rho"))], parse_vector)
    if (is.null(names(case$rho))) {
        if (!is.null(case$rho)) settings$rho <- parse_vector(case$rho)
    } else {
        settings$rho <- parse_matrix(case$rho$matrix)
        labels <- parse_vector(case$rho$levels)
        dimnames(settings$rho) <- list(labels, labels)
    }
    observed_warnings <- character()
    matrix <- withCallingHandlers({
        if (is.null(case$rvars)) {
            do.call(metafor::vcalc, settings)
        } else {
            input <- data.frame(vi = settings$vi, cluster = settings$cluster, parse_matrix(case$rvars))
            names(input)[3:5] <- c("r1", "r2", "r3")
            metafor::vcalc(vi, cluster, data = input, rvars = c(r1, r2, r3))
        }
    }, warning = function(w) {
        observed_warnings <<- c(observed_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    list(covariance = unclass(matrix), warnings = I(observed_warnings))
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("metafor")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
