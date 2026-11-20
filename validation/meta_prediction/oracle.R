args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
fallback <- function(value, default) if (is.null(value)) default else value

group <- rep(0:6, each = 3)
j <- seq_along(group) - 1
x <- group / 3 - 1
z <- rep(c(-1, 0, 1), 7)
vi <- 0.024 + ((j * 13) %% 7) / 150
y <- 0.6 + 0.25 * x + 0.42 * z + 0.4 * cos(group * 2) + 0.15 * sin(j * 3)
data <- data.frame(group, x, z, vi, y)
V <- 0.23 * outer(sqrt(vi), sqrt(vi)) * outer(group, group, "==")
diag(V) <- vi

evaluate <- function(case) {
    model <- metafor::rma.mv(y, V, mods = ~ x + z, random = ~ 1 | group,
                            data = data, test = case$test,
                            dfs = fallback(case$dfs, "residual"),
                            sigma2 = fallback(case$fix_variance, NA_real_))
    options <- list(object = model, level = 100 * fallback(case$confidence_level, .95),
                    adjust = isTRUE(case$adjust), pi.type = fallback(case$prediction, "default"),
                    vcov = is.null(case$transform))
    if (!is.null(case$rows)) options$newmods <- do.call(rbind, lapply(case$rows, unlist))
    if (!is.null(case$heterogeneity)) options$hetvar <- unlist(case$heterogeneity)
    if (!is.null(case$new_sampling_variance)) options$newvi <- unlist(case$new_sampling_variance)
    if (!is.null(case$transform)) options$transf <- get(case$transform, baseenv())
    output <- do.call(predict, options)
    covariance <- if (is.null(case$transform)) unclass(output$vcov) else NULL
    values <- if (is.null(case$transform)) output$pred else output
    list(
        estimate = I(values$pred), standard_errors = I(values$se),
        confidence_intervals = cbind(values$ci.lb, values$ci.ub),
        prediction_standard_errors = I(values$pi.se),
        prediction_intervals = cbind(values$pi.lb, values$pi.ub),
        degrees_of_freedom = if (is.null(values$ddf)) NULL else I(values$ddf),
        prediction_degrees_of_freedom = if (is.null(values$pi.ddf)) NULL else I(values$pi.ddf),
        covariance = covariance
    )
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("metafor")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null", null = "null"), "\n", sep = "")
