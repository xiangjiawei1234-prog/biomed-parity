args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
unpack <- function(row) vapply(row, function(value) if (is.null(value)) NA_real_ else value, numeric(1))

evaluate <- function(case) {
    x <- cbind(x = unpack(case$x), z = unpack(case$z))
    y <- unpack(case$y)
    umx <- cbind(aux = unpack(case$aux))
    ry <- !is.na(y)
    offset <- case$delta + case$x_weight * x[, "x"] + case$aux_weight * umx[, "aux"]
    ums <- paste0(case$delta, sprintf("%+g*x%+g*aux", case$x_weight, case$aux_weight))
    conditional <- getExportedValue("mice", paste0("mice.impute.", case$method))
    set.seed(71582)
    draws <- replicate(case$n_draws, as.numeric(conditional(y, ry, x, ums = ums, umx = umx)))
    set.seed(71582)
    base <- replicate(case$n_draws, as.numeric(conditional(y, ry, x, ums = "0")))
    paired_error <- if (case$method == "mnar.norm") {
        max(abs(draws - base - offset[!ry]))
    } else {
        sum((draws - base) * offset[!ry] < 0)
    }
    list(means = rowMeans(draws), standard_deviations = apply(draws, 1L, sd),
         baseline_means = rowMeans(base), paired_error = paired_error)
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("mice")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE), "\n", sep = "")
