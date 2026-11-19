args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
suppressPackageStartupMessages(library(Hmisc))
default <- function(value, fallback) if (is.null(value)) fallback else value

fit_case <- function(case) {
    x <- as.numeric(case$x)
    xnew <- as.numeric(case$xnew)
    ordinary <- default(case$kind, "ordinary") == "ordinary"
    include_linear <- default(case$include_linear, TRUE)
    norm <- default(case$norm, 2L)
    input <- list(
        x = x, knots = case$knots, nk = default(case$n_knots, 5L), inclx = include_linear,
        type = default(case$kind, "ordinary"), norm = norm, rpm = case$missing_value,
        pc = default(case$principal_components, FALSE), fractied = default(case$tie_fraction, 0.05)
    )
    warnings <- character()
    design <- withCallingHandlers(do.call(rcspline.eval, input), warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    knots <- attr(design, "knots")
    pca <- attr(design, "pcparms")
    prediction <- rcspline.eval(
        xnew, knots = knots, inclx = include_linear,
        type = input$type, norm = norm, rpm = case$missing_value
    )
    if (!is.null(pca)) {
        prediction <- (sweep(sweep(prediction, 2L, pca$center, "-"), 2L, pca$scale, "/")) %*% pca$rotation
    }
    rms_design <- NULL
    if (ordinary && include_linear && norm == 2 && is.null(pca) && !anyNA(x)) {
        previous <- options(fractied = input$fractied)
        on.exit(options(previous))
        rms_design <- suppressWarnings(rms::rcs(x, if (is.null(case$knots)) input$nk else case$knots))
        rms_design <- matrix(as.numeric(rms_design), nrow = length(x))
    }
    integral_derivative <- if (ordinary || !is.null(pca)) NULL else {
        cbind(1, rcspline.eval(xnew, knots = knots, inclx = TRUE, norm = norm, rpm = case$missing_value))
    }
    list(
        id = case$id, knots = as.list(knots), design = unname(design),
        prediction = unname(prediction), rms_design = if (is.null(rms_design)) NULL else unname(rms_design),
        integral_derivative = integral_derivative, warnings = as.list(warnings),
        pc_center = if (is.null(pca)) NULL else as.list(unname(pca$center)),
        pc_scale = if (is.null(pca)) NULL else as.list(unname(pca$scale)),
        pc_rotation = if (is.null(pca)) NULL else unname(pca$rotation)
    )
}

cases <- jsonlite::fromJSON(args[[1]], simplifyDataFrame = FALSE)
output <- list(
    versions = list(
        R = as.character(getRversion()), Hmisc = packageDescription("Hmisc", fields = "Version"),
        rms = packageDescription("rms", fields = "Version")
    ),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE), "\n", sep = "")
