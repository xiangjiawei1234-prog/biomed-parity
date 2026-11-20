args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value

evaluate <- function(case) {
    data <- input$datasets[[case$dataset]]
    roc <- pROC::roc(unlist(data$labels), unlist(data$scores), levels = c(0, 1),
        quiet = TRUE, direction = default(case$direction, "<"), percent = isTRUE(case$percent))
    method <- default(case$method, "binormal")
    bw <- default(case$bw, "nrd0")
    adjust <- default(case$adjust, 1)
    density.controls <- unlist(case$density_controls)
    density.cases <- unlist(case$density_cases)
    warnings <- character()
    capture <- function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    }
    tryCatch(withCallingHandlers({
        smoothed <- pROC::smooth(roc, method = method, n = default(case$n, 512),
            bw = bw, kernel = default(case$kernel, "gaussian"), cut = default(case$cut, 3),
            adjust = adjust, density.controls = density.controls, density.cases = density.cases,
            reuse.auc = FALSE)
        area <- pROC::auc(smoothed, partial.auc = default(unlist(case$partial_auc), FALSE),
            partial.auc.focus = default(case$focus, "specificity"),
            partial.auc.correct = isTRUE(case$standardize),
            allow.invalid.partial.auc.correct = isTRUE(case$allow_below_chance))
        bandwidth <- NULL
        if (method == "density" && (is.null(density.controls) || is.null(density.cases))) {
            predictor <- c(roc$controls, roc$cases)
            bandwidth <- adjust * if (is.numeric(bw)) bw else {
                get(paste0("bw.", bw), envir = asNamespace("stats"))(predictor)
            }
        }
        list(specificity = I(smoothed$specificities), sensitivity = I(smoothed$sensitivities),
            auc = as.numeric(area), direction = smoothed$direction, method = method,
            percent = isTRUE(smoothed$percent), bandwidth = bandwidth,
            coefficients = if (method == "binormal") I(unname(coef(smoothed$model))) else NULL,
            warnings = I(warnings))
    }, warning = capture), error = function(e) list(error = conditionMessage(e)))
}

results <- lapply(input$cases, evaluate)
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), "\n", sep = "")
