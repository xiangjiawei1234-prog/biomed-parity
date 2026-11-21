args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value

evaluate <- function(case) {
    data <- input$datasets[[case$dataset]]
    roc <- pROC::roc(unlist(data$labels), unlist(data$scores), levels = c(0, 1),
        quiet = TRUE, direction = default(case$direction, "<"), percent = isTRUE(case$percent))
    warnings <- character()
    tryCatch(withCallingHandlers({
        smoothed <- pROC::smooth(roc, method = case$method, n = case$n, reuse.auc = FALSE)
        area <- pROC::auc(smoothed, partial.auc = default(unlist(case$partial_auc), FALSE),
            partial.auc.focus = default(case$focus, "specificity"),
            partial.auc.correct = isTRUE(case$standardize),
            allow.invalid.partial.auc.correct = isTRUE(case$allow_below_chance))
        list(specificity = I(smoothed$specificities), sensitivity = I(smoothed$sensitivities),
            auc = as.numeric(area), direction = smoothed$direction, warnings = I(warnings))
    }, warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    }), error = function(e) list(error = conditionMessage(e)))
}

results <- lapply(input$cases, evaluate)
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")),
    logcondens = as.character(packageVersion("logcondens")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), "\n", sep = "")
