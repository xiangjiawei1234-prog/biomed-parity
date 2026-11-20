args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
with_na <- function(values) unlist(lapply(values, function(value) if (is.null(value)) NA else value))

evaluate <- function(case) {
    data <- input$datasets[[case$dataset]]
    labels <- with_na(data$labels)
    multivariate <- identical(case$kind, "matrix")
    scores <- if (multivariate) do.call(rbind, lapply(data$matrix, with_na)) else with_na(data$single)
    if (multivariate) colnames(scores) <- unlist(data$columns)
    area_args <- list(partial.auc.correct = isTRUE(case$standardize),
        allow.invalid.partial.auc.correct = isTRUE(case$allow_below_chance))
    if (!is.null(case$partial_auc)) area_args$partial.auc <- unlist(case$partial_auc)
    if (!is.null(case$focus)) area_args$partial.auc.focus <- case$focus
    options <- c(list(response = labels, predictor = scores, quiet = TRUE,
        percent = isTRUE(case$percent)), area_args)
    if (!is.null(case$levels)) options$levels <- unlist(case$levels)
    if (!is.null(case$direction)) options$direction <- case$direction
    warnings <- character()
    capture <- function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    }
    tryCatch(withCallingHandlers({
        reference <- do.call(pROC::multiclass.roc, options)
        # Convert R's factor names back to the input label type for comparison.
        levels <- labels[match(as.character(reference$levels), as.character(labels))]
        class_pairs <- combn(levels, 2, simplify = FALSE)
        pairs <- lapply(seq_along(class_pairs), function(i) {
            curves <- if (multivariate) reference$rocs[[i]] else list(reference$rocs[[i]])
            components <- suppressWarnings(vapply(curves, function(curve) {
                as.numeric(do.call(pROC::auc, c(list(curve), area_args)))
            }, numeric(1)))
            list(classes = I(class_pairs[[i]]), auc = mean(components), component_aucs = I(components),
                directions = I(vapply(curves, function(curve) curve$direction, character(1))))
        })
        list(auc = as.numeric(reference$auc), levels = I(levels), pairs = pairs, warnings = I(warnings))
    }, warning = capture), error = function(e) list(error = conditionMessage(e), warnings = I(warnings)))
}

results <- lapply(input$cases, evaluate)
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), "\n", sep = "")
