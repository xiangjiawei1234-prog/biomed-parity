args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
stopifnot(requireNamespace("MASS", quietly = TRUE))
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value

evaluate <- function(case) {
    data <- input$datasets[[case$dataset]]
    roc <- pROC::roc(unlist(data$labels), unlist(data$scores), levels = c(0, 1), quiet = TRUE,
        direction = default(case$direction, "<"), percent = isTRUE(case$percent), auc = FALSE)
    warnings <- character()
    capture <- function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    }
    describe <- function(fit) {
        list(distribution = fit$densfun, parameter_names = I(names(fit$estimate)),
            estimates = I(unname(fit$estimate)), standard_errors = I(unname(fit$sd)),
            covariance = fit$vcov, log_likelihood = fit$loglik, n_observations = fit$n)
    }
    tryCatch(withCallingHandlers({
        fit <- pROC::smooth(roc, method = "fitdistr", n = default(case$n, 512L),
            density.controls = case$control_distribution, density.cases = case$case_distribution)
        area <- pROC::auc(fit, partial.auc = default(unlist(case$partial_auc), FALSE),
            partial.auc.focus = default(case$focus, "specificity"),
            partial.auc.correct = isTRUE(case$standardize),
            allow.invalid.partial.auc.correct = isTRUE(case$allow_below_chance))
        list(specificity = I(fit$specificities), sensitivity = I(fit$sensitivities), auc = as.numeric(area),
            direction = fit$direction, percent = isTRUE(fit$percent),
            control_fit = describe(fit$fit.controls), case_fit = describe(fit$fit.cases), warnings = I(warnings))
    }, warning = capture), error = function(e) list(error = conditionMessage(e), warnings = I(warnings)))
}

results <- lapply(input$cases, evaluate)
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")),
    MASS = as.character(packageVersion("MASS")))
cat(jsonlite::toJSON(list(versions = versions, results = results), auto_unbox = TRUE, digits = NA,
    na = "null", null = "null", pretty = TRUE), "\n", sep = "")
