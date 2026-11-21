args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

evaluate <- function(case) {
    data <- input$datasets[[case$dataset]]
    labels <- unlist(data$labels)
    scores_2 <- if (isTRUE(case$identical_scores)) data$scores_1 else data$scores_2
    scores <- lapply(list(data$scores_1, scores_2), unlist)
    settings <- list(method = default(case$method, "binormal"), n = case$n)
    if (identical(settings$method, "fitdistr")) {
        settings$density.controls <- case$control_distribution
        settings$density.cases <- case$case_distribution
    } else if (identical(settings$method, "density")) {
        settings <- c(settings, list(kernel = default(case$kernel, "gaussian"), bw = default(case$bw, "nrd0"),
            cut = default(case$cut, 3), adjust = default(case$adjust, 1)))
    }
    area_args <- list(partial.auc = default(unlist(case$partial_auc), FALSE),
        partial.auc.focus = default(case$focus, "specificity"), partial.auc.correct = isTRUE(case$standardize))
    measure <- function(curve, allow_below_chance = FALSE) {
        as.numeric(do.call(pROC::auc, c(list(curve), area_args,
            allow.invalid.partial.auc.correct = allow_below_chance)))
    }
    smooth <- function(index, predictor, direction, store = FALSE) {
        roc <- pROC::roc(labels[index], predictor[index], levels = c(0, 1), direction = direction,
            percent = isTRUE(case$percent), quiet = TRUE, auc = TRUE)
        if (store) roc$auc <- do.call(pROC::auc, c(list(roc), area_args))
        do.call(pROC::smooth, c(list(roc), settings))
    }
    reuse <- isTRUE(case$reuse_auc)
    curves <- lapply(scores, function(predictor) {
        smooth(seq_along(labels), predictor, default(case$direction, "<"), store = reuse)
    })
    direction <- vapply(curves, function(curve) curve$direction, character(1))
    estimates <- vapply(curves, measure, numeric(1L))
    observed_difference <- estimates[1L] - estimates[2L]
    controls <- which(labels == 0)
    cases <- which(labels == 1)
    stratified <- default(case$stratified, TRUE)
    alternative <- default(case$alternative, "two.sided")
    count <- 47L
    set.seed(case$seed)
    draws <- replicate(count, {
        if (stratified) {
            c(controls[sample.int(length(controls), replace = TRUE)],
                cases[sample.int(length(cases), replace = TRUE)])
        } else {
            sample.int(length(labels), replace = TRUE)
        }
    }, simplify = FALSE)
    draw_areas <- function(index) {
        vapply(seq_along(scores), function(i) {
            suppressWarnings(tryCatch(measure(smooth(index, scores[[i]], direction[i]), TRUE),
                error = function(error) NA_real_))
        }, numeric(1L))
    }
    summarize <- function(differences) {
        valid <- differences[is.finite(differences)]
        standard_error <- sd(valid)
        statistic <- observed_difference / standard_error
        if (is.nan(statistic) && observed_difference == 0 && all(valid == 0)) statistic <- 0
        p_value <- switch(alternative, two.sided = 2 * pnorm(-abs(statistic)),
            greater = pnorm(-statistic), less = pnorm(statistic))
        list(standard_error = standard_error, statistic = statistic,
            p_value = p_value, omitted_resamples = length(differences) - length(valid))
    }
    areas <- vapply(draws, draw_areas, numeric(2L))
    differences <- areas[1L, ] - areas[2L, ]
    result <- summarize(differences)
    set.seed(case$seed)
    # Give the reuse case a deliberately different area request to check which specification is used.
    test_area <- if (reuse) list(partial.auc = FALSE) else area_args
    warnings <- character()
    public <- withCallingHandlers(do.call(pROC::roc.test, c(curves, list(method = "bootstrap",
        paired = TRUE, reuse.auc = reuse, boot.n = count, boot.stratified = stratified,
        alternative = alternative), test_area)), warning = function(w) {
            warnings <<- c(warnings, conditionMessage(w))
            invokeRestart("muffleWarning")
        })
    assert_close <- function(actual, expected) stopifnot(isTRUE(all.equal(
        as.numeric(actual), as.numeric(expected), tolerance = 4e-13)))
    assert_close(estimates, public$estimate)
    assert_close(result$statistic, public$statistic)
    assert_close(result$p_value, public$p.value)
    omissions <- grep("produced during bootstrap were ignored", warnings, value = TRUE)
    if (result$omitted_resamples) {
        stopifnot(identical(omissions, sprintf("%i NA value(s) produced during bootstrap were ignored.",
            result$omitted_resamples)))
    } else {
        stopifnot(length(omissions) == 0L)
    }
    if (isTRUE(case$append_failed_pair)) {
        failed <- c(rep(controls[[1L]], length(controls)), rep(tail(cases, 1L), length(cases)))
        failed_areas <- draw_areas(failed)
        stopifnot(is.finite(failed_areas[1L]), is.na(failed_areas[2L]))
        draws <- c(draws, list(failed))
        result <- summarize(c(differences, NA_real_))
    }
    c(list(auc_1 = estimates[1L], auc_2 = estimates[2L], difference = observed_difference), result,
        list(n_resamples = length(draws), bootstrap_indices = do.call(rbind, draws) - 1L))
}

results <- lapply(input$cases, function(case) {
    tryCatch(evaluate(case), error = function(error) stop(case$id, ": ", conditionMessage(error), call. = FALSE))
})
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")),
    MASS = as.character(packageVersion("MASS")), logcondens = as.character(packageVersion("logcondens")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), "\n", sep = "")
