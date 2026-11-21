args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

evaluate <- function(case) {
    data <- input$datasets[[case$dataset]]
    labels <- lapply(list(data$labels_1, data$labels_2), unlist)
    scores <- lapply(list(data$scores_1, data$scores_2), unlist)
    if (isTRUE(case$same_data)) {
        labels[[2L]] <- labels[[1L]]
        scores[[2L]] <- scores[[1L]]
    }
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
    measure <- function(curve, allow_below_chance) {
        as.numeric(do.call(pROC::auc, c(list(curve), area_args,
            list(allow.invalid.partial.auc.correct = allow_below_chance))))
    }
    smooth <- function(i, index, direction) {
        roc <- pROC::roc(labels[[i]][index], scores[[i]][index], levels = c(0, 1),
            direction = direction, percent = isTRUE(case$percent), quiet = TRUE, auc = FALSE)
        do.call(pROC::smooth, c(list(roc), settings))
    }
    curves <- lapply(seq_along(labels), function(i) {
        smooth(i, seq_along(labels[[i]]), default(case$direction, "<"))
    })
    directions <- vapply(curves, function(curve) curve$direction, character(1L))
    estimates <- vapply(curves, measure, numeric(1L), allow_below_chance = isTRUE(case$allow_below_chance))
    observed_difference <- estimates[1L] - estimates[2L]
    stratified <- default(case$stratified, TRUE)
    draw_sample <- function(label) {
        if (stratified) {
            controls <- which(label == 0)
            cases <- which(label == 1)
            c(controls[sample.int(length(controls), replace = TRUE)],
                cases[sample.int(length(cases), replace = TRUE)])
        } else {
            sample.int(length(label), replace = TRUE)
        }
    }
    count <- 43L
    set.seed(case$seed)
    draws <- replicate(count, lapply(labels, draw_sample), simplify = FALSE)
    draw_areas <- function(indices) {
        vapply(seq_along(labels), function(i) {
            suppressWarnings(tryCatch(measure(smooth(i, indices[[i]], directions[i]), TRUE),
                error = function(error) NA_real_))
        }, numeric(1L))
    }
    areas <- vapply(draws, draw_areas, numeric(2L))
    differences <- areas[1L, ] - areas[2L, ]
    usable <- differences[is.finite(differences)]
    standard_error <- sd(usable)
    statistic <- observed_difference / standard_error
    if (is.nan(statistic) && observed_difference == 0 && all(usable == 0)) statistic <- 0
    alternative <- default(case$alternative, "two.sided")
    p_value <- switch(alternative, two.sided = 2 * pnorm(-abs(statistic)),
        greater = pnorm(-statistic), less = pnorm(statistic))
    omitted <- count - length(usable)

    set.seed(case$seed)
    warnings <- character()
    public <- withCallingHandlers(do.call(pROC::roc.test, c(curves, list(method = "bootstrap",
        paired = FALSE, reuse.auc = FALSE, boot.n = count, boot.stratified = stratified,
        alternative = alternative, allow.invalid.partial.auc.correct = isTRUE(case$allow_below_chance)),
        area_args)), warning = function(w) {
            warnings <<- c(warnings, conditionMessage(w))
            invokeRestart("muffleWarning")
        })
    assert_close <- function(actual, expected) stopifnot(isTRUE(all.equal(
        as.numeric(actual), as.numeric(expected), tolerance = 4e-13)))
    assert_close(estimates, public$estimate)
    assert_close(statistic, public$statistic)
    assert_close(p_value, public$p.value)
    omissions <- grep("produced during bootstrap were ignored", warnings, value = TRUE)
    if (omitted) {
        stopifnot(identical(omissions, sprintf("%i NA value(s) produced during bootstrap were ignored.", omitted)))
    } else {
        stopifnot(length(omissions) == 0L)
    }
    if (isTRUE(case$same_data)) stopifnot(standard_error > 0, observed_difference == 0)
    if (identical(case$id, "fitdistr_unstratified_omissions")) {
        stopifnot(any(is.na(areas[1L, ]) & !is.na(areas[2L, ])), any(!is.na(areas[1L, ]) & is.na(areas[2L, ])))
    }
    draw_matrix <- function(i) do.call(rbind, lapply(draws, `[[`, i)) - 1L
    list(auc_1 = estimates[1L], auc_2 = estimates[2L], difference = observed_difference,
        standard_error = standard_error, statistic = statistic, p_value = p_value,
        omitted_resamples = omitted, n_resamples = count,
        bootstrap_indices_1 = draw_matrix(1L), bootstrap_indices_2 = draw_matrix(2L))
}

results <- lapply(input$cases, function(case) {
    tryCatch(evaluate(case), error = function(error) stop(case$id, ": ", conditionMessage(error), call. = FALSE))
})
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")),
    MASS = as.character(packageVersion("MASS")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), "\n", sep = "")
