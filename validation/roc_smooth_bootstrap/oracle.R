args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
datasets <- input$datasets
for (method in c("roc_smoothing", "roc_parametric")) {
    linked <- file.path(dirname(args[[1L]]), "..", method, "cases.json")
    datasets <- c(datasets, jsonlite::read_json(linked, simplifyVector = FALSE)$datasets)
}
default <- function(value, fallback) if (is.null(value)) fallback else value
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

evaluate <- function(case) {
    data <- datasets[[case$dataset]]
    labels <- unlist(data$labels)
    scores <- unlist(data$scores)
    percent <- isTRUE(case$percent)
    sensitivity <- identical(case$kind, "sensitivity")
    targets <- unlist(case$specificity)
    scale <- if (percent) 100 else 1
    settings <- list(method = default(case$method, "binormal"), n = default(case$n, 512L), reuse.auc = FALSE)
    if (identical(settings$method, "fitdistr")) {
        settings$density.controls <- case$control_distribution
        settings$density.cases <- case$case_distribution
    } else {
        settings <- c(settings, list(kernel = default(case$kernel, "gaussian"), bw = default(case$bw, "nrd0"),
            cut = default(case$cut, 3), adjust = default(case$adjust, 1)))
    }
    area_args <- list(partial.auc = default(unlist(case$partial_auc), FALSE),
        partial.auc.focus = default(case$focus, "specificity"),
        partial.auc.correct = isTRUE(case$standardize),
        allow.invalid.partial.auc.correct = isTRUE(case$allow_below_chance))
    draw_area <- area_args
    if (!identical(draw_area$partial.auc, FALSE)) draw_area$partial.auc <- draw_area$partial.auc / scale
    draw_area$allow.invalid.partial.auc.correct <- TRUE
    smooth <- function(index, direction, percent) {
        empirical <- pROC::roc(labels[index], scores[index], levels = c(0, 1),
            direction = direction, percent = percent, quiet = TRUE, auc = FALSE)
        do.call(pROC::smooth, c(list(empirical), settings))
    }
    measure <- function(curve, area) {
        if (sensitivity) {
            return(as.numeric(pROC::coords(curve, x = targets, input = "specificity", ret = "sensitivity")[, 1]))
        }
        as.numeric(do.call(pROC::auc, c(list(curve), area)))
    }
    observed <- smooth(seq_along(labels), default(case$direction, "<"), percent)
    point <- suppressWarnings(measure(observed, area_args))
    n_boot <- default(case$n_resamples, 43L)
    stratified <- default(case$stratified, TRUE)
    confidence <- default(case$confidence_level, 0.95)
    tail_probability <- (1 - confidence) / 2
    probabilities <- c(tail_probability, .5, 1 - tail_probability)
    controls <- which(labels == 0)
    cases <- which(labels == 1)
    set.seed(case$seed)
    draws <- replicate(n_boot, {
        if (stratified) {
            c(controls[sample.int(length(controls), replace = TRUE)],
              cases[sample.int(length(cases), replace = TRUE)])
        } else {
            sample.int(length(labels), replace = TRUE)
        }
    }, simplify = FALSE)
    null <- rep(NA_real_, max(1L, length(targets)))
    draw_values <- function(index) {
        refit <- suppressWarnings(tryCatch(smooth(index, observed$direction, percent && sensitivity), error = identity))
        if (inherits(refit, "error")) null else measure(refit, draw_area)
    }
    summarize <- function(values) {
        values <- do.call(rbind, values)
        valid <- apply(is.finite(values), 1, all)
        bounds <- t(apply(values, 2, quantile, probs = probabilities, na.rm = TRUE))
        if (!sensitivity) bounds <- bounds * scale
        list(intervals = bounds, omitted_resamples = sum(!valid))
    }
    values <- lapply(draws, draw_values)
    manual <- summarize(values)
    set.seed(case$seed)
    ci_options <- list(conf.level = confidence, boot.n = n_boot, boot.stratified = stratified)
    public <- suppressWarnings(if (sensitivity) {
        do.call(pROC::ci.se, c(list(observed, specificities = targets), ci_options))
    } else {
        do.call(pROC::ci.auc, c(list(observed, reuse.auc = FALSE), ci_options, area_args))
    })
    stopifnot(isTRUE(all.equal(as.numeric(t(manual$intervals)), as.numeric(t(public)), tolerance = 3e-13)))

    if (identical(case$append_draw, "separated")) {
        draws <- c(draws, list(c(rep(controls[[1L]], length(controls)), rep(tail(cases, 1L), length(cases)))))
    } else if (identical(case$append_draw, "one_class")) {
        draws <- c(draws, list(rep(controls[[1L]], length(labels))))
    }
    if (length(draws) != length(values)) {
        manual <- summarize(c(values, list(draw_values(draws[[length(draws)]]))))
    }
    list(observed = I(point), intervals = manual$intervals,
        n_resamples = length(draws), omitted_resamples = manual$omitted_resamples,
        bootstrap_indices = do.call(rbind, draws) - 1L)
}

results <- lapply(input$cases, function(case) {
    tryCatch(evaluate(case), error = function(e) stop(case$id, ": ", conditionMessage(e), call. = FALSE))
})
names(results) <- vapply(input$cases, function(case) case$id, character(1))
versions <- list(R = as.character(getRversion()), pROC = as.character(packageVersion("pROC")),
    MASS = as.character(packageVersion("MASS")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE), "\n", sep = "")
