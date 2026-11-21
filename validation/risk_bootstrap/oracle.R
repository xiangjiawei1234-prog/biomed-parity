args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(prodlim))

j <- 0:41
data <- data.frame(time = j + 1, status = ((j * 17) %/% 5) %% 3,
                   marker = factor(j %% 2), site = factor((j %/% 3) %% 2), z = cos(j))
forms <- list(grouped = Hist(time, status) ~ marker, alternative = Hist(time, status) ~ site)
models <- list(grouped = prodlim(Hist(time, status) ~ marker, data = data),
               alternative = prodlim(Hist(time, status) ~ site, data = data))

evaluate <- function(case) {
    fit_formula <- if (isTRUE(case$cox)) Hist(time, status) ~ z else Hist(time, status) ~ 1
    times <- unlist(case$horizons)
    training_size <- if (is.null(case$training_size)) nrow(data) else case$training_size
    invisible(capture.output(fit <- suppressMessages(riskRegression::Score(
        models, formula = fit_formula, data = data, times = times, cause = case$cause,
        split.method = "bootcv", B = case$resamples, M = training_size, seed = case$seed,
        null.model = FALSE, cens.model = if (isTRUE(case$cox)) "cox" else "km",
        summary = "ibs", keep = c("splitindex", "cv"), progress.bar = 0
    ))))
    indices <- lapply(seq_len(case$resamples), fit$split.method$index)
    predictions <- lapply(indices, function(train) {
        validation <- setdiff(seq_len(nrow(data)), train)
        lapply(forms, function(formula) {
            refit <- prodlim(formula, data = data[train, ])
            expected <- matrix(NA_real_, nrow(data), length(times))
            expected[validation, ] <- riskRegression::predictRisk(
                refit, newdata = data[validation, ], times = times, cause = case$cause)
            expected
        })
    })
    get_metric <- function(metric, name) {
        scores <- lapply(c(name, "se", "lower", "upper"), function(field) {
            do.call(rbind, lapply(names(models), function(m) metric$score[[field]][metric$score$model == m]))
        })
        setNames(scores, c("estimate", "standard_error", "lower", "upper"))
    }
    extract_replicates <- function(metric, name) {
        lapply(seq_len(case$resamples), function(i) {
            offset <- (i - 1L) * length(models) * length(times)
            rows <- offset + seq_len(length(models) * length(times))
            matrix(metric$cv.score[[name]][rows], nrow = length(models), byrow = TRUE)
        })
    }
    contrast <- function(metric, name) {
        fields <- c(paste0("delta.", name), "lower", "upper")
        setNames(lapply(fields, function(field) I(metric$contrasts[[field]])),
                 c("estimate", "lower", "upper"))
    }
    list(train = lapply(indices, function(idx) I(idx - 1L)), predictions = predictions,
         auc = get_metric(fit$AUC, "AUC"), brier = get_metric(fit$Brier, "Brier"),
         auc_replicates = extract_replicates(fit$AUC, "AUC"),
         brier_replicates = extract_replicates(fit$Brier, "Brier"),
         integrated_brier = extract_replicates(fit$Brier, "IBS"),
         auc_contrast = contrast(fit$AUC, "AUC"), brier_contrast = contrast(fit$Brier, "Brier"))
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("riskRegression")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
