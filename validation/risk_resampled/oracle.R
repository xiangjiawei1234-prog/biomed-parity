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
    times <- unlist(case$horizons)
    fit_formula <- if (isTRUE(case$cox)) Hist(time, status) ~ z else Hist(time, status) ~ 1
    fit <- suppressMessages(riskRegression::Score(
        models, formula = fit_formula, data = data, times = times, cause = case$cause,
        split.method = case$split, B = case$n, seed = case$seed, conf.int = FALSE,
        null.model = FALSE, cens.model = if (isTRUE(case$cox)) "cox" else "km",
        summary = "ibs", keep = "splitindex", progress.bar = NULL, verbose = 0
    ))
    indices <- lapply(seq_len(case$n), fit$split.method$index)
    predictions <- lapply(indices, function(index) {
        splits <- if (case$split == "loob") {
            list(list(train = index, test = setdiff(seq_len(nrow(data)), index)))
        } else {
            lapply(unique(index), function(fold) list(train = which(index != fold), test = which(index == fold)))
        }
        lapply(forms, function(formula) {
            risk <- matrix(NA_real_, nrow(data), length(times))
            for (split in splits) {
                refit <- prodlim(formula, data = data[split$train, ])
                risk[split$test, ] <- riskRegression::predictRisk(
                    refit, newdata = data[split$test, ], times = times, cause = case$cause)
            }
            # Use 17 significant digits to preserve adjacent floating-point risks.
            formatted <- matrix(sprintf("%.17g", risk), nrow(risk))
            formatted[is.na(risk)] <- NA_character_
            formatted
        })
    })
    score <- function(metric, field) do.call(rbind, lapply(names(models), function(model) {
        metric$score[[field]][metric$score$model == model]
    }))
    list(index = lapply(indices, function(x) I(x - 1L)), predictions = predictions,
         auc = score(fit$AUC, "AUC"), brier = score(fit$Brier, "Brier"),
         integrated_brier = score(fit$Brier, "IBS"))
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("riskRegression")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
