args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(prodlim))

j <- 0:41
data <- data.frame(time = j + 1, status = ((j * 17) %/% 5) %% 3,
                   marker = factor(j %% 2), site = factor((j %/% 3) %% 2), z = cos(j))
models <- list(grouped = prodlim(Hist(time, status) ~ marker, data = data),
               alternative = prodlim(Hist(time, status) ~ site, data = data),
               marginal = prodlim(Hist(time, status) ~ 1, data = data))
reference_models <- c("grouped", "alternative")

evaluate <- function(case) {
    use_cox <- identical(case$censoring, "cox")
    formula <- if (use_cox) Hist(time, status) ~ z else Hist(time, status) ~ 1
    invisible(capture.output(output <- suppressMessages(riskRegression::Score(
        models, formula, data = data, times = unlist(case$horizons),
        cause = case$cause, split.method = paste0("cv", case$n_folds), B = 1,
        seed = case$seed, null.model = FALSE, cens.model = if (use_cox) "cox" else "km",
        summary = c("risks", "ibs"), keep = "splitindex", progress.bar = 0
    ))))
    summary <- function(metric, name) {
        values <- metric$score
        fields <- c(name, "se", "lower", "upper")
        output <- lapply(fields, function(field) do.call(rbind, lapply(reference_models, function(model) {
            values[[field]][values$model == model]
        })))
        setNames(output, c("estimate", "standard_error", "lower", "upper"))
    }
    extract_risks <- function(name) {
        observed <- output$risks$score[output$risks$score$model == name, ]
        sapply(unlist(case$horizons), function(h) {
            at_horizon <- observed[observed$times == h, ]
            at_horizon$risk[match(seq_len(nrow(data)), at_horizon$riskRegression_ID)]
        })
    }
    predictions <- lapply(reference_models, extract_risks)
    folds <- as.integer(output$split.method$index(1))
    marginal_expected <- matrix(NA_real_, nrow(data), length(case$horizons))
    for (fold in unique(folds)) {
        refitted <- prodlim(Hist(time, status) ~ 1, data = data[folds != fold, ])
        risk <- riskRegression::predictRisk(refitted, newdata = data[folds == fold, ],
                                            times = unlist(case$horizons), cause = case$cause)
        marginal_expected[folds == fold, ] <- rep(risk, each = sum(folds == fold))
    }
    malformed <- sum(abs(extract_risks("marginal") - marginal_expected) > 1e-12)
    list(folds = folds - 1L,
         predictions = predictions, auc = summary(output$AUC, "AUC"),
         brier = summary(output$Brier, "Brier"),
         integrated_brier = do.call(rbind, lapply(reference_models, function(name)
             output$Brier$score$IBS[output$Brier$score$model == name])),
         marginal_predictions = marginal_expected, malformed_marginal_cells = malformed)
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("riskRegression")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
