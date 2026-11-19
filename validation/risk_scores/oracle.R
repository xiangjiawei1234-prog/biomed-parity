suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(riskRegression))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
default <- function(value, fallback) if (is.null(value)) fallback else value

summary_result <- function(table, target, influence) {
    list(
        estimate = unname(table[[target]]),
        standard_error = unname(table$se),
        lower = unname(table$lower),
        upper = unname(table$upper),
        influence = influence
    )
}

results <- lapply(cases, function(case) {
    data <- data.frame(time = unlist(case$time), status = unlist(case$status))
    order_for_reference <- order(data$time, data$status == 0, method = "radix")
    original_order <- order(order_for_reference)
    data <- data[order_for_reference, ]
    horizons <- unlist(case$horizons)
    risks <- lapply(case$risks, function(model) do.call(rbind, lapply(model, unlist)))
    risks <- lapply(risks, function(model) model[order_for_reference, , drop = FALSE])
    names(risks) <- paste0("m", seq_along(risks) - 1L)
    output <- Score(
        risks, Hist(time, status) ~ 1, data = data, times = horizons,
        cause = default(case$cause, 1L), null.model = FALSE, cens.model = "km",
        keep = "iid", conf.int = default(case$confidence_level, .95),
        conservative = default(case$conservative, FALSE)
    )
    model_result <- function(metric, name, model) {
        mask <- metric$score$model == model
        rows <- metric$score[mask, ]
        iid <- metric$iid.decomp
        influence <- lapply(horizons, function(horizon) {
            keep <- iid$model == model & iid$times == horizon
            iid[[paste0("IF.", name)]][keep][order(iid$riskRegression_ID[keep])][original_order]
        })
        summary_result(rows, name, influence)
    }
    auc <- lapply(names(risks), function(model) model_result(output$AUC, "AUC", model))
    brier <- lapply(names(risks), function(model) model_result(output$Brier, "Brier", model))
    pairs <- combn(seq_along(risks), 2L, simplify = FALSE)
    contrasts <- lapply(pairs, function(pair) {
        reference <- names(risks)[pair[1]]
        model <- names(risks)[pair[2]]
        a <- output$AUC$contrasts
        b <- output$Brier$contrasts
        keep_a <- a$model == model & a$reference == reference
        keep_b <- b$model == model & b$reference == reference
        a <- a[keep_a, ]
        b <- b[keep_b, ]
        list(
            model = pair[2] - 1L,
            reference = pair[1] - 1L,
            auc = summary_result(a, "delta.AUC", Map("-", auc[[pair[2]]]$influence,
                auc[[pair[1]]]$influence)),
            brier = summary_result(b, "delta.Brier", Map("-", brier[[pair[2]]]$influence,
                brier[[pair[1]]]$influence)),
            auc_pvalues = a$p,
            brier_pvalues = b$p
        )
    })
    list(horizons = horizons, auc = auc, brier = brier, contrasts = contrasts)
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("riskRegression")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
