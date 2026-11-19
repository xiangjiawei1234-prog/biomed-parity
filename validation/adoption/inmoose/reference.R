#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected a data directory and scenario name")
library(edgeR)
message("R ", getRversion(), ", edgeR ", packageVersion("edgeR"), ", limma ", packageVersion("limma"))
options(digits = 17)
folder <- args[[1]]
scenario <- args[[2]]
counts <- as.matrix(read.csv(file.path(folder, paste0(scenario, ".csv")), header = FALSE))
design <- cbind(1, rep(0:1, each = 4))
for (mode in c("legacy", "v4", "v4_auto")) {
  legacy <- mode == "legacy"
  dispersion <- if (mode == "v4_auto") NULL else 0.14
  fit <- glmQLFit(counts, design = design, dispersion = dispersion,
                  lib.size = rep(1000, 8), legacy = legacy)
  tested <- glmQLFTest(fit, coef = 2L)
  effective_df <- if (legacy) fit$df.residual.zeros else fit$df.residual.adj
  out <- data.frame(
    logFC = tested$table[["logFC"]], logCPM = tested$table[["logCPM"]],
    F = tested$table[["F"]], PValue = tested$table[["PValue"]],
    df_effective = effective_df, var_post = fit$s2.post,
    dispersion = rep_len(fit$dispersion, nrow(counts))
  )
  write.csv(out, file.path(folder, paste(scenario, mode, "csv", sep = ".")), row.names = FALSE)
}
