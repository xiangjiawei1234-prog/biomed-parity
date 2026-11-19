#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Expected a fixture directory")
library(limma)
options(digits = 17)
folder <- args[[1]]
counts <- as.matrix(read.csv(file.path(folder, "counts.csv"), header = FALSE))
design <- cbind(1, rep(0:1, each = 4))
fit <- voomWithQualityWeights(counts, design)
write_values <- function(path, value) {
  write.table(value, file.path(folder, path), sep = ",", row.names = FALSE, col.names = FALSE)
}
write_values("E.csv", fit$E)
write_values("weights.csv", fit$weights)
write_values("sample_weights.csv", fit$targets$sample.weights)
