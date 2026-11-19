#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Expected a fixture directory")
library(limma)
message("R ", getRversion(), ", limma ", packageVersion("limma"))
options(digits = 17)
folder <- args[[1]]
read_matrix <- function(name) as.matrix(read.csv(file.path(folder, name), header = FALSE))
expression <- read_matrix("expression.csv")
design <- read_matrix("design.csv")
block <- as.vector(read_matrix("block.csv"))
contrast <- matrix(c(-1, 1), ncol = 1)
dups <- duplicateCorrelation(expression, design, block = block)
write.csv(data.frame(atanh = dups$atanh.correlations, consensus = dups$consensus.correlation),
          file.path(folder, "duplicates.csv"), row.names = FALSE)
for (mode in c("ordinary", "block")) {
  fit <- if (mode == "ordinary") lmFit(expression, design) else lmFit(
    expression, design, block = block, correlation = dups$consensus.correlation
  )
  contrasted <- eBayes(contrasts.fit(fit, contrast))
  out <- data.frame(
    coefficient = contrasted$coefficients[, 1],
    stdev_unscaled = contrasted$stdev.unscaled[, 1],
    sigma = contrasted$sigma,
    df_residual = contrasted$df.residual,
    s2_post = contrasted$s2.post,
    df_total = contrasted$df.total,
    t = contrasted$t[, 1],
    p_value = contrasted$p.value[, 1],
    adjusted_p = p.adjust(contrasted$p.value[, 1], "BH")
  )
  write.csv(out, file.path(folder, paste0(mode, ".csv")), row.names = FALSE)
}
