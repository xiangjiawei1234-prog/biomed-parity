#!/usr/bin/env Rscript

workspace <- commandArgs(trailingOnly = TRUE)[[1]]
input <- function(name) read.delim(file.path(workspace, name), row.names = 1)
output <- function(values, name) {
    write.table(values, file.path(workspace, name), sep = "\t", quote = FALSE, col.names = NA)
}

suppressPackageStartupMessages(library(DESeq2))
counts <- t(as.matrix(input("counts.tsv")))
metadata <- input("metadata.tsv")
metadata$condition <- factor(metadata$condition, levels = c("reference", "tumor"))
metadata$batch <- factor(metadata$batch, levels = c("A", "B"))
writeLines(
    c(as.character(getRversion()), as.character(packageVersion("DESeq2"))),
    file.path(workspace, "r_versions.txt")
)

for (method in c("parametric", "mean")) {
    data <- DESeqDataSetFromMatrix(counts, metadata, design = ~batch + condition)
    fitted <- DESeq(
        data, test = "Wald", fitType = method, sfType = "ratio",
        minReplicatesForReplace = Inf, betaPrior = FALSE, quiet = TRUE
    )
    result <- as.data.frame(results(
        fitted, contrast = c("condition", "tumor", "reference"),
        cooksCutoff = FALSE, independentFiltering = FALSE
    ))
    result$dispersion <- dispersions(fitted)
    output(result, paste0(method, "_wald.tsv"))
    output(data.frame(size_factor = sizeFactors(fitted)), paste0(method, "_sf.tsv"))
    actual <- attr(dispersionFunction(fitted), "fitType")
    writeLines(actual, file.path(workspace, paste0(method, "_type.txt")))
}

data <- DESeqDataSetFromMatrix(counts, metadata, design = ~batch + condition)
for (method in c("parametric", "local")) {
    transformed <- varianceStabilizingTransformation(data, blind = TRUE, fitType = method)
    output(t(assay(transformed)), paste0(method, "_vst.tsv"))
}
lrt <- DESeq(
    data, test = "LRT", reduced = ~batch, fitType = "mean", sfType = "ratio",
    minReplicatesForReplace = Inf, quiet = TRUE
)
res <- results(lrt, cooksCutoff = FALSE, independentFiltering = FALSE)
writeLines(as.character(sum(is.finite(res$pvalue))), file.path(workspace, "lrt_count.txt"))
