suppressPackageStartupMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))

results <- lapply(cases, function(case) {
    counts <- as_matrix(case$counts)
    colnames(counts) <- paste0("sample", seq_len(ncol(counts)))
    data <- data.frame(row.names = colnames(counts))
    full <- as_matrix(case$full)
    reduced <- as_matrix(case$reduced)
    colnames(full) <- paste0("x", seq_len(ncol(full)))
    colnames(reduced) <- paste0("r", seq_len(ncol(reduced)))
    dds <- suppressMessages(DESeqDataSetFromMatrix(counts, data, design = ~1))
    if (is.list(case$size_factors[[1L]])) {
        normalizationFactors(dds) <- as_matrix(case$size_factors)
    } else {
        sizeFactors(dds) <- unlist(case$size_factors)
    }
    dispersions(dds) <- unlist(case$dispersions)
    out <- nbinomLRT(dds, full = full, reduced = reduced, quiet = TRUE)
    metadata <- mcols(out)
    if (!all(metadata$fullBetaConv[!metadata$allZero]) ||
        !all(metadata$reducedBetaConv[!metadata$allZero])) {
        stop("R model failed to converge")
    }
    statistic <- metadata$LRTStatistic
    pvalue <- metadata$LRTPvalue
    log_likelihood <- -metadata$deviance / 2
    list(
        statistic = statistic, pvalue = pvalue,
        adjusted_pvalue = p.adjust(pvalue, method = "BH"),
        full_log_likelihood = log_likelihood,
        reduced_log_likelihood = log_likelihood - statistic / 2,
        tested = !metadata$allZero,
        degrees_of_freedom = ncol(full) - ncol(reduced)
    )
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("DESeq2")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
