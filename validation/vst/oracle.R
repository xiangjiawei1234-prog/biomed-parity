suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
spec <- jsonlite::read_json(args[[1]], simplifyVector = FALSE)
as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))
as_matrix_rows <- function(mat) lapply(seq_len(nrow(mat)), function(i) keep(mat[i, ]))
keep <- function(x) I(unname(as.numeric(x)))

results <- lapply(spec$cases, function(case) {
    count_data <- as_matrix(spec$counts[[case$counts]])
    colnames(count_data) <- paste0("S", seq_len(ncol(count_data)))
    rownames(count_data) <- paste0("F", seq_len(nrow(count_data)))
    covariates <- data.frame(row.names = colnames(count_data))
    dds <- DESeqDataSetFromMatrix(count_data, covariates, design = ~1)
    if (!is.null(case$size_factors)) sizeFactors(dds) <- unlist(case$size_factors)
    if (!is.null(case$normalization_factors)) {
        normalizationFactors(dds) <- as_matrix(case$normalization_factors)
    }
    curve <- case$curve
    if (curve$type == "fitted") {
        dds <- estimateDispersions(estimateSizeFactors(dds), fitType = "local", quiet = TRUE)
        stopifnot(attr(dispersionFunction(dds), "fitType") == "local")
    } else {
        fit <- if (curve$type == "analytic") {
            function(q) {
                curve$base + curve$slope / (q + curve$shift) +
                    curve$hump * exp(-((log1p(q) - 2.5) / 1.2)^2)
            }
        } else {
            approxfun(unlist(curve$mean), unlist(curve$dispersion), method = "linear")
        }
        attr(fit, "fitType") <- "local"
        suppressMessages({ dispersionFunction(dds) <- fit })
    }
    normalized <- counts(dds, normalized = TRUE)
    grid <- sinh(seq(0, asinh(max(normalized)), length.out = 1000))[-1]
    output <- list(transform = as_matrix_rows(getVarianceStabilizedData(dds)))
    if (curve$type == "fitted") {
        output$size_factors <- keep(sizeFactors(dds))
        output$means <- keep(grid)
        output$dispersions <- keep(dispersionFunction(dds)(grid))
    }
    output
})
names(results) <- vapply(spec$cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("DESeq2")),
    locfit_version = as.character(packageVersion("locfit")),
    results = results
), auto_unbox = TRUE, digits = NA, na = "null"), "\n", sep = "")
