# Install the reference packages used by the core parity suite.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 1L) stop("Usage: Rscript --vanilla validation/install_r.R [library-directory]")
library_dir <- if (length(args)) args[[1L]] else ".r-lib"
dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)
library_dir <- normalizePath(library_dir, mustWork = TRUE)
.libPaths(c(library_dir, .libPaths()))

available <- function(package) {
    suppressMessages(suppressWarnings(requireNamespace(package, quietly = TRUE)))
}

bioc <- c("DESeq2", "variancePartition")
packages <- c("jsonlite", "survival", "pROC", "logcondens", "survRM2", "gsDesign",
    "mice", "mitml", "lme4", "MASS", "nnet", "survey", "metafor", "clubSandwich", "rms", "Hmisc",
    "multcomp", "mvtnorm",
    "riskRegression", "prodlim", "mstate", "glmmTMB", "icenReg")
bioc_missing <- bioc[!vapply(bioc, available, logical(1))]
if (length(bioc_missing)) packages <- c(packages, "BiocManager")
missing <- packages[!vapply(packages, available, logical(1))]
if (length(missing)) {
    utils::install.packages(missing, lib = library_dir, repos = "https://cloud.r-project.org")
}
if (length(bioc_missing)) {
    BiocManager::install(bioc_missing, lib = library_dir, ask = FALSE,
        update = FALSE, dependencies = NA)
}

for (package in c(packages, bioc)) {
    if (!available(package)) stop("Unavailable package: ", package)
    cat(package, utils::packageDescription(package, fields = "Version"), "\n")
}
