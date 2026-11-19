args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("expected one JSON fixture path")
suppressPackageStartupMessages(library(rms))

fit_case <- function(case) {
    x <- as.matrix(case$x)
    y <- case$y
    n <- length(y)
    fit <- lrm(y ~ x, x = TRUE, y = TRUE)
    confidence <- if (is.null(case$confidence)) 0 else case$confidence
    limits <- if (is.null(case$emax_limits)) c(0, 1) else case$emax_limits

    RNGkind("Mersenne-Twister", "Inversion", "Rejection")
    set.seed(case$r_seed)
    indices <- t(replicate(case$B, sample(n, replace = TRUE))) - 1L
    set.seed(case$r_seed)
    # validate draws from the same sequence as sample(n, replace = TRUE).
    result <- validate(
        fit, method = "boot", B = case$B,
        emax.lim = limits, conf.int = confidence
    )
    extract <- function(column) as.list(as.numeric(result[, column]))
    list(
        id = case$id, bootstrap_indices = indices, metric_names = as.list(rownames(result)),
        original = extract("index.orig"), training = extract("training"),
        test = extract("test"), optimism = extract("optimism"), corrected = extract("index.corrected"),
        lower = if (confidence == 0) NULL else extract("Lower"),
        upper = if (confidence == 0) NULL else extract("Upper"),
        counts = extract("n")
    )
}

cases <- jsonlite::fromJSON(args[[1]], simplifyDataFrame = FALSE)
output <- list(
    versions = list(
        R = as.character(getRversion()), rms = as.character(packageVersion("rms")),
        Hmisc = as.character(packageVersion("Hmisc"))
    ),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
