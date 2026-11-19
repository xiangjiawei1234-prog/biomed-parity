args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript oracle.R {chain|distribution} cases.json")
experiment <- match.arg(args[[1L]], c("chain", "distribution"))
cases <- jsonlite::read_json(args[[2L]], simplifyVector = FALSE)

default <- function(value, fallback) if (is.null(value)) fallback else value
as_matrix <- function(rows) do.call(rbind, lapply(rows, function(row) {
    vapply(row, function(value) default(value, NA), numeric(1))
}))

as_columns <- function(rows) {
    result <- as.data.frame(as_matrix(rows))
    names(result) <- paste0("x", seq_len(ncol(result)))
    result
}

run_chain <- function(case) {
    data <- as_columns(case$data)
    methods <- vapply(case$methods, default, character(1), fallback = "")
    methods[methods == "passive"] <- "~I(2 * x2 + x3^2)"
    design <- as_matrix(case$predictor_matrix)
    rownames(design) <- colnames(design) <- names(data)
    arguments <- list(
        data = data, method = methods, predictorMatrix = design,
        m = case$m, maxit = case$iterations, printFlag = FALSE, seed = 93212,
        donors = default(case$donors, 5), matchtype = default(case$match_type, 1),
        remove.collinear = FALSE, remove.constant = FALSE
    )
    if (!is.null(case$visit_sequence)) arguments$visitSequence <- unlist(case$visit_sequence) + 1L
    if (!is.null(case$initial)) arguments$data.init <- as_columns(case$initial)
    if (!is.null(case$where)) arguments$where <- as_matrix(case$where) == 1
    if (!is.null(case$ignore)) arguments$ignore <- unlist(case$ignore)
    fit <- do.call(mice::mice, arguments)
    by_chain <- function(array) lapply(seq_len(fit$m), function(chain) {
        t(array[, , chain])
    })
    list(
        completed = lapply(seq_len(fit$m), function(chain) as.matrix(mice::complete(fit, chain))),
        chain_mean = by_chain(fit$chainMean),
        chain_variance = by_chain(fit$chainVar)
    )
}

run_distribution <- function(case) {
    # Use the published R kernels on the same complete predictor matrix.
    set.seed(case$seed)
    data <- as_matrix(case$data)
    y <- data[, 3]
    x <- data[, 1:2, drop = FALSE]
    observed <- !is.na(y)
    method <- getExportedValue("mice", paste0("mice.impute.", case$methods[[3L]]))
    draws <- replicate(case$m, as.numeric(method(y, observed, x, donors = case$donors,
        matchtype = case$match_type)))
    centered <- draws - rowMeans(draws)
    covariance_se <- outer(seq_len(nrow(draws)), seq_len(nrow(draws)), Vectorize(
        function(j, k) sd(centered[j, ] * centered[k, ]) / sqrt(ncol(draws))
    ))
    list(
        means = rowMeans(draws),
        standard_deviations = apply(draws, 1L, sd),
        covariance = cov(t(draws)),
        covariance_se = covariance_se
    )
}

runner <- if (experiment == "chain") run_chain else run_distribution
results <- lapply(cases, runner)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    package_version = as.character(utils::packageVersion("mice")),
    results = results
), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
