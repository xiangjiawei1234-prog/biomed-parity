args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))
pack_test <- function(result) {
    values <- as.numeric(result$test[1, ])
    as.list(setNames(values, c("statistic", "df1", "df2", "pvalue", "relative_increase")))
}
pack_direct <- function(values) {
    values <- c(values$F, values$k, values$v,
        pf(values$F, values$k, values$v, lower.tail = FALSE), values$r)
    as.list(setNames(as.numeric(values), c("statistic", "df1", "df2", "pvalue", "relative_increase")))
}
test_d <- function(transformed, covariance, method, df) {
    names <- paste0("p", seq_len(ncol(transformed)))
    qhat <- lapply(seq_len(nrow(transformed)), function(i) setNames(transformed[i, ], names))
    mitml::testConstraints(
        qhat = qhat, uhat = covariance, constraints = names, method = method, df.com = df
    )
}

results <- lapply(cases, function(case) {
    if (!is.null(case$statistics)) {
        d <- sqrt(unlist(case$statistics))
        q <- cbind(d, matrix(0, length(d), case$df - 1L))
        u <- rep(list(diag(case$df)), length(d))
        return(list(
            d2 = pack_test(test_d(q, u, "D2", NULL)),
            d2_direct = pack_direct(getFromNamespace(".D2", "mitml")(d^2, case$df))
        ))
    }

    q <- as_matrix(case$estimates)
    u <- lapply(case$covariances, as_matrix)
    m <- nrow(q)
    design <- if (is.list(case$contrast[[1L]])) as_matrix(case$contrast) else {
        matrix(unlist(case$contrast), nrow = 1L)
    }
    null <- rep(unlist(case$null), length.out = nrow(design))
    transformed <- sweep(q %*% t(design), 2L, null, "-")
    covariances <- lapply(u, function(matrix) design %*% matrix %*% t(design))
    degrees <- case$complete_df
    estimates <- mitml::testEstimates(
        qhat = t(q), uhat = array(unlist(u), c(ncol(q), ncol(q), m)), df.com = degrees
    )$estimates
    names <- c("estimate", "standard_error", "statistic", "df", "pvalue", "r", "fmi")
    point <- unname(estimates[, "Estimate"])
    rows <- lapply(seq_len(nrow(estimates)), function(i) {
        as.list(setNames(as.numeric(estimates[i, ]), names))
    })
    list(
        d1 = pack_test(test_d(transformed, covariances, "D1", degrees)),
        d1_direct = pack_direct(getFromNamespace(".D1", "mitml")(
            t(transformed), array(unlist(covariances), c(nrow(design), nrow(design), m)), degrees
        )),
        parameters = list(
            estimate = point,
            within_covariance = Reduce("+", u) / m,
            between_covariance = cov(q),
            total_covariance = attr(estimates, "T"),
            marginal = rows
        )
    )
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    package_version = as.character(utils::packageVersion("mitml")),
    results = results
), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
