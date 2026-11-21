args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

evaluate <- function(case) {
    levels <- c("followup", "baseline", "discharge")
    studies <- rep(seq_len(20L), each = 3L)
    outcome <- rep(c(2, 1, 3), times = 20L)
    if (isTRUE(case$incomplete)) {
        keep <- (3 * studies + outcome) %% 11 != 0
        studies <- c(studies[keep], 3, 5)
        outcome <- c(outcome[keep], 2, 1)
    }
    index <- seq_along(studies)
    moderator <- ((7 * index) %% 19) / 17
    variance <- 0.052 + ((11 * index) %% 13) / 300
    random <- cbind(
        .75 * sin(studies * 1.24) + .37 * cos(studies * 2.59),
        -.51 * sin(studies * 1.24) + .66 * cos(studies * .92),
        .45 * sin(studies * 1.24) - .32 * cos(studies * .92) + .65 * sin(studies * 2.17)
    )
    if (!(case$structure %in% c("AR", "HAR"))) random[, 2] <- -random[, 2]
    y <- c(.26, .55, -.12)[outcome] + .36 * moderator +
        random[cbind(index, outcome)] + .13 * cos(index * 2.8)
    V <- .18 * outer(sqrt(variance), sqrt(variance)) * outer(studies, studies, "==")
    diag(V) <- variance
    study <- factor(studies)
    endpoint <- factor(levels[outcome], levels = levels)
    X <- cbind(1.0 * (outcome == 1), 1.0 * (outcome == 2),
               1.0 * (outcome == 3), moderator)
    dfs <- if (is.null(case$dfs)) "residual" else case$dfs
    fit <- metafor::rma.mv(
        y, V, mods = X, intercept = FALSE, random = ~ endpoint | study,
        struct = case$structure, method = case$method, test = case$test, dfs = dfs,
        btt = 4, level = 90,
        control = list(optimizer = "BFGS", reltol = 1e-13, maxit = 2000L)
    )
    list(
        levels = levels, random_covariance = unname(as.matrix(fit$G)),
        coefficients = I(unname(coef(fit))), coefficient_covariance = unname(vcov(fit)),
        se = I(fit$se), p = I(fit$pval), confint = cbind(fit$ci.lb, fit$ci.ub), df = I(unname(fit$ddf)),
        likelihood = as.numeric(logLik(fit)), omnibus = fit$QM,
        omnibus_p = fit$QMp, qe = fit$QE
    )
}
results <- lapply(cases, function(case) tryCatch(
    evaluate(case), error = function(err) stop(case$id, ": ", conditionMessage(err))
))
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("metafor")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
