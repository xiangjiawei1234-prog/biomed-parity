args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("expected a JSON fixture path")
suppressPackageStartupMessages(library(variancePartition))

fit_case <- function(case) {
    x <- as.matrix(case$x)
    n <- nrow(x)
    p <- ncol(x)
    fixed <- paste0("x", seq_len(p))
    names <- paste0("sample_", seq_len(n))
    dat <- as.data.frame(x)
    colnames(dat) <- fixed
    rownames(dat) <- names
    dat$group <- factor(case$group)
    dat$y <- case$y
    dat$w <- if (is.null(case$weights)) rep(1, n) else case$weights
    formula <- as.formula(paste("~ 0 +", paste(fixed, collapse = " + "), "+ (1|group)"))
    model_formula <- update(formula, y ~ .)
    expr <- new("EList", list(
        E = matrix(case$y, nrow = 1, dimnames = list("feature", names)),
        weights = matrix(dat$w, nrow = 1, dimnames = list("feature", names))
    ))
    if (is.null(case$contrasts)) {
        dream_fit <- dream(expr, formula, dat, ddf = "Satterthwaite", REML = TRUE, quiet = TRUE)
        tests <- fixed
    } else {
        L <- t(as.matrix(case$contrasts))
        tests <- paste0("contrast_", seq_len(ncol(L)))
        dimnames(L) <- list(fixed, tests)
        dream_fit <- dream(expr, formula, dat, L = L, ddf = "Satterthwaite", REML = TRUE, quiet = TRUE)
    }
    fit <- suppressMessages(lmerTest::lmer(model_formula, dat, weights = w, REML = TRUE))
    if (!all(tests %in% colnames(dream_fit$coefficients))) stop("dream returned missing contrasts")
    row <- function(value, columns) as.list(as.numeric(value[1, columns]))
    list(
        id = case$id,
        coefficients = row(dream_fit$coefficients, fixed),
        covariance = as.matrix(vcov(fit)), residual_variance = sigma(fit)^2,
        random_intercept_variance = as.numeric(lme4::VarCorr(fit)$group),
        estimate = row(dream_fit$coefficients, tests),
        standard_error = row(dream_fit$stdev.unscaled * dream_fit$sigma, tests),
        df = row(dream_fit$df.residual, tests), statistic = row(dream_fit$t, tests),
        pvalue = row(dream_fit$p.value, tests)
    )
}

cases <- jsonlite::fromJSON(args[1], simplifyDataFrame = FALSE)
output <- list(
    versions = list(R = as.character(getRversion()),
                    variancePartition = as.character(packageVersion("variancePartition")),
                    lmerTest = as.character(packageVersion("lmerTest"))),
    cases = lapply(cases, fit_case)
)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, digits = NA, pretty = TRUE), "\n", sep = "")
