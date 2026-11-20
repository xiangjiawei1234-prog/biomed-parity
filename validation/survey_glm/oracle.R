args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript --vanilla oracle.R cases.json")
suppressPackageStartupMessages(library(survey))
options(survey.replicates.mse = FALSE, na.action = "na.omit")
cases <- jsonlite::read_json(args[[1L]])
default <- function(x, fallback) if (is.null(x)) fallback else x

fixture <- function(case) {
    i <- seq_len(48)
    cluster <- (i - 1) %/% 6
    x1 <- ((17 * i) %% 37 - 18) / 15
    x2 <- ((13 * i) %% 29 - 14) / 9
    offset <- if (isTRUE(case$offset)) ((3 * i) %% 11 - 5) / 18 else rep(0, length(i))
    y <- if (case$family == "gaussian") {
        125 + 4.3 * x1 - 2.1 * x2 + offset + 3.1 * sin(.9 * i) + 1.2 * cos(.41 * i)
    } else {
        as.integer(((29 * i) %% 47) / 47 < plogis(-.3 + .65 * x1 - .5 * x2 + offset))
    }
    w <- 1.2 + ((13 * i) %% 17) / 8
    if (isTRUE(case$zero_weights)) w[i %in% c(6, 18)] <- 0
    active <- if (isTRUE(case$subset)) (i %% 7 != 1 & i <= 43) else rep(TRUE, length(i))
    if (case$scheme == "jk1") {
        repweights <- (outer(cluster, 0:7, `!=`)) * 8 / 7
        type <- "JK1"
        scale <- .75
    } else if (case$scheme == "fay") {
        half <- cluster %% 2
        stratum <- (cluster %/% 2) + 1L
        repweights <- sapply(0:7, function(draw) {
            sign <- sapply(stratum, function(s) sum(as.integer(intToBits(bitwAnd(draw, s)))) %% 2)
            ifelse(sign == half, 1.6, .4)
        })
        type <- "Fay"
        scale <- NULL
    } else {
        repweights <- sapply(1:8, function(draw) .3 + ((i * draw + 2L * draw + 3L * i) %% 11) / 5)
        type <- "bootstrap"
        scale <- NULL
    }
    if (isTRUE(case$bad_replicate)) repweights[, 1] <- as.numeric(i <= 2)
    if (isTRUE(case[["missing"]])) {
        x1[i %in% c(5, 9, 22, 32)] <- NA
        y[i == 37] <- NA
        offset[i == 17] <- NA
    }
    if (isTRUE(case$missing_rank)) {
        repweights[, 8] <- repweights[, 7]
        repweights[3, 8] <- repweights[3, 8] + 1
        y[3] <- NA
    }
    if (isTRUE(case$combined)) repweights <- repweights * w
    if (isTRUE(case$zero_sampling_weight_only)) w[i == 2] <- 0
    list(data = data.frame(y = y, x1 = x1, x2 = x2, off = offset, w = w, active = active),
        repweights = repweights, type = type, scale = scale)
}

vector <- function(values) as.list(unname(as.numeric(values)))
results <- lapply(cases, function(case) {
    sample <- fixture(case)
    num_covariates <- default(case$n_covariates, 2L)
    terms <- if (num_covariates == 0) character(0) else paste0("x", seq_len(num_covariates))
    if (isTRUE(case$offset)) terms <- c(terms, "offset(off)")
    formula <- if (length(terms)) reformulate(terms, "y", intercept = default(case$intercept, TRUE)) else y ~ 1
    design <- svrepdesign(data = sample$data, repweights = sample$repweights, weights = sample$data$w,
        type = sample$type, scale = sample$scale, bootstrap.average = 2, rscales = unlist(case$rscales),
        rho = if (sample$type == "Fay") .4 else NULL, combined.weights = isTRUE(case$combined),
        mse = isTRUE(case$mse), degf = case$degf)
    family <- if (case$family == "gaussian") gaussian() else quasibinomial()
    fit <- suppressWarnings(svyglm(formula, design, subset = active, family = family,
        rescale = case$rescale, return.replicates = TRUE))
    df <- fit$df.residual
    inference <- if (is.null(case$df_resid)) df else as.numeric(case$df_resid)
    table <- summary(fit, df.resid = inference)
    beta <- coef(fit)
    covariance <- vcov(fit)
    se <- sqrt(diag(covariance))
    level <- default(case$confidence_level, .95)
    critical <- if (inference > 0) qt((1 + level) / 2, inference) else NaN
    list(coefficients = vector(beta), covariance = unname(covariance),
        naive_covariance = unname(fit$naive.cov), standard_errors = vector(se), t_statistics = vector(beta / se),
        pvalues = vector(if (inference > 0) table$coefficients[, 4] else rep(NaN, length(beta))),
        lower = vector(beta - critical * se), upper = vector(beta + critical * se),
        design_df = df + length(beta) - 1, residual_df = df, inference_df = inference,
        deviance = fit$deviance, dispersion = as.numeric(table$dispersion), fitted_values = vector(fitted(fit)),
        replicate_estimates = unname(fit$replicates), variance_center = vector(attr(covariance, "means")),
        included = seq_len(nrow(sample$data)) %in% as.integer(rownames(fit$model)))
})
names(results) <- vapply(cases, function(case) case$name, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    survey_version = as.character(utils::packageVersion("survey")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
