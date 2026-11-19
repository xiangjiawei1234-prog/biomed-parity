args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

mean_result <- function(row) {
    as.list(setNames(as.numeric(row), c("estimate", "standard_error", "lower", "upper")))
}

arm_result <- function(result) {
    list(rmst = mean_result(result$rmst), rmtl = mean_result(result$rmtl))
}

contrast_result <- function(row) {
    as.list(setNames(as.numeric(row), c("estimate", "lower", "upper", "pvalue")))
}

model_result <- function(model) {
    lapply(seq_len(nrow(model)), function(i) {
        row <- model[i, ]
        if (ncol(model) == 6L) row <- c(row[1:4], row[1], row[5:6])
        as.list(setNames(as.numeric(row),
            c("coefficient", "standard_error", "statistic", "pvalue", "estimate", "lower", "upper")))
    })
}

versions <- as.list(sapply(c("survRM2", "survival"), function(pkg) {
    as.character(utils::packageVersion(pkg))
}))
results <- lapply(cases, function(case) {
    alpha <- if (is.null(case$alpha)) 0.05 else case$alpha
    covariates <- case$covariates
    if (!is.null(covariates)) {
        if (is.list(covariates[[1L]])) {
            covariates <- do.call(rbind, lapply(covariates, unlist))
            colnames(covariates) <- paste0("x", seq_len(ncol(covariates)))
        } else {
            covariates <- unlist(covariates)
        }
    }
    fit <- suppressWarnings(survRM2::rmst2(
        unlist(case$time), unlist(case$status), unlist(case$arm),
        tau = case$tau, covariates = covariates, alpha = alpha
    ))
    if (!is.null(covariates)) {
        contrasts <- fit$adjusted.result
        return(list(
            tau = fit$tau,
            alpha = alpha,
            rmst_difference = contrast_result(contrasts[1, ]),
            rmst_ratio = contrast_result(contrasts[2, ]),
            rmtl_ratio = contrast_result(contrasts[3, ]),
            rmst_difference_model = model_result(fit$RMST.difference.adjusted),
            rmst_ratio_model = model_result(fit$RMST.ratio.adjusted),
            rmtl_ratio_model = model_result(fit$RMTL.ratio.adjusted)
        ))
    }
    contrasts <- fit$unadjusted.result
    list(
        tau = fit$tau,
        alpha = alpha,
        arm0 = arm_result(fit$RMST.arm0),
        arm1 = arm_result(fit$RMST.arm1),
        rmst_difference = contrast_result(contrasts[1, ]),
        rmst_ratio = contrast_result(contrasts[2, ]),
        rmtl_ratio = contrast_result(contrasts[3, ])
    )
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_versions = versions, results = results), auto_unbox = TRUE,
    digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
