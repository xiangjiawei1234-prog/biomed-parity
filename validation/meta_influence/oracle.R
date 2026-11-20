args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
j <- 0:23
group <- rep(c(4, 1, 6, 3, 2, 5), each = 4L)
x <- ((j * 7) %% 17) / 13
rare <- as.numeric(group == 4)
y <- .7 - .4 * x + .6 * sin(group * 3) + .2 * cos(j)
vi <- .11 + ((j * 3) %% 5) / 40
data <- data.frame(group, x, rare, y)
V <- .2 * outer(sqrt(vi), sqrt(vi)) * outer(group, group, "==")
diag(V) <- vi

evaluate <- function(case) {
    fit <- function(keep = rep(TRUE, nrow(data)), sigma = NA_real_) {
        moderators <- if (isTRUE(case$rare)) ~ x + rare else ~ x
        if (isTRUE(case$random)) {
            return(suppressWarnings(metafor::rma.mv(
                y, V, mods = moderators, random = ~ 1 | group, sigma2 = sigma,
                subset = keep, test = case$test, data = data
            )))
        }
        suppressWarnings(metafor::rma.mv(y, V, mods = moderators, subset = keep,
                                        test = case$test, data = data))
    }
    overall <- fit()
    options <- list(overall, reestimate = case$reestimate)
    if (isTRUE(case$clustered)) options$cluster <- group
    cook_options <- options
    if (!is.null(case$cook_coefficients)) cook_options$btt <- unlist(case$cook_coefficients) + 1L
    cooks <- suppressWarnings(do.call(stats::cooks.distance, cook_options))
    dfbs <- suppressWarnings(do.call(stats::dfbetas, options))
    res <- suppressWarnings(do.call(stats::rstudent, options))
    labels <- if (isTRUE(case$clustered)) unique(group) else seq_along(y)
    order <- if (isTRUE(case$clustered)) match(labels, res$cluster$slab) else seq_along(y)
    obs <- if (isTRUE(case$clustered)) res$obs else res
    chi <- if (isTRUE(case$clustered)) res$cluster$X2[order] else obs$z^2
    df <- if (isTRUE(case$clustered)) res$cluster$k[order] else rep(1L, length(y))
    deleted <- lapply(labels, function(label) {
        cases <- if (isTRUE(case$clustered)) group else seq_along(y)
        refit <- try(fit(cases != label, if (case$reestimate) NA_real_ else overall$sigma2[1]),
                     silent = TRUE)
        if (inherits(refit, "try-error") || length(coef(refit)) != length(coef(overall))) {
            return(list(coefficients = rep(NA_real_, length(coef(overall))), variance = NA_real_))
        }
        list(coefficients = unname(coef(refit)), variance = refit$sigma2[1])
    })
    list(labels = I(labels - as.integer(!isTRUE(case$clustered))),
         cooks = I(unname(cooks[order])), dfbetas = unname(as.matrix(dfbs)[order, , drop = FALSE]),
         residuals = I(obs$resid), residual_se = I(obs$se), studentized = I(obs$z),
         cluster_chi_squared = I(chi), cluster_df = I(df),
         leverage = I(unname(hatvalues(overall))), hat = unname(as.matrix(hatvalues(overall, type = "matrix"))),
         deleted_coefficients = do.call(rbind, lapply(deleted, function(value) value$coefficients)),
         deleted_variances = if (isTRUE(case$random)) I(vapply(deleted, function(value) value$variance, 0.0)) else NULL)
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("metafor")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null", null = "null"), "\n", sep = "")
