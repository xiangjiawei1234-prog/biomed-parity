args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

evaluate <- function(case) {
    i <- seq_len(40)
    x <- ((i * 7) %% 13 - 6) / 4
    z <- as.numeric((i * 5) %% 11 > 5)
    lp <- -.36 * x + if (isTRUE(case$two_covariates)) .22 * z else 0
    latent <- exp(1.1 + lp + .48 * (sin(i * 1.7) + cos(i * .93)))
    left <- floor(latent * 2) / 2
    right <- left + .5
    right[left >= 4.5] <- Inf
    left[left >= 4.5] <- 4.5
    left[right <= 1.5] <- 0
    exact <- i %% 7L == 0L & left > 0 & left < 4.5
    left[exact] <- right[exact] <- latent[exact]
    weights <- if (isTRUE(case$weighted)) .5 + ((i * 11L) %% 9L) / 6 else rep(1, length(i))
    bounds <- if (is.null(case$bounds)) c(0, 1) else unlist(case$bounds)
    form <- if (isTRUE(case$two_covariates)) cbind(left, right) ~ x + z else cbind(left, right) ~ x
    fit <- icenReg::ic_sp(form, data = data.frame(left, right, x, z),
                         model = case$model, B = bounds, weights = weights)
    curves <- icenReg::getSCurves(fit, newdata = data.frame(x = c(0, .3), z = c(0, 1)))
    list(coefficients = I(unname(coef(fit))), intervals = unname(curves$Tbull_ints[-1, , drop = FALSE]),
         survival = unname(do.call(cbind, curves$S_curves)), log_likelihood = fit$llk)
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("icenReg")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
