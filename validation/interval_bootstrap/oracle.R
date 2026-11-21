args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
suppressPackageStartupMessages(library(icenReg))

evaluate <- function(case) {
    i <- seq_len(40)
    x <- ((i * 7) %% 13 - 6) / 4
    z <- as.numeric((i * 5) %% 11 > 5)
    latent <- exp(1.1 - .36 * x + .22 * z + .48 * (sin(i * 1.7) + cos(i * .93)))
    left <- floor(latent * 2) / 2
    right <- left + .5
    right[left >= 4.5] <- Inf
    left[left >= 4.5] <- 4.5
    left[right <= 1.5] <- 0
    exact <- i %% 7L == 0L & left > 0 & left < 4.5
    left[exact] <- right[exact] <- latent[exact]
    weights <- if (isTRUE(case$weighted)) .5 + ((i * 11L) %% 9L) / 6 else rep(1, length(i))
    bounds <- if (is.null(case$bounds)) c(0, 1) else unlist(case$bounds)
    level <- if (is.null(case$confidence_level)) .95 else case$confidence_level
    formula <- cbind(left, right) ~ x + z
    d <- data.frame(left, right, x, z)

    # Reconstruct the individual case draws from the same seeds as ic_sp.
    set.seed(case$seed)
    seeds <- as.integer(runif(case$n_resamples, 0, 2^31))
    draw_size <- ceiling(sum(weights))
    draws <- t(vapply(seeds, function(seed) {
        set.seed(seed)
        sample.int(nrow(d), draw_size, replace = TRUE, prob = weights)
    }, integer(draw_size)))
    set.seed(case$seed)
    official <- ic_sp(formula, d, model = case$model, B = bounds,
                      weights = weights, bs_samples = case$n_resamples)
    if (isTRUE(case$singular_draw)) draws <- rbind(draws, rep(1L, draw_size))

    # Independently refit through the public interface using draw counts.
    coefficients <- t(apply(draws, 1, function(draw) {
        counts <- tabulate(draw, nrow(d))
        keep <- counts > 0
        subset <- d[keep, , drop = FALSE]
        if (qr(cbind(1, subset$x, subset$z))$rank < 3) return(rep(NA_real_, 2))
        refit <- ic_sp(formula, subset, model = case$model, B = bounds, weights = counts[keep])
        unname(coef(refit))
    }))
    successful <- complete.cases(coefficients)
    replicates <- coefficients[successful, , drop = FALSE]
    stopifnot(isTRUE(all.equal(unname(official$bsMat), unname(replicates), tolerance = 2e-6)))
    covariance <- cov(replicates)
    stopifnot(isTRUE(all.equal(unname(vcov(official)), covariance, tolerance = 2e-6)))

    estimate <- unname(coef(official))
    standard_error <- sqrt(diag(covariance))
    z_value <- estimate / standard_error
    intervals <- unname(confint(official, level = level))
    list(id = case$id, left = left, right = right, design = unname(cbind(x, z)), weights = weights,
         bootstrap_indices = unname(draws - 1L), coefficients = estimate, replicates = unname(replicates),
         failed_replicates = I(which(!successful) - 1L), covariance = covariance,
         standard_error = standard_error, z = z_value, pvalue = 2 * pnorm(-abs(z_value)),
         lower = intervals[, 1], upper = intervals[, 2])
}

results <- lapply(cases, evaluate)
cat(jsonlite::toJSON(list(R = as.character(getRversion()), icenReg = as.character(packageVersion("icenReg")),
    cases = results), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
