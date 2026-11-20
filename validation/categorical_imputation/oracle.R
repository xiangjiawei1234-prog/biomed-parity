args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

compare <- function(case) {
    unpack <- function(row) vapply(row, function(value) {
        if (is.null(value)) NA_real_ else value
    }, numeric(1))
    d <- as.data.frame(do.call(rbind, lapply(case$data, unpack)))
    levels <- unlist(case$levels)
    y <- factor(d[[ncol(d)]], levels = levels, ordered = case$method == "polr")
    x <- d[-ncol(d)]
    if (!is.null(case$predictor_levels)) {
        x[[2L]] <- factor(x[[2L]], levels = unlist(case$predictor_levels))
    }
    x <- model.matrix(~ ., x)[, -1L, drop = FALSE]
    names_x <- paste0("x", seq_len(ncol(x)))
    colnames(x) <- names_x
    ry <- !is.na(y)
    wy <- !ry
    aug <- getFromNamespace("augment", "mice")(y, ry, x, wy)
    train <- data.frame(y = aug$y[aug$ry], aug$x[aug$ry, , drop = FALSE])
    future <- as.data.frame(x[wy, , drop = FALSE])
    weights <- aug$w[aug$ry]
    fit <- if (case$method == "polr" && length(levels) > 2L) {
        suppressWarnings(MASS::polr(
            y ~ ., data = train, weights = weights, control = list(reltol = 1e-13)
        ))
    } else {
        nnet::multinom(y ~ ., data = train, weights = weights,
                       trace = FALSE, maxit = 2000, reltol = 1e-13)
    }
    predicted <- predict(fit, newdata = future, type = "probs")
    probabilities <- matrix(predicted, nrow = sum(wy), ncol = length(levels))
    if (length(levels) == 2L) probabilities <- cbind(1 - predicted, predicted)
    probabilities <- probabilities[, match(as.character(levels), fit$lev), drop = FALSE]
    sampled <- NULL
    if (case$draws > 0L) {
        set.seed(9281)
        kernel <- getExportedValue("mice", paste0("mice.impute.", case$method))
        draws <- replicate(case$draws, kernel(y, ry, x))
        sampled <- sapply(as.character(levels), function(level) rowMeans(draws == level))
    }
    list(probabilities = probabilities, sampled = sampled)
}

results <- lapply(cases, compare)
names(results) <- vapply(cases, function(case) case$id, character(1))
packages <- c("mice", "nnet", "MASS")
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    packages = as.list(setNames(vapply(packages, function(package)
        as.character(utils::packageVersion(package)), character(1)), packages)),
    contrasts = lapply(c(3, 5, 9), stats::contr.poly),
    results = results
), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
