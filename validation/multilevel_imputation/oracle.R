args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript oracle.R {structure|distribution} cases.json")
experiment <- match.arg(args[[1L]], c("structure", "distribution"))
spec <- jsonlite::read_json(args[[2L]], simplifyVector = FALSE)

default <- function(value, fallback) if (is.null(value)) fallback else value

as_rows <- function(rows) do.call(rbind, lapply(rows, function(row) {
    vapply(row, function(value) default(value, NA), numeric(1))
}))

case_inputs <- function(case) {
    dataset <- spec$datasets[[case$dataset]]
    columns <- vapply(dataset$columns, identity, character(1))
    data <- as_rows(dataset$rows)
    colnames(data) <- columns
    chosen <- vapply(case$predictors, identity, character(1))
    x <- data[, chosen, drop = FALSE]
    list(y = as.numeric(data[, "y"]), x = x, ry = !is.na(data[, "y"]))
}

# The Python entry point receives the response, the predictor matrix and the
# type codes for one incomplete column.  This mirrors that call through the
# published mice kernel.
impute_method <- function(case, y, x, ry) {
    type <- as.numeric(unlist(case$types))
    if (case$method == "2l.pan") {
        mice::mice.impute.2l.pan(
            y, ry, x, type = type, intercept = case$intercept,
            groupcenter.slope = default(case$groupcenter_slope, FALSE)
        )
    } else {
        mice::mice.impute.2l.norm(y, ry, x, type = type, intercept = case$intercept)
    }
}

# Recover the deterministic design matrix that mice hands to the pan Gibbs
# sampler.  Capturing the real call keeps the reference independent of how the
# aggregation is re-derived here.
pan_designs <- function(case, y, x, ry) {
    trace(
        "pan",
        tracer = quote({
            assign("captured_pan_design",
                   list(pred = pred, xcol = xcol, zcol = zcol,
                        subj = subj, prior = prior),
                   envir = globalenv())
        }),
        print = FALSE, where = asNamespace("pan")
    )
    on.exit({
        untrace("pan", where = asNamespace("pan"))
        if (exists("captured_pan_design", envir = globalenv())) {
            rm("captured_pan_design", envir = globalenv())
        }
    }, add = TRUE)
    impute_method(case, y, x, ry)
    value <- get("captured_pan_design", envir = globalenv())
    observed <- which(ry)
    list(
        fixed_design = unname(value$pred[observed, value$xcol, drop = FALSE]),
        random_design = unname(value$pred[, value$zcol, drop = FALSE]),
        xcol = as.integer(value$xcol),
        zcol = as.integer(value$zcol),
        prior = list(
            a = as.numeric(value$prior$a),
            Binv = unname(value$prior$Binv),
            c = as.numeric(value$prior$c),
            Dinv = unname(value$prior$Dinv)
        )
    )
}

norm_designs <- function(case, y, x, ry) {
    prepared <- cbind(1, as.matrix(x))
    type <- c(2, as.numeric(unlist(case$types)))
    if (!isTRUE(case$intercept)) {
        prepared <- as.matrix(x)
        type <- as.numeric(unlist(case$types))
    }
    selected <- type == 2
    list(
        fixed_design = NULL,
        random_design = unname(prepared[ry, selected, drop = FALSE]),
        n_random = sum(selected)
    )
}

run_structure <- function(case) {
    inputs <- case_inputs(case)
    y <- inputs$y
    x <- inputs$x
    ry <- inputs$ry
    cluster <- x[, as.numeric(unlist(case$types)) == -2]
    result <- tryCatch({
        imputed <- impute_method(case, y, x, ry)
        list(
            n_imputed = length(imputed),
            imputed_rows = as.integer(which(!ry) - 1L),
            imputed_finite = all(is.finite(imputed)),
            r_error = NULL
        )
    }, error = function(error) {
        list(n_imputed = NA_integer_, imputed_rows = integer(0),
             imputed_finite = NA, r_error = conditionMessage(error))
    })
    designs <- tryCatch({
        if (case$method == "2l.pan") pan_designs(case, y, x, ry) else norm_designs(case, y, x, ry)
    }, error = function(error) list(design_error = conditionMessage(error)))
    list(
        n_imputed = result$n_imputed,
        imputed_rows = result$imputed_rows,
        imputed_finite = result$imputed_finite,
        r_error = result$r_error,
        fixed_design = default(designs$fixed_design, NULL),
        random_design = default(designs$random_design, NULL),
        xcol = default(designs$xcol, NULL),
        zcol = default(designs$zcol, NULL),
        prior = default(designs$prior, NULL),
        n_random = default(designs$n_random, NULL),
        design_error = default(designs$design_error, NULL),
        observed_rows = as.integer(which(ry) - 1L),
        n_groups = length(unique(cluster))
    )
}

distribution_draws <- function(case, y, x, ry) {
    m <- as.integer(case$m)
    seeds <- local({
        set.seed(as.integer(case$seed))
        sample.int(.Machine$integer.max, m)
    })
    cores <- max(1L, min(4L, parallel::detectCores()))
    values <- parallel::mclapply(seeds, function(seed) {
        set.seed(seed)
        as.numeric(impute_method(case, y, x, ry))
    }, mc.cores = cores, mc.set.seed = FALSE)
    failed <- vapply(values, inherits, logical(1), what = "try-error")
    if (any(failed)) stop(conditionMessage(attr(values[[which(failed)[1L]]], "condition")))
    do.call(cbind, values)
}

run_distribution <- function(case) {
    inputs <- case_inputs(case)
    y <- inputs$y
    x <- inputs$x
    ry <- inputs$ry
    # observed_rows are rows the Python entry point withholds from the model;
    # mice expresses the same training set by setting those responses to NA.
    withheld <- as.integer(default(case$observed_rows, numeric(0))) + 1L
    y[withheld] <- NA
    ry <- !is.na(y)
    draws <- distribution_draws(case, y, x, ry)
    targets <- as.integer(default(case$target_rows, numeric(0))) + 1L
    positions <- if (length(targets)) match(targets, which(!ry)) else seq_len(nrow(draws))
    if (anyNA(positions)) stop("target_rows must be unobserved rows")
    draws <- draws[positions, , drop = FALSE]
    means <- rowMeans(draws)
    standard_deviations <- apply(draws, 1L, sd)
    centered <- draws - means
    covariance_se <- outer(seq_len(nrow(draws)), seq_len(nrow(draws)), Vectorize(
        function(j, k) sd(centered[j, ] * centered[k, ]) / sqrt(ncol(draws))
    ))
    list(
        means = means,
        standard_deviations = standard_deviations,
        covariance = cov(t(draws)),
        covariance_se = covariance_se,
        n_draws = ncol(draws)
    )
}

runner <- if (experiment == "structure") run_structure else run_distribution
results <- lapply(spec$cases, runner)
names(results) <- vapply(spec$cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    package_version = as.character(utils::packageVersion("mice")),
    pan_version = tryCatch(as.character(utils::packageVersion("pan")), error = function(e) NA),
    experiment = experiment,
    results = results
), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null", null = "null"), "\n", sep = "")
