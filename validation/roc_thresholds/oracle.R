#!/usr/bin/env Rscript
#
# Generate the stored reference snapshot for biomed_parity.roc_thresholds.
#
# Everything is produced from pROC 1.19.1 on the same fixed inputs:
#   * coordinate scenarios are plain pROC::coords() calls;
#   * bootstrap scenarios use the pROC::ci.coords() stratified/non-stratified
#     resampling routines on a fixed index matrix, so the Python module can be
#     driven with the very same draws.
#
# Fixed index matrices are emitted with `--draws`; the expected long table is
# printed otherwise.  The script also cross-checks the manual bootstrap loop
# against the actual pROC::ci.coords() implementation before printing.
#
# pROC 1.19.1 bug: enforce.best.policy("omit") uses `res[1, drop = FALSE]`
# instead of `res[1, , drop = FALSE]`, returning the first *column* of every
# tied row.  ci.coords() results for the omit policy therefore recycle draws and
# are not meaningful.  For that single scenario the oracle evaluates the
# documented semantics (an NA row per ambiguous draw) and verifies it against
# pROC with the helper patched; see `with_documented_best_policy`.

arguments <- commandArgs(trailingOnly = TRUE)
fixture_dir <- if (length(arguments)) arguments[[1]] else "validation/roc_thresholds"
emit_draws <- "--draws" %in% arguments

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
suppressPackageStartupMessages(library(pROC))

data <- read.delim(file.path(fixture_dir, "observations.tsv"))
cases <- read.delim(
    file.path(fixture_dir, "cases.tsv"), colClasses = "character",
    na.strings = "", check.names = FALSE
)
labels <- data$label
n_subjects <- nrow(data)
n_controls <- sum(labels == 0)
n_cases <- sum(labels == 1)
control_rows <- seq_len(n_controls)
case_rows <- n_controls + seq_len(n_cases)
n_boot <- 40

scores_of <- function(name) data[[name]]

is_empty <- function(value) is.na(value) || identical(value, "")
parse_numbers <- function(value) as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
parse_ret <- function(value) {
    if (is_empty(value)) NULL else if (value == "all") "all" else {
        strsplit(value, ",", fixed = TRUE)[[1]]
    }
}
parse_x <- function(value) {
    if (startsWith(value, "num:")) {
        list(numeric = TRUE, value = parse_numbers(substr(value, 5, nchar(value))))
    } else {
        list(numeric = FALSE, value = value)
    }
}

make_roc <- function(score, direction, percent, partial, focus) {
    args <- list(
        response = labels, predictor = scores_of(score), direction = direction,
        levels = c(0, 1), percent = percent, quiet = TRUE
    )
    if (!is_empty(partial)) {
        args$partial.auc <- parse_numbers(partial)
        args$partial.auc.focus <- focus
    }
    suppressWarnings(do.call(pROC::roc, args))
}

coords_of <- function(roc, x, input, ret, method, weights) {
    args <- list(roc = roc, x = x, input = input, best.method = method,
                 best.weights = weights)
    if (!is.null(ret)) args$ret <- ret
    do.call(pROC::coords, args)
}

# --- fixed bootstrap index matrices -----------------------------------------
# Controls occupy the first rows of observations.tsv, so R indices and the
# zero-based Python row indices differ only by one.

stratified_draws <- function(seed) {
    set.seed(seed)
    t(replicate(n_boot, c(
        sample(control_rows, replace = TRUE), sample(case_rows, replace = TRUE)
    )))
}

resample_stratified <- function(template, score, draw) {
    controls <- scores_of(score)[draw[seq_len(n_controls)]]
    cases <- scores_of(score)[draw[n_controls + seq_len(n_cases)]]
    thresholds <- pROC:::roc_utils_thresholds(
        c(cases, controls), template$direction
    )
    perfs <- pROC:::roc_utils_perfs_all(
        thresholds = thresholds, controls = controls, cases = cases,
        direction = template$direction
    )
    roc <- template
    roc$sensitivities <- perfs$se
    roc$specificities <- perfs$sp
    roc$cases <- cases
    roc$controls <- controls
    roc$predictor <- c(controls, cases)
    roc$response <- c(rep(template$levels[1], n_controls),
                      rep(template$levels[2], n_cases))
    roc$thresholds <- thresholds
    roc
}

resample_nonstratified <- function(template, score, index) {
    predictor <- template$predictor[index]
    response <- template$response[index]
    splitted <- split(predictor, response)
    controls <- splitted[[as.character(template$levels[1])]]
    cases <- splitted[[as.character(template$levels[2])]]
    if (is.null(controls)) controls <- numeric(0)
    if (is.null(cases)) cases <- numeric(0)
    if (!length(controls) || !length(cases)) return(NULL)
    thresholds <- pROC:::roc_utils_thresholds(
        c(cases, controls), template$direction
    )
    perfs <- pROC:::roc_utils_perfs_all(
        thresholds = thresholds, controls = controls, cases = cases,
        direction = template$direction
    )
    roc <- template
    roc$sensitivities <- perfs$se
    roc$specificities <- perfs$sp
    roc$cases <- cases
    roc$controls <- controls
    roc$predictor <- c(controls, cases)
    roc$response <- c(rep(template$levels[1], length(controls)),
                      rep(template$levels[2], length(cases)))
    roc$thresholds <- thresholds
    roc
}

enforce_policy <- function(res, policy, choice = NULL) {
    if (nrow(res) == 1) return(res)
    if (policy == "stop") {
        stop("more than one best threshold")
    }
    if (policy == "omit") {
        res[1, ] <- NA
        return(res[1, , drop = FALSE])
    }
    if (is.null(choice)) choice <- sample(seq_len(nrow(res)), 1)
    res[choice, , drop = FALSE]
}

# Mirror pROC's RNG consumption exactly for the "random" best policy: every
# draw samples controls, cases and - when several best points tie - one row.
random_draws <- function(template, score, seed) {
    set.seed(seed)
    draws <- matrix(NA_integer_, n_boot, n_subjects)
    choices <- rep(NA_integer_, n_boot)
    for (i in seq_len(n_boot)) {
        draw <- c(
            sample(control_rows, replace = TRUE),
            sample(case_rows, replace = TRUE)
        )
        draws[i, ] <- draw
        res <- coords_of(
            resample_stratified(template, score, draw), "best", "threshold",
            c("threshold", "specificity", "sensitivity"), "youden", c(1, 0.5)
        )
        choices[i] <- if (nrow(res) > 1) sample(seq_len(nrow(res)), 1) else 1L
    }
    list(draws = draws, choices = choices)
}

template_tiebest <- make_roc("score_tiebest", "<", FALSE, "", "specificity")

draws_main <- stratified_draws(90210)
draws_perfect <- stratified_draws(312)
draws_policy <- stratified_draws(2024)
draws_random_info <- random_draws(template_tiebest, "score_tiebest", 11)
draws_random <- draws_random_info$draws
draws_random_choices <- draws_random_info$choices
set.seed(101)
draws_nonstrat <- t(replicate(n_boot, sample(seq_len(n_subjects), replace = TRUE)))

draws_by_name <- list(
    draws_main = draws_main, draws_perfect = draws_perfect,
    draws_policy = draws_policy, draws_random = draws_random,
    draws_nonstrat = draws_nonstrat
)
seed_by_name <- list(
    draws_main = 90210, draws_perfect = 312, draws_policy = 2024,
    draws_random = 11, draws_nonstrat = 101
)

if (emit_draws) {
    for (name in names(draws_by_name)) {
        cat("# ", name, "\n", sep = "")
        write.table(draws_by_name[[name]] - 1L, sep = "\t", row.names = FALSE,
                    col.names = FALSE, quote = FALSE)
    }
    quit(save = "no")
}

# --- manual bootstrap loop ---------------------------------------------------

bootstrap_performance <- function(template, score, draws, x, input, ret,
                                  method, weights, policy, stratified,
                                  choices = NULL) {
    perfs <- vector("list", nrow(draws))
    for (i in seq_len(nrow(draws))) {
        roc <- if (stratified) {
            resample_stratified(template, score, draws[i, ])
        } else {
            resample_nonstratified(template, score, draws[i, ])
        }
        if (is.null(roc)) {
            perfs[[i]] <- NA
            next
        }
        res <- coords_of(roc, x, input, ret, method, weights)
        if (is.character(x) && length(x) == 1 && x == "best" && nrow(res) != 1) {
            if (policy == "random") {
                choice <- if (is.null(choices)) NULL else choices[i]
                res <- enforce_policy(res, "random", choice)
            } else {
                res <- enforce_policy(res, policy)
            }
        }
        perfs[[i]] <- res
    }
    perfs
}

summarize_perfs <- function(perfs, n_points, n_fields, confidence_level,
                            fields) {
    perfs <- lapply(perfs, as.matrix)
    array_values <- array(
        unlist(perfs), dim = c(n_points, n_fields, length(perfs)),
        dimnames = list(NULL, fields, NULL)
    )
    suppressWarnings(apply(
        array_values, 1:2, quantile,
        probs = c(0 + (1 - confidence_level) / 2, 0.5,
                  1 - (1 - confidence_level) / 2),
        na.rm = TRUE, type = 7
    ))
}

counts_perfs <- function(perfs, n_points, n_fields, fields) {
    array_values <- array(
        unlist(lapply(perfs, as.matrix)), dim = c(n_points, n_fields, length(perfs)),
        dimnames = list(NULL, fields, NULL)
    )
    apply(array_values, 1:2, function(values) sum(!is.na(values)))
}

format_value <- function(value) {
    if (is.na(value)) {
        "NA"
    } else if (is.infinite(value)) {
        if (value > 0) "Inf" else "-Inf"
    } else {
        sprintf("%.17g", value)
    }
}

result <- list()
emit <- function(id, kind, row = NA_integer_, field = NA_character_,
                 quantile = NA_character_, value) {
    result[[length(result) + 1L]] <<- data.frame(
        id = id, kind = kind, row = row, field = field, quantile = quantile,
        value = value, stringsAsFactors = FALSE
    )
}

emit_matrix <- function(id, kind, matrix_values, quantile = NA_character_,
                        rows = seq_len(nrow(matrix_values))) {
    for (i in rows) {
        for (field in colnames(matrix_values)) {
            emit(id, kind, i, field, quantile,
                 format_value(matrix_values[i, field]))
        }
    }
}

# The documented enforce.best.policy() semantics, used to verify the omit
# scenario against pROC despite the 1.19.1 indexing bug.
documented_enforce <- function(res, best.policy) {
    if (best.policy == "stop") {
        stop("More than one \"best\" threshold was found, aborting.")
    }
    if (best.policy == "omit") {
        res[1, ] <- NA
        return(res[1, , drop = FALSE])
    }
    res[sample(seq_len(nrow(res)), size = 1), , drop = FALSE]
}
with_documented_best_policy <- function(code) {
    namespace <- asNamespace("pROC")
    original <- get("enforce.best.policy", envir = namespace)
    unlockBinding("enforce.best.policy", namespace)
    assignInNamespace("enforce.best.policy", documented_enforce, "pROC")
    on.exit({
        assignInNamespace("enforce.best.policy", original, "pROC")
        lockBinding("enforce.best.policy", namespace)
    }, add = TRUE)
    force(code)
}

check_against_proc <- function(template, x, input, ret, method, weights,
                               policy, confidence_level, stratified, draws_name,
                               expected_quantiles, expected_omitted,
                               tolerance = 1e-13) {
    set.seed(seed_by_name[[draws_name]])
    warnings_seen <- character()
    actual <- tryCatch(
        suppressWarnings(withCallingHandlers(
            ci.coords(
                template, x = x, input = input, ret = ret,
                best.method = method, best.weights = weights,
                best.policy = policy, conf.level = confidence_level,
                boot.n = n_boot, boot.stratified = stratified
            ),
            warning = function(warning) {
                warnings_seen <<- c(warnings_seen, conditionMessage(warning))
                invokeRestart("muffleWarning")
            }
        )),
        error = function(error) error
    )
    if (inherits(actual, "error")) return(actual)
    fields <- dimnames(expected_quantiles)[[3]]
    for (field in fields) {
        reference <- t(expected_quantiles[, , field])
        observed <- unname(actual[[field]])
        if (!isTRUE(all.equal(
            as.numeric(reference), as.numeric(observed),
            tolerance = tolerance, check.attributes = FALSE
        ))) {
            stop("pROC cross-check failed for ", field)
        }
    }
    ignored <- grep("NA value\\(s\\) produced during bootstrap were ignored",
                    warnings_seen, value = TRUE)
    if (length(ignored)) {
        reported <- as.integer(sub(
            "^(\\d+) NA value\\(s\\).*", "\\1", ignored[[1]]
        ))
        stopifnot(identical(reported, as.integer(expected_omitted)))
    } else if (expected_omitted > 0 && !any(grepl("All bootstrap iterations", warnings_seen))) {
        stop("pROC did not report the expected omitted bootstrap draws")
    }
    NULL
}

# --- coordinate scenarios ----------------------------------------------------

for (i in seq_len(nrow(cases))) {
    scenario <- cases[i, ]
    if (scenario$kind != "coords") next
    roc <- make_roc(
        scenario$score, scenario$direction, scenario$percent == "TRUE",
        scenario$partial_auc, scenario$focus
    )
    x <- parse_x(scenario$x)
    res <- coords_of(
        roc, x$value, scenario$input, parse_ret(scenario$ret),
        scenario$best_method, parse_numbers(scenario$best_weights)
    )
    as_matrix <- as.matrix(res)
    emit_matrix(scenario$id, "coord", as_matrix)
}

# --- bootstrap scenarios -----------------------------------------------------
# Percent-scale bootstrap scenarios are deliberately absent: pROC 1.19.1
# ci.coords() keeps roc$percent = TRUE while assigning resampled sensitivities
# and specificities on the 0-1 scale, which makes numeric coordinate inputs
# error out and mixes scales in the best-threshold criterion.  The Python module
# keeps the percent scale throughout, so that path cannot be aligned.

for (i in seq_len(nrow(cases))) {
    scenario <- cases[i, ]
    if (!scenario$kind %in% c("ci", "ci_error")) next
    template <- make_roc(
        scenario$score, scenario$direction, scenario$percent == "TRUE",
        scenario$partial_auc, scenario$focus
    )
    x <- parse_x(scenario$x)
    ret <- parse_ret(scenario$ret)
    weights <- parse_numbers(scenario$best_weights)
    confidence_level <- as.numeric(scenario$confidence_level)
    stratified <- scenario$stratified == "TRUE"
    draws <- draws_by_name[[scenario$draws]]
    choices <- if (scenario$draws == "draws_random") {
        draws_random_choices
    } else {
        NULL
    }
    # ci.coords() drops the threshold from the default ret unless thresholds are
    # the input; bootstrap_coords() follows the same rule.
    if (is.null(ret)) {
        ret <- if (identical(scenario$input, "threshold")) {
            c("threshold", "specificity", "sensitivity")
        } else {
            c("specificity", "sensitivity")
        }
    }
    observed <- coords_of(
        template, x$value, scenario$input, ret, scenario$best_method, weights
    )
    emit_matrix(scenario$id, "point", as.matrix(observed))

    if (scenario$kind == "ci_error") {
        set.seed(seed_by_name[[scenario$draws]])
        failure <- tryCatch({
            suppressWarnings(ci.coords(
                template, x = x$value, input = scenario$input, ret = ret,
                best.method = scenario$best_method, best.weights = weights,
                best.policy = scenario$best_policy,
                conf.level = confidence_level, boot.n = n_boot,
                boot.stratified = stratified
            ))
            NULL
        }, error = function(error) error)
        stopifnot(!is.null(failure))
        emit(scenario$id, "error", NA_integer_, NA_character_, NA_character_,
             "error")
        next
    }

    run_loop <- function(patch_policy = FALSE) {
        loop <- function() bootstrap_performance(
            template, scenario$score, draws, x$value, scenario$input, ret,
            scenario$best_method, weights, scenario$best_policy, stratified,
            choices
        )
        if (patch_policy) with_documented_best_policy(loop()) else loop()
    }
    patched <- scenario$best_policy == "omit" && scenario$best_method != ""
    perfs <- run_loop(patch_policy = patched)

    n_points <- if (is.character(x$value)) 1L else length(x$value)
    fields <- colnames(as.matrix(observed))
    n_fields <- length(fields)
    quantiles <- summarize_perfs(perfs, n_points, n_fields, confidence_level, fields)
    counts <- counts_perfs(perfs, n_points, n_fields, fields)
    omitted <- sum(vapply(perfs, function(perf) all(is.na(as.matrix(perf))), logical(1)))

    for (point in seq_len(n_points)) {
        for (field in fields) {
            for (q in seq_len(3)) {
                emit(
                    scenario$id, "interval", point, field,
                    c("lower", "median", "upper")[[q]],
                    format_value(quantiles[q, point, field])
                )
            }
            emit(scenario$id, "count", point, field, NA_character_,
                 format_value(counts[point, field]))
        }
    }
    emit(scenario$id, "summary", NA_integer_, "n_resamples", NA_character_,
         format_value(n_boot))
    emit(scenario$id, "summary", NA_integer_, "omitted", NA_character_,
         format_value(omitted))

    # Cross-check against the real pROC implementation.
    check <- if (patched) {
        with_documented_best_policy(check_against_proc(
            template, x$value, scenario$input, ret, scenario$best_method, weights,
            scenario$best_policy, confidence_level, stratified,
            scenario$draws, quantiles, omitted
        ))
    } else {
        check_against_proc(
            template, x$value, scenario$input, ret, scenario$best_method, weights,
            scenario$best_policy, confidence_level, stratified,
            scenario$draws, quantiles, omitted
        )
    }
    if (inherits(check, "error")) stop(conditionMessage(check))
}

output <- do.call(rbind, result)
cat(
    "# R ", as.character(getRversion()), ", pROC ",
    as.character(packageVersion("pROC")), "\n", sep = ""
)
write.table(output, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
