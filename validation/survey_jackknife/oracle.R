args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript --vanilla oracle.R cases.json")
suppressPackageStartupMessages(library(survey))
options(survey.drop.replicates = TRUE, survey.replicates.mse = FALSE)
fixture <- jsonlite::read_json(args[[1L]])
default <- function(x, fallback) if (is.null(x)) fallback else x
outcomes <- do.call(rbind, lapply(fixture$outcomes, unlist))
weights <- unlist(fixture$weights)

pack <- function(fit) {
    list(estimate = unname(as.numeric(coef(fit))), covariance = unname(vcov(fit)),
        standard_errors = unname(as.numeric(SE(fit))), replicate_estimates = unname(fit$replicates))
}

results <- lapply(fixture$cases, function(case) {
    original <- fixture$designs[[case$design]]
    settings <- case$options
    rows <- data.frame(w = weights)
    if (!is.null(original$psu)) rows$psu <- unlist(original$psu)
    if (!is.null(original$strata)) rows$stratum <- unlist(original$strata)
    options(survey.lonely.psu = default(settings$lonely_psu, "fail"))
    fpc <- unlist(settings$fpc)
    fpctype <- if (is.null(fpc)) NULL else default(settings$fpctype, "fraction")
    inherited_fpc <- NULL
    if (isTRUE(case$inherit_fpc)) {
        rows$fpc <- fpc
        inherited_fpc <- ~fpc
        fpc <- fpctype <- NULL
    }
    design <- svydesign(ids = if (is.null(original$psu)) ~1 else ~psu,
        strata = if (is.null(original$strata)) NULL else ~stratum,
        weights = ~w, fpc = inherited_fpc, data = rows, nest = TRUE)
    converted <- tryCatch(as.svrepdesign(design, type = default(settings$type, "auto"),
        fpc = fpc, fpctype = fpctype, mse = default(settings$mse, FALSE), compress = TRUE),
        error = function(error) error)
    if (inherits(converted, "error")) {
        if (is.null(case$error) || !grepl(case$error, conditionMessage(converted), fixed = TRUE)) stop(converted)
        return(list(error = conditionMessage(converted)))
    }
    if (!is.null(case$error)) stop("Expected conversion failure for ", case$name)

    list(type = converted$type, scale = converted$scale, rscales = converted$rscales,
        combined_weights = converted$combined.weights, mse = converted$mse, design_df = degf(converted),
        sampling_weights = unname(weights(converted, "sampling")),
        replicate_weights = unname(weights(converted, "replication")),
        analysis_weights = unname(weights(converted, "analysis")),
        mean = pack(svymean(outcomes, converted, return.replicates = TRUE)),
        total = pack(svytotal(outcomes, converted, return.replicates = TRUE)))
})
names(results) <- vapply(fixture$cases, function(case) case$name, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    survey_version = as.character(packageVersion("survey")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
