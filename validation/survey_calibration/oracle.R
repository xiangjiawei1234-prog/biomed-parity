args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript --vanilla oracle.R cases.json")
fixture <- jsonlite::read_json(args[[1L]])
default <- function(x, fallback) if (is.null(x)) fallback else x
rows <- function(values) do.call(rbind, lapply(values, unlist))
controls <- function(weights, model_matrix) unname(drop(crossprod(weights, model_matrix)))
obs <- fixture$observations
d <- data.frame(w = unlist(obs$weights), psu = unlist(obs$psu),
    stratum = unlist(obs$strata), fpc = unlist(obs$fpc))
outcomes <- rows(obs$outcomes)
colnames(outcomes) <- c("pressure", "event")
factors <- rows(obs$repweight_factors)

pack_total <- function(design, return_replicates = FALSE) {
    fit <- survey::svytotal(outcomes, design, return.replicates = return_replicates)
    packed <- list(estimate = unname(as.numeric(coef(fit))), covariance = unname(vcov(fit)),
        standard_errors = unname(as.numeric(survey::SE(fit))), design_df = survey::degf(design))
    if (return_replicates) packed$replicate_estimates <- unname(fit$replicates)
    packed
}

results <- lapply(fixture$cases, function(case) {
    matrix <- rows(case$variables)
    colnames(matrix) <- paste0("x", seq_len(ncol(matrix)))
    population <- unlist(case$population)
    d[colnames(matrix)] <- as.data.frame(matrix)
    formula <- reformulate(colnames(matrix), intercept = FALSE)
    options <- list(formula = formula, population = population,
        calfun = default(case$calfun, "linear"), variance = unlist(case$variance), epsilon = 1e-12)
    linear <- survey::svydesign(ids = ~psu, strata = ~stratum, weights = ~w, fpc = ~fpc, data = d, nest = TRUE)
    if (!is.null(case$maxit)) options <- c(options, list(maxit = case$maxit, force = TRUE))
    calibrated <- suppressWarnings(do.call(survey::calibrate, c(list(design = linear), options)))
    w <- weights(calibrated)
    achieved <- controls(w, matrix)
    relative_errors <- abs(achieved - population) / (1 + abs(population))
    result <- list(weights = unname(w), multipliers = unname(w / d$w), control_totals = achieved,
        residuals = achieved - population, relative_errors = relative_errors,
        rank = qr(matrix * sqrt(d$w))$rank, converged = all(relative_errors < options$epsilon))
    if (result$converged) result$taylor_total <- pack_total(calibrated)
    if (!is.null(case$replication)) {
        combined <- case$replication$combined_weights
        rep_des <- survey::svrepdesign(data = d, repweights = if (combined) factors * d$w else factors,
            weights = d$w, type = "BRR", combined.weights = combined, degf = case$replication$degf, mse = TRUE)
        rep_cal <- do.call(survey::calibrate, c(list(design = rep_des, compress = FALSE), options))
        result$replication <- list(
            analysis_weights = unname(weights(rep_cal, type = "analysis")), df = survey::degf(rep_cal),
            total = pack_total(rep_cal, return_replicates = TRUE)
        )
    }
    result
})
names(results) <- vapply(fixture$cases, function(case) case$name, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    survey_version = as.character(utils::packageVersion("survey")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
