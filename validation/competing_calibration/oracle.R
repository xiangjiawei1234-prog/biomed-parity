suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(riskRegression))
Hist <- prodlim::Hist

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

default <- function(value, fallback) if (is.null(value)) fallback else value

results <- lapply(cases, function(case) {
    data <- data.frame(time = unlist(case$time), status = unlist(case$status))
    cause <- default(case$cause, 1L)
    horizon <- case$horizon
    if (isTRUE(case$pseudo_only)) {
        fit <- prodlim::prodlim(Hist(time, status) ~ 1, data = data)
        return(list(pseudo_values = as.numeric(prodlim::jackknife(fit, times = horizon, cause = cause))))
    }
    data$risk <- unlist(case$risks)
    index <- order(data$time, data$status == 0, method = "radix")
    data <- data[index, ]
    compiled <- if (any(data$status > 1)) Hist(time, status) ~ 1 else Surv(time, status) ~ 1
    method <- default(case$method, "quantile")
    estimator <- default(case$estimator, "local")
    digits <- default(case$round_digits, 2L)
    score <- Score(
        list(risk = data$risk), compiled, data = data, times = horizon, cause = cause,
        plots = "calibration", metrics = NULL, null.model = FALSE,
        cens.method = if (estimator == "jackknife") "pseudo" else "ipcw"
    )
    graph <- plotCalibration(
        score, times = horizon, method = method, cens.method = estimator,
        q = unlist(default(case$groups, 10L)), bandwidth = case$bandwidth, round = digits,
        plot = FALSE, legend = FALSE
    )$plotFrames[[1L]]
    if (method == "quantile") {
        breaks <- attr(graph, "quantiles")
        count <- if (length(unique(data$risk)) <= length(breaks)) {
            as.numeric(table(data$risk))
        } else {
            as.numeric(table(cut(data$risk, breaks, include.lowest = TRUE)))
        }
    } else {
        breaks <- NULL
        count <- if (case$bandwidth >= 1) nrow(data) else {
            r <- if (digits > 0) round(data$risk, digits) else data$risk
            prodlim::neighborhood(r, bandwidth = case$bandwidth)$size.nbh
        }
    }
    pseudo <- NULL
    if (estimator == "jackknife") {
        frame <- score$Calibration$plotframe
        pseudo <- frame$pseudovalue[order(frame$riskRegression_ID)][order(index)]
    }
    list(horizon = horizon, predicted = graph$Pred, observed = as.numeric(graph$Obs),
        observations = count, cutpoints = breaks, pseudo_values = pseudo)
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    package_versions = as.list(vapply(c("riskRegression", "prodlim"), function(name) {
        as.character(utils::packageVersion(name))
    }, character(1))),
    results = results
), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string", null = "null"), "\n", sep = "")
