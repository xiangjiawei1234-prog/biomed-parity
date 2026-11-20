args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
suppressPackageStartupMessages(library(glmmTMB))
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
base <- dirname(dirname(normalizePath(args[[1L]])))
zip <- read.delim(file.path(base, "zero_inflated", "observations.tsv"))
nb <- jsonlite::read_json(file.path(base, "zinb2", "cases.json"), simplifyVector = FALSE)
names(nb) <- vapply(nb, function(case) case$id, character(1))

evaluate <- function(case) {
    poisson <- case$family == "poisson"
    if (poisson) {
        data <- zip[zip$dataset == case$data_id, ]
    } else {
        data <- as.data.frame(do.call(rbind, lapply(nb[[case$data_id]]$observations, unlist)))
        names(data) <- c("count", "group", "x", "z", "d", "offset")
    }
    data$group <- factor(data$group)
    fit <- glmmTMB(
        count ~ x + offset(offset) + (1 | group), ziformula = ~z,
        dispformula = if (poisson) ~1 else ~d, family = if (poisson) stats::poisson else nbinom2,
        data = data, control = glmmTMBControl(
            optimizer = optim, optArgs = list(method = "BFGS"), optCtrl = list(reltol = 1e-13, maxit = 600)
        )
    )
    rows <- unlist(case$rows) + 1L
    newdata <- data[rows, ]
    conditional <- function(population) {
        settings <- newdata
        if (!population) {
            settings$group <- as.character(settings$group)
            settings$group[4:5] <- c(NA_character_, "unseen")
        }
        values <- lapply(c("link", "conditional", "zlink", "zprob", "response", "disp"), function(scale) {
            predict(fit, newdata = settings, type = scale, allow.new.levels = TRUE,
                    re.form = if (population) NA else NULL)
        })
        names(values) <- c("link", "conditional_mean", "inflation_logit", "structural_zero_probability",
                           "response_mean", "dispersion")
        mu <- values$conditional_mean
        ordinary_zero <- if (poisson) dpois(0, mu) else dnbinom(0, size = values$dispersion, mu = mu)
        values$zero_probability <- values$structural_zero_probability +
            (1 - values$structural_zero_probability) * ordinary_zero
        values
    }
    draws <- as.matrix(simulate(fit, nsim = case$nsim, seed = 17395L))[rows, ]
    centered <- draws - rowMeans(draws)
    covariance_se <- outer(seq_along(rows), seq_along(rows), Vectorize(function(i, j) {
        sd(centered[i, ] * centered[j, ]) / sqrt(case$nsim)
    }))
    list(
        fixed = conditional(TRUE), mixed = conditional(FALSE),
        simulation = list(means = rowMeans(draws), zero = rowMeans(draws == 0),
                          covariance = cov(t(draws)), covariance_se = covariance_se)
    )
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    glmmTMB = as.character(packageVersion("glmmTMB")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
