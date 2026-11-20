args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript --vanilla oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]])

completed_data <- function(case) {
    i <- seq_len(case$n)
    base <- ((17 * i) %% 37 - 18) / 12
    predictor <- ((11 * i) %% 29 - 14) / 9
    extra <- ((13 * i) %% 31 - 15) / 11
    y <- if (case$family == "gaussian") {
        120 + 3 * base + 2.3 * predictor + 1.7 * extra + 4 * (sin(1.4 * i) + .7 * cos(.7 * i))
    } else {
        as.integer(((19 * i) %% 43) / 43 < plogis(-.45 + .63 * base + .52 * predictor + .3 * extra))
    }
    lapply(seq_len(case$m), function(draw) {
        data.frame(y = y,
            base = base + case$base_scale * (i %% 6 < 3) * ((5 * i * draw + 7 * draw) %% 19 - 9) / 7,
            predictor = predictor + case$draw_scale * (i %% 4 < 2) * ((7 * i * draw + 17 * draw + 3 * i) %% 23 - 11) / 7,
            extra = extra + case$draw_scale * (i %% 5 < 2) * ((3 * i * draw + 11 * draw + i) %% 17 - 8) / 6)
    })
}

fit_model <- function(data, formula, family) {
    if (family == "gaussian") do.call(stats::lm, list(formula = formula, data = data))
    else do.call(stats::glm, list(formula = formula, data = data, family = "binomial"))
}

pack_test <- function(values) {
    as.list(setNames(as.numeric(values), c("statistic", "df1", "df2", "pvalue", "relative_increase")))
}

likelihood <- function(fitted) vapply(fitted, function(fit) as.numeric(logLik(fit)), numeric(1))
mitml_restricted <- getFromNamespace(".evaluateUserLogLik", "mitml")

results <- lapply(cases, function(case) {
    completed <- completed_data(case)
    full_formula <- if (case$added == 1) y ~ base + predictor else y ~ base + predictor + extra
    reduced_formula <- y ~ base
    full <- lapply(completed, fit_model, formula = full_formula, family = case$family)
    reduced <- lapply(completed, fit_model, formula = reduced_formula, family = case$family)
    d3 <- mice::D3(full, reduced)
    mice_d3_inputs <- list(
        full_loglik = -d3$deviances$dev1.M / 2, reduced_loglik = -d3$deviances$dev0.M / 2,
        restricted_full_loglik = -d3$deviances$dev1.L / 2,
        restricted_reduced_loglik = -d3$deviances$dev0.L / 2, df = case$added
    )
    stack <- do.call(rbind, completed)
    stacked_full <- logLik(fit_model(stack, full_formula, case$family))
    stacked_reduced <- logLik(fit_model(stack, reduced_formula, case$family))
    d4_inputs <- list(full_loglik = likelihood(full), reduced_loglik = likelihood(reduced),
        stacked_full_loglik = as.numeric(stacked_full), stacked_reduced_loglik = as.numeric(stacked_reduced),
        df = case$added, full_model_df = attr(stacked_full, "df"))
    d4 <- lapply(setNames(c("default", "positive", "robust"), c("default", "positive", "robust")), function(ariv) {
        pack_test(mitml::testModels(full, reduced, method = "D4", ariv = ariv, data = completed)$test[1L, ])
    })
    result <- list(mice_d3_inputs = mice_d3_inputs, mice_d3 = pack_test(d3$result), d4_inputs = d4_inputs, d4 = d4)
    if (case$family == "gaussian") {
        result$mitml_d3_inputs <- list(full_loglik = likelihood(full), reduced_loglik = likelihood(reduced),
            restricted_full_loglik = mitml_restricted(full)$LL.pooled,
            restricted_reduced_loglik = mitml_restricted(reduced)$LL.pooled, df = case$added)
        result$mitml_d3 <- lapply(setNames(c("default", "positive"), c("default", "positive")), function(ariv) {
            pack_test(mitml::testModels(full, reduced, method = "D3", ariv = ariv)$test[1L, ])
        })
    }
    result
})
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(
    r_version = as.character(getRversion()),
    mice_version = as.character(utils::packageVersion("mice")),
    mitml_version = as.character(utils::packageVersion("mitml")),
    mice_d3_exact = exists("d3.restricted", envir = asNamespace("mice"), inherits = FALSE),
    results = results
), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
