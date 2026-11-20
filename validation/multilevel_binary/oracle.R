args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
unpack <- function(row) vapply(row, function(value) if (is.null(value)) NA_real_ else value, numeric(1))

moments <- function(draws) {
    centered <- draws - rowMeans(draws)
    covariance_se <- outer(seq_len(nrow(draws)), seq_len(nrow(draws)), Vectorize(
        function(i, j) sd(centered[i, ] * centered[j, ]) / sqrt(ncol(draws))
    ))
    list(mean = rowMeans(draws), covariance = cov(t(draws)), covariance_se = covariance_se)
}

evaluate <- function(case) {
    x <- do.call(rbind, lapply(case$predictors, unpack))
    colnames(x) <- c("cluster", paste0("x", seq_len(ncol(x) - 1L)))
    y <- unpack(case$y)
    observed <- !is.na(y)
    targets <- !observed
    columns <- cbind(response = y, as.data.frame(x))
    fixed <- colnames(x)[-1L]
    formula <- reformulate(c(fixed, "(1|cluster)"), response = "response")
    control <- lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(rhoend = 1e-11),
                                 tolPwrss = 1e-12)
    fit <- suppressMessages(lme4::glmer(formula, data = columns[observed, ],
                                        family = binomial, control = control))
    beta <- lme4::fixef(fit)
    covariance <- as.matrix(vcov(fit))
    eta <- lme4::ranef(fit, condVar = TRUE)$cluster
    sigma2 <- as.numeric(lme4::VarCorr(fit)$cluster)
    group_ids <- as.numeric(rownames(eta))
    modes <- eta[[1L]]
    v_conditional <- as.numeric(attr(eta, "postVar"))
    k <- length(group_ids)
    target_groups <- sort(unique(x[targets, "cluster"]))
    group_map <- match(target_groups, group_ids)
    target_map <- match(x[targets, "cluster"], target_groups)
    conditioning <- !is.na(group_map)
    future <- model.matrix(reformulate(fixed), columns[targets, ])
    beta_fluctuation <- t(chol(covariance))
    total <- sum(modes^2) + sigma2
    set.seed(25907L)
    distributions <- lapply(c("laplace", "eb", "marginal"), function(strategy) {
        samples <- replicate(case$n_draws, {
            beta_draw <- beta + beta_fluctuation %*% rnorm(length(beta))
            marginal <- total / rchisq(1, df = k + 1)
            means <- rep(0, length(target_groups))
            variances <- rep(marginal, length(target_groups))
            if (strategy != "marginal") means[conditioning] <- modes[group_map[conditioning]]
            if (strategy == "laplace") variances[conditioning] <- v_conditional[group_map[conditioning]]
            intercepts <- rnorm(length(target_groups), means, sqrt(variances))
            rbinom(sum(targets), 1L, plogis(future %*% beta_draw + intercepts[target_map]))
        })
        moments(samples)
    })
    names(distributions) <- c("laplace", "eb", "marginal")
    historical <- NULL
    if (case$historical_draws > 0L && packageVersion("mice") <= "3.19.0") {
        type <- setNames(unlist(case$types), colnames(x))
        samples <- replicate(case$historical_draws,
                             mice::mice.impute.2l.bin(y, observed, x, type, control = control))
        historical <- moments(samples)
    }
    list(coefficients = unname(beta), coefficient_covariance = unname(covariance),
         random_variance = sigma2, random_modes = modes, conditional_variances = v_conditional,
         groups = group_ids, log_likelihood = as.numeric(logLik(fit)),
         distributions = distributions, historical = historical)
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    mice_version = as.character(packageVersion("mice")), lme4_version = as.character(packageVersion("lme4")),
    results = results), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null", null = "null"),
    "\n", sep = "")
