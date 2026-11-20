args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
suppressPackageStartupMessages(library(rms))
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)

j <- 0:179
age <- (((j * 13) %% 41) - 20) / 4
z <- (((j * 7) %% 19) - 9) / 7
treated <- j %% 2
arm <- factor(ifelse(treated == 1L, "therapy", "control"), levels = c("control", "therapy"))
continuous <- .75 + .21 * age + .1 * age^2 + .48 * z - .8 * treated + .2 * sin(j)
binary <- as.integer(((j * 8917) %% 487) / 487 <
                    plogis(-.35 + .16 * age + .08 * age^2 + .3 * z + treated * (.3 - .2 * age)))
data <- data.frame(age, z, arm, continuous, binary)
dd <- datadist(data[c("age", "arm", "z")])
options(datadist = dd)
knots <- c(-4, -1, 1.5, 4)

evaluate <- function(case) {
    additive <- case$model == "ols"
    fit <- if (additive) {
        ols(continuous ~ rcs(age, knots) + arm + z, data = data, x = TRUE)
    } else {
        lrm(binary ~ rcs(age, knots) * arm + z, data = data, x = TRUE, maxit = 100L,
            eps = 1e-10)
    }
    scale <- if (additive) exp else plogis
    axes <- nomogram(
        fit, age = unlist(case$age), z = unlist(case$z), lp.at = unlist(case$lp),
        fun = list(outcome = scale), fun.lp.at = unlist(case$outcome_at),
        maxscale = case$max_points
    )
    info <- attr(axes, "info")
    settings <- names(axes)[!names(axes) %in% c("total.points", "lp", "outcome")]
    details <- lapply(axes[settings], function(axis) {
        predictor <- attr(axis, "info")$predictor
        by <- if (predictor == "age" && !additive) unique(as.character(axis$arm)) else NULL
        list(predictor = predictor, by = by, ticks = I(axis[[predictor]]),
             effects = I(axis$Xbeta), points = I(axis$points))
    })
    list(axes = unname(details), scale = info$sc, zero = info$Intercept,
         lp_points = I(axes$lp$x), outcome_label = I(axes$outcome$x.real),
         outcome_lp = I(axes$outcome$x / info$sc + info$Intercept), outcome_points = I(axes$outcome$x))
}
results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("rms")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null", null = "null"), "\n", sep = "")
