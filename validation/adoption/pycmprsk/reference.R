#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Expected one input CSV path")
d <- read.csv(args[[1]])
library(cmprsk)
options(digits = 17)
message("R ", getRversion(), ", cmprsk ", packageVersion("cmprsk"))
results <- list()
sink(stderr())

record <- function(key, value) {
  results[[key]] <<- as.numeric(t(value))
}

for (rho in c(0, 1)) {
  ci <- cuminc(d$time, d$status, group = d$arm, strata = d$stratum, rho = rho, cencode = 9)
  key <- paste0("cuminc_", rho)
  record(paste0(key, ".tests"), ci$Tests)
  for (curve in sort(setdiff(names(ci), "Tests"))) {
    for (field in c("time", "est", "var")) {
      record(paste(key, curve, field, sep = "."), ci[[curve]][[field]])
    }
  }
  points <- timepoints(ci, c(0, 7, 20, 35))
  for (curve in rownames(points$est)) {
    record(paste(key, curve, "point_est", sep = "."), points$est[curve, ])
    record(paste(key, curve, "point_var", sep = "."), points$var[curve, ])
  }
}

x <- as.matrix(d[, c("x1", "x2")])
predict_x <- matrix(c(-0.5, 0, 0.5, 1), nrow = 2, byrow = TRUE)
tf <- function(times) cbind(log1p(times))
for (varying in c(FALSE, TRUE)) {
  if (varying) {
    fit <- crr(d$time, d$status, cov1 = x, cov2 = x[, 1], tf = tf,
               cengroup = d$arm, failcode = 2, cencode = 9)
    predicted <- predict(fit, cov1 = predict_x, cov2 = predict_x[, 1])
  } else {
    fit <- crr(d$time, d$status, cov1 = x, failcode = 2, cencode = 9)
    predicted <- predict(fit, cov1 = predict_x)
  }
  key <- paste0("crr_", if (varying) "time_varying" else "fixed")
  for (field in c("coef", "var", "res", "loglik", "loglik.null", "uftime", "converged", "n", "n.missing")) {
    record(paste(key, field, sep = "."), fit[[field]])
  }
  record(paste0(key, ".prediction"), predicted)
}

sink()
for (key in names(results)) {
  cat(key, paste(sprintf("%.17g", results[[key]]), collapse = ","), sep = "\t")
  cat("\n")
}
