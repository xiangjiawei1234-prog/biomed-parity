#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survey))
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected cases.json and output JSON path")
message("R ", getRversion(), ", survey ", packageVersion("survey"))
raw <- fromJSON(args[[1]], simplifyVector = FALSE)
results <- list()

for (item in raw) {
  selected <- as.logical(unlist(item$phase2))
  frame <- data.frame(id = seq_along(selected), selected = selected)
  phase1_strata <- if (!is.null(item$phase1_strata)) {
    frame$level1 <- as.factor(unlist(item$phase1_strata))
    ~level1
  } else NULL
  phase2_strata <- if (!is.null(item$phase2_strata)) {
    frame$level2 <- as.factor(unlist(item$phase2_strata))
    ~level2
  } else NULL
  id2 <- if (!is.null(item$phase2_psu)) {
    frame$cluster <- as.factor(unlist(item$phase2_psu))
    ~cluster + id
  } else ~id
  fpc <- if (!is.null(item$phase1_population)) {
    if (!is.null(phase1_strata)) {
      counts <- unlist(item$phase1_population)
      frame$population <- unname(counts[as.character(frame$level1)])
    } else {
      frame$population <- item$phase1_population
    }
    list(~population, NULL)
  } else NULL

  if (!is.null(item$categories)) {
    frame$outcome <- factor(unlist(item$values), levels = unlist(item$categories))
    formula <- ~outcome
  } else {
    values <- do.call(rbind, lapply(item$values, unlist))
    names <- paste0("outcome", seq_len(ncol(values)))
    frame[names] <- as.data.frame(values)
    formula <- reformulate(names)
  }
  make_design <- function(data) {
    twophase(id = list(~id, id2), strata = list(phase1_strata, phase2_strata),
             fpc = fpc, subset = ~selected, data = data, method = "full")
  }
  design <- make_design(frame)

  summarize <- function(fit) {
    covariance <- vcov(fit)
    phases <- attr(covariance, "phases")
    list(
      estimate = unname(as.numeric(fit)), covariance = unname(covariance),
      phase1_covariance = unname(phases[["phase1"]]),
      phase2_covariance = unname(phases[["phase2"]]),
      standard_errors = unname(as.numeric(SE(fit)))
    )
  }
  summarize_design <- function(design) {
    list(
      mean = summarize(svymean(formula, design)),
      total = summarize(svytotal(formula, design)),
      weights = unname(as.numeric(weights(design))),
      phase1_probabilities = unname(as.numeric(design$phase1$sample$prob)),
      phase2_probabilities = unname(as.numeric(design$phase2$prob))
    )
  }
  if (isTRUE(item$audit_unequal_phase1)) {
    # Stable reorder leaves the observed phase-two row order unchanged.
    # survey 4.5 indexes phase-one sample sizes at the positions of those rows.
    reordered <- make_design(frame[c(which(selected), which(!selected)), , drop = FALSE])
    results[[item$name]] <- summarize_design(reordered)
    results[[item$name]]$original_row_order <- summarize_design(design)
  } else {
    results[[item$name]] <- summarize_design(design)
  }
}
write_json(results, args[[2]], digits = 17, pretty = TRUE)
