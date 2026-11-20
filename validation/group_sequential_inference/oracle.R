args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cases <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)

default <- function(x, value) if (is.null(x)) value else x

results <- lapply(cases, function(case) {
  p <- case$options
  ldof <- identical(p$spending, "ldof")
  spending <- if (ldof) gsDesign::sfLDOF else gsDesign::sfHSD
  parameter <- default(p$parameter, if (ldof) 1 else -4)
  if (identical(case$kind, "interval")) {
    fractions <- unlist(case$fractions)
    two_sided <- identical(p$alternative, "two-sided")
    design <- gsDesign::gsDesign(
      k = length(fractions), test.type = if (two_sided) 2 else 1,
      alpha = default(p$alpha, .025), timing = fractions, r = 80, tol = 1e-7,
      sfu = spending, sfupar = parameter
    )
    # Technical manual section 9.5 inverts each look's nominal upper-bound level.
    estimates <- unlist(case$estimates)
    errors <- unlist(case$standard_errors)
    critical <- qnorm(pnorm(-design$upper$bound), lower.tail = FALSE)
    return(list(
      id = case$id, lower = estimates - critical * errors,
      upper = if (two_sided) estimates + critical * errors
    ))
  }
  planned <- default(p$planned_final_information, 1)
  design <- gsDesign::gsDesign(k = 3L, test.type = 1L, n.I = planned * (1:3) / 3, sfu = spending, sfupar = parameter)
  information <- unlist(case$information)
  z <- unlist(case$z)
  times <- default(unlist(p$spending_times), information / planned)
  interval <- default(unlist(p$search_interval), c(1e-5, .9999))
  evaluate <- function(stage, values) {
    gsDesign::sequentialPValue(
      gsD = design, n.I = information[seq_len(stage)], Z = values,
      usTime = times[seq_len(stage)], interval = interval
    )
  }
  repeated <- vapply(seq_along(z), function(i) {
    # Keep the integration over previous looks, while testing the selected look.
    evaluate(i, c(rep(-20, i - 1L), z[i]))
  }, numeric(1))
  list(id = case$id, repeated = repeated, sequential = evaluate(length(z), z))
})

cat(jsonlite::toJSON(list(
  versions = list(R = as.character(getRversion()), gsDesign = as.character(utils::packageVersion("gsDesign"))),
  cases = results
), auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE), "\n", sep = "")
