args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
spec <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
parse_vector <- function(values) unlist(lapply(values, function(value) {
    if (is.null(value)) NA else value
}), use.names = FALSE)
parse_matrix <- function(rows) do.call(rbind, lapply(rows, parse_vector))
keep <- as.data.frame(lapply(spec$keep, parse_vector))

compare <- function(case) {
    fields <- keep
    ordered <- unlist(case$ordered)
    for (name in names(case$factor_levels)) {
        fields[[name]] <- factor(fields[[name]], levels = unlist(case$factor_levels[[name]]),
                                 ordered = name %in% ordered)
    }
    long <- mstate::msprep(
        parse_matrix(spec$time), parse_matrix(spec$status),
        trans = mstate::trans.illdeath(), keep = fields
    )
    terms <- unlist(case$covariates)
    expanded <- mstate::expand.covs(
        long, terms, append = isTRUE(case$append), longnames = !identical(case$long_names, FALSE)
    )
    values <- if (isTRUE(case$append)) expanded[setdiff(names(expanded), names(long))] else expanded
    list(expanded = as.list(values), trans = long$trans)
}
results <- lapply(spec$cases, compare)
names(results) <- vapply(spec$cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("mstate")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), "\n", sep = "")
