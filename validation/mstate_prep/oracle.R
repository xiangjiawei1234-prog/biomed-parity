suppressPackageStartupMessages(library(mstate))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
cases <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
fallback <- function(value, default) if (is.null(value)) default else value
parse_vector <- function(cells) {
    unlist(lapply(cells, function(x) if (is.null(x)) NA else x), use.names = FALSE)
}
parse_matrix <- function(rows) do.call(rbind, lapply(rows, parse_vector))

evaluate <- function(case) {
    a <- case$arguments
    columns <- if (!is.null(a$data)) as.data.frame(lapply(a$data, parse_vector)) else NULL
    time <- if (is.null(columns)) parse_matrix(a$time) else parse_vector(a$time)
    status <- if (is.null(columns)) parse_matrix(a$status) else parse_vector(a$status)
    size <- if (is.null(columns)) nrow(time) else nrow(columns)
    if (is.null(a$transition_matrix)) {
        trans <- matrix(NA_integer_, ncol(time), ncol(time))
        for (i in seq_along(a$transitions)) {
            endpoints <- parse_vector(a$transitions[[i]]) + 1
            trans[endpoints[1], endpoints[2]] <- i
        }
    } else {
        trans <- parse_matrix(a$transition_matrix)
    }
    id <- if (is.null(a$id)) seq_len(size) else parse_vector(a$id)
    if (!is.null(a$id_order)) {
        order <- parse_vector(a$id_order)
        if (is.null(columns)) id <- factor(id, levels = order)
        else columns[[id]] <- factor(columns[[id]], levels = order)
    }
    keep <- a$keep
    if (!is.null(keep)) {
        keep <- if (!is.null(names(keep))) as.data.frame(lapply(keep, parse_vector)) else parse_vector(keep)
    }
    opt <- list(
        time = time, status = status, trans = trans, start = list(
            state = rep_len(parse_vector(fallback(a$start_state, 0)) + 1, size),
            time = rep_len(parse_vector(fallback(a$start_time, 0)), size)
        ), id = id
    )
    if (!is.null(columns)) opt$data <- columns
    if (!is.null(keep)) opt$keep <- keep
    warnings <- character()
    long <- tryCatch(withCallingHandlers(do.call(msprep, opt), warning = function(condition) {
        warnings <<- c(warnings, conditionMessage(condition))
        invokeRestart("muffleWarning")
    }), error = function(condition) condition)
    if (inherits(long, "error")) return(list(error = conditionMessage(long), warnings = I(warnings)))
    id_name <- if (!is.null(columns) && length(id) == 1) id else "id"
    long$id <- long[[id_name]]
    if (is.factor(long$id)) long$id <- as.character(long$id)
    regular <- c("id", "from", "to", "trans", "Tstart", "Tstop", "time", "status")
    additional <- setdiff(names(long), c(regular, id_name))
    long$from <- long$from - 1L
    long$to <- long$to - 1L
    long$trans <- long$trans - 1L
    values <- lapply(c(regular, additional), function(name) I(long[[name]]))
    names(values) <- c(regular, additional)
    list(rows = values, warnings = I(warnings))
}

results <- lapply(cases, evaluate)
names(results) <- vapply(cases, function(case) case$id, character(1))
cat(jsonlite::toJSON(list(r_version = as.character(getRversion()),
    package_version = as.character(packageVersion("mstate")), results = results),
    auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "string"), "\n", sep = "")
