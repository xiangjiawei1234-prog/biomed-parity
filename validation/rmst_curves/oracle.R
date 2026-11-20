args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript oracle.R cases.json")
input <- jsonlite::read_json(args[[1L]], simplifyVector = FALSE)
linked <- file.path(dirname(args[[1L]]), "..", "rmst", "cases.json")
datasets <- c(jsonlite::read_json(linked, simplifyVector = FALSE), input$additional_datasets)
names(datasets) <- vapply(datasets, function(data) data$id, character(1))

capture_plot <- function(fit) {
    # Substitute recording functions in a private copy of the installed plot
    # method so the snapshot contains the actual graphical input coordinates.
    plotter <- get("plot.rmst2", asNamespace("survRM2"))
    plot_env <- new.env(parent = environment(plotter))
    geometry <- list()
    current <- NULL
    plot_env$par <- function(...) NULL
    plot_env$plot <- function(x, main, mark.time, conf.int, ...) {
        stopifnot(isFALSE(mark.time), isFALSE(conf.int))
        current <<- sub("=", "", main, fixed = TRUE)
        geometry[[current]] <<- list(rmst_polygons = list(), rmtl_polygons = list())
    }
    plot_env$lines <- function(x, y, ...) {
        # Drop two surplus origin points with identical x and y.
        stopifnot(all(x[1:3] == 0), all(y[1:3] == 1))
        geometry[[current]]$steps <<- cbind(x, y)[-(1:2), , drop = FALSE]
    }
    plot_env$polygon <- function(x, y, col, ...) {
        if (is.finite(x[2L]) && x[1L] < x[2L]) {
            geometry[[current]][[col]] <<- c(geometry[[current]][[col]], list(cbind(x, y)))
        }
    }
    environment(plotter) <- plot_env
    plotter(fit, col.RMST = "rmst_polygons", col.RMTL = "rmtl_polygons")
    geometry
}

evaluate <- function(case) {
    data <- datasets[[case$dataset]]
    time <- unlist(data$time)
    status <- unlist(data$status)
    arm <- unlist(data$arm)
    alpha <- if (is.null(data$alpha)) 0.05 else data$alpha
    fitted <- suppressWarnings(survRM2::rmst2(time, status, arm, tau = data$tau, alpha = alpha))
    geometry <- capture_plot(fitted)
    curve <- function(group) {
        name <- paste0("arm", group)
        result <- fitted[[paste0("RMST.", name)]]
        fit <- result$fit
        band <- NULL
        if (!is.null(case$pointwise_confidence_level)) {
            km <- survival::survfit(survival::Surv(time, status) ~ 1,
                subset = arm == group, conf.int = case$pointwise_confidence_level, conf.type = "log")
            repeat_previous_and_current <- function(values) {
                c(1, as.vector(rbind(c(1, head(values, -1L)), values)))
            }
            band <- cbind(geometry[[name]]$steps[, 1L],
                repeat_previous_and_current(km$lower), repeat_previous_and_current(km$upper))
        }
        c(list(time = I(fit$time), n_risk = I(fit$n.risk), n_event = I(fit$n.event),
            n_censor = I(fit$n.censor), survival = I(fit$surv),
            log_survival_standard_error = I(fit$std.err),
            rmst = as.numeric(result$rmst[1L]), rmtl = as.numeric(result$rmtl[1L]),
            confidence_band = band), geometry[[name]])
    }
    list(tau = fitted$tau, alpha = alpha, arm0 = curve(0L), arm1 = curve(1L),
        pointwise_confidence_level = case$pointwise_confidence_level)
}

results <- lapply(input$cases, function(case) {
    tryCatch(evaluate(case), error = function(e) stop(case$dataset, ": ", conditionMessage(e), call. = FALSE))
})
names(results) <- vapply(input$cases, function(case) case$dataset, character(1))
versions <- list(R = as.character(getRversion()), survRM2 = as.character(packageVersion("survRM2")),
    survival = as.character(packageVersion("survival")))
cat(jsonlite::toJSON(list(versions = versions, results = results),
    auto_unbox = TRUE, digits = NA, na = "string", null = "null", pretty = TRUE), "\n", sep = "")
