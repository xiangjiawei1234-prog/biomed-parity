"""Conditional multinomial and proportional-odds models for categorical imputation."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp


def augment_categorical(design, labels, train, available, classes):
    """Give each class balanced pseudocases near the predictor center."""
    observed = design[train]
    predictors = design[available, 1:]
    n_predictors = predictors.shape[1]
    if not n_predictors or (~train).sum() == 1:
        return observed, labels, np.ones(len(labels))
    rows_per_predictor = 2 * len(classes)
    coordinates = np.repeat(np.arange(n_predictors), rows_per_predictor)
    signs = np.tile([0.5, -0.5], len(coordinates) // 2)
    rows = np.broadcast_to(predictors.mean(axis=0), (len(coordinates), n_predictors)).copy()
    spread = predictors.std(axis=0, ddof=1)
    rows[np.arange(len(rows)), coordinates] += signs * spread[coordinates]
    rows = np.clip(rows, predictors.min(axis=0), predictors.max(axis=0))
    pseudo_design = np.column_stack((np.ones(len(rows)), rows))
    return (
        np.vstack((observed, pseudo_design)),
        np.r_[labels, np.tile(np.repeat(classes, 2), n_predictors)],
        np.r_[np.ones(len(labels)), np.full(len(rows), observed.shape[1] / len(rows))],
    )


def _projected_gradient(gradient, parameters, bounds):
    projected = np.array(gradient, dtype=float, copy=True)
    if bounds is not None:
        for index, (lower, upper) in enumerate(bounds):
            if lower is not None and parameters[index] <= lower + 1e-12 and projected[index] > 0:
                projected[index] = 0.0
            if upper is not None and parameters[index] >= upper - 1e-12 and projected[index] < 0:
                projected[index] = 0.0
    return np.max(np.abs(projected)) if projected.size else 0.0


def _minimize_converged(objective, start, bounds, message):
    """Run L-BFGS-B and keep nearly converged solutions.

    BLAS/LAPACK builds differ in the last bits of the gradient, and the
    L-BFGS-B line search can then report abnormal termination on an
    otherwise usable optimum. Retry with looser tolerances and accept a
    finite candidate with a small projected gradient before declaring
    failure, so the same model is fitted on every platform.
    """
    attempts = (
        {"ftol": 1e-13, "gtol": 1e-8, "maxiter": 1000},
        {"ftol": 1e-11, "gtol": 1e-7, "maxiter": 2000},
        {"ftol": 1e-9, "gtol": 1e-6, "maxiter": 4000},
    )
    best = None
    for options in attempts:
        fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                       bounds=bounds, options=options)
        if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)):
            continue
        value, gradient = objective(fit.x)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            continue
        if best is None or value < best[0]:
            best = (value, fit)
        if fit.success or _projected_gradient(gradient, fit.x, bounds) <= 1e-6:
            return fit
    if best is not None:
        return best[1]
    raise ValueError(message)


def _multinomial_probabilities(x, y, weights, target, proportions):
    n_parameters = len(proportions) - 1
    rows = np.arange(len(y))

    def scores(predictors, coefficients):
        linear = predictors @ coefficients.reshape(x.shape[1], n_parameters)
        return np.column_stack((np.zeros(len(predictors)), linear))

    def objective(coefficients):
        linear = scores(x, coefficients)
        totals = logsumexp(linear, axis=1)
        residual = np.exp(linear - totals[:, None])
        residual[rows, y] -= 1
        gradient = x.T @ (weights[:, None] * residual[:, 1:])
        return weights @ (totals - linear[rows, y]), gradient.ravel()

    initial = np.zeros((x.shape[1], n_parameters))
    initial[0] = np.log(proportions[1:] / proportions[0])
    fit = _minimize_converged(
        objective, initial.ravel(), None, "multinomial regression did not converge",
    )
    linear = scores(target, fit.x)
    return np.exp(linear - logsumexp(linear, axis=1, keepdims=True))


def _thresholds(parameters):
    return np.r_[parameters[0], parameters[0] + np.cumsum(np.exp(parameters[1:]))]


def _interval_log_probability(lower, upper, negative_width):
    # log(F(upper) - F(lower)), including intervals with infinite endpoints.
    return (
        -np.logaddexp(0, -upper) - np.logaddexp(0, lower)
        + np.log(-np.expm1(negative_width))
    )


def _ordinal_probabilities(x, y, weights, target, proportions):
    n_predictors = x.shape[1]
    n_classes = len(proportions)
    upper_observed = y < n_classes - 1
    lower_observed = y > 0

    def objective(parameters):
        threshold_parameters = parameters[n_predictors:]
        cutoffs = _thresholds(threshold_parameters)
        bounds = np.r_[-np.inf, cutoffs, np.inf]
        eta = x @ parameters[:n_predictors]
        lower, upper = bounds[y] - eta, bounds[y + 1] - eta
        log_p = _interval_log_probability(lower, upper, bounds[y] - bounds[y + 1])
        upper_grad = weights * np.exp(
            -np.logaddexp(0, upper) - np.logaddexp(0, -upper) - log_p
        )
        lower_grad = weights * np.exp(
            -np.logaddexp(0, lower) - np.logaddexp(0, -lower) - log_p
        )
        cutoff_grad = np.bincount(y[upper_observed], weights=upper_grad[upper_observed],
                                  minlength=n_classes - 1)
        cutoff_grad -= np.bincount(y[lower_observed] - 1, weights=lower_grad[lower_observed],
                                   minlength=n_classes - 1)
        suffix = np.cumsum(cutoff_grad[::-1])[::-1]
        gradient = np.r_[x.T @ (lower_grad - upper_grad), suffix[0],
                         suffix[1:] * np.exp(threshold_parameters[1:])]
        return -(weights @ log_p), -gradient

    cutoffs = logit(np.cumsum(proportions)[:-1])
    start = np.r_[np.zeros(n_predictors), cutoffs[0], np.log(np.diff(cutoffs))]
    # Wide gap bounds keep trial steps ordered and numerically distinguishable.
    bounds = [(None, None)] * (n_predictors + 1) + [(-16, 16)] * (n_classes - 2)
    fit = _minimize_converged(
        objective, start, bounds, "proportional-odds regression did not converge",
    )
    cutoffs = _thresholds(fit.x[n_predictors:])
    bounds = np.r_[-np.inf, cutoffs, np.inf]
    adjusted = bounds - (target @ fit.x[:n_predictors])[:, None]
    return np.exp(_interval_log_probability(adjusted[:, :-1], adjusted[:, 1:], -np.diff(bounds)))


def categorical_probabilities(method, design, labels, train, target, available, levels):
    classes = levels[np.isin(levels, labels)]
    if len(classes) == 1:
        return classes, np.ones((target.sum(), 1))
    x, augmented, weights = augment_categorical(
        design, labels, train, available, classes,
    )
    order = classes.argsort()
    level_codes = order[np.searchsorted(classes, augmented, sorter=order)]
    counts = np.bincount(level_codes, weights=weights)
    proportions = counts / counts.sum()
    if design.shape[1] == 1:
        return classes, np.broadcast_to(proportions, (target.sum(), len(classes)))
    if method == "polr" and len(classes) > 2:
        try:
            probabilities = _ordinal_probabilities(
                x[:, 1:], level_codes, weights, design[target, 1:], proportions,
            )
            return classes, probabilities
        except ValueError:
            # Sparse ordinal fits use the same nominal fallback as mice.
            pass
    return classes, _multinomial_probabilities(
        x, level_codes, weights, design[target], proportions,
    )
