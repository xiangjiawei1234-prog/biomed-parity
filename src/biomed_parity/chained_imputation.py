"""Fully conditional imputation for numeric and binary data."""

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import qr, solve_triangular
from scipy.special import expit


PassiveImputation = Callable[[NDArray[np.float64]], ArrayLike]
ImputationMethod = str | PassiveImputation | None
_METHODS = {"pmm", "norm", "norm.predict", "mean", "logreg"}


@dataclass(frozen=True)
class ChainedImputations:
    """Completed data (chain × row × column) and per-sweep diagnostics."""

    completed: NDArray[np.float64]
    chain_mean: NDArray[np.float64]
    chain_variance: NDArray[np.float64]
    where: NDArray[np.bool_]


def _numeric_matrix(data, name):
    matrix = np.asarray(data)
    if (
        matrix.ndim != 2 or not all(matrix.shape) or matrix.dtype.kind not in "buif"
        or np.isinf(matrix).any()
    ):
        raise ValueError(f"{name} must be a nonempty numeric matrix of finite values or NaN")
    return matrix.astype(float)


def _mask(values, shape, name):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return array.copy()


def _positive_integer(value, name, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _methods(methods, n_columns):
    if methods is None:
        return ("pmm",) * n_columns
    if len(methods) != n_columns:
        raise ValueError("methods must have one entry per column")
    normalized = tuple(None if isinstance(m, str) and not m else m for m in methods)
    if not all(m is None or callable(m) or isinstance(m, str) and m in _METHODS
               for m in normalized):
        raise ValueError("methods must select pmm, norm, norm.predict, mean, logreg, or a callable")
    return normalized


def _design(x, train):
    """Center predictors and retain a full-rank design of estimable columns."""
    observed = x[train]
    center = observed.mean(axis=0)
    scale = observed.std(axis=0)
    keep = scale > 0
    standardized = (x[:, keep] - center[keep]) / scale[keep]
    if standardized.shape[1]:
        _, triangular, pivot = qr(standardized[train], mode="economic", pivoting=True)
        diagonal = abs(np.diag(triangular))
        rank = np.count_nonzero(diagonal > diagonal.max() * 1e-10)
        standardized = standardized[:, np.sort(pivot[:rank])]
    return np.column_stack((np.ones(len(x)), standardized))


def _normal_fit(design, outcome):
    orthogonal, triangular = qr(design, mode="economic")
    coef = solve_triangular(triangular, orthogonal.T @ outcome)
    residual = outcome - design @ coef
    if np.isclose(residual @ residual, 0, atol=np.finfo(float).eps**2 * (outcome @ outcome),
                  rtol=0):
        residual = np.zeros_like(residual)
    return coef, triangular, residual


def _pmm_donors(observed, predicted_observed, predicted_target, donors, rng):
    pool_size = min(len(observed), donors)
    if pool_size == len(observed):
        return rng.choice(observed, size=len(predicted_target))
    order = np.argsort(predicted_observed, kind="stable")
    scores = predicted_observed[order]
    values = observed[order]
    draws = np.empty(len(predicted_target))
    positions = np.searchsorted(scores, predicted_target)
    for row, (target, middle) in enumerate(zip(predicted_target, positions, strict=True)):
        lower, upper = max(0, middle - pool_size), middle + pool_size
        distances = abs(scores[lower:upper] - target)
        threshold = np.partition(distances, pool_size - 1)[pool_size - 1]
        # Distances grow monotonically away from the insertion point.
        left, right = scores[:middle][::-1], scores[middle:]
        left_count = bisect_left(left, threshold, key=lambda value: target - value)
        right_count = bisect_left(right, threshold, key=lambda value: value - target)
        closer = left_count + right_count
        ticket = rng.integers(pool_size)
        if ticket < closer:
            donor = middle - ticket - 1 if ticket < left_count else middle + ticket - left_count
        else:
            left_ties = bisect_right(left, threshold, key=lambda value: target - value) - left_count
            right_ties = bisect_right(right, threshold, key=lambda value: value - target) - right_count
            index = rng.integers(left_ties + right_ties)
            if index < left_ties:
                donor = middle - left_count - index - 1
            else:
                donor = middle + right_count + index - left_ties
        draws[row] = values[donor]
    return draws


def _continuous_draw(method, design, y, train, target, donors, match_type, rng):
    observed_design = design[train]
    coef, triangular, residual = _normal_fit(observed_design, y)
    target_design = design[target]
    if method == "norm.predict":
        return target_design @ coef
    if method == "pmm" and match_type == 0:
        return _pmm_donors(y, observed_design @ coef, target_design @ coef, donors, rng)

    df = max(1, len(y) - len(coef))
    sigma = np.linalg.norm(residual) / np.sqrt(rng.chisquare(df))
    draw = coef + sigma * solve_triangular(triangular, rng.standard_normal(len(coef)))
    if method == "norm":
        return target_design @ draw + sigma * rng.standard_normal(target.sum())

    donor_coef = draw if match_type == 2 else coef
    return _pmm_donors(y, observed_design @ donor_coef, target_design @ draw, donors, rng)


def _augment_logistic(design, y, train, unignored):
    observed = design[train]
    predictors = design[unignored, 1:]
    n_predictors = predictors.shape[1]
    if not n_predictors or (~train).sum() == 1:
        return observed, y, np.ones(len(y))
    center = predictors.mean(axis=0)
    spread = predictors.std(axis=0, ddof=1)
    low, high = predictors.min(axis=0), predictors.max(axis=0)
    # Pair each coordinate's two perturbed values with both response classes.
    rows = np.repeat(center[None, :], 4 * n_predictors, axis=0)
    coordinates = np.repeat(np.arange(n_predictors), 4)
    signs = np.tile([0.5, -0.5, 0.5, -0.5], n_predictors)
    rows[np.arange(len(rows)), coordinates] += signs * spread[coordinates]
    pseudo_design = np.column_stack((np.ones(len(rows)), np.clip(rows, low, high)))
    pseudo_labels = np.tile([0.0, 0.0, 1.0, 1.0], n_predictors)
    pseudo_weights = np.full(len(rows), observed.shape[1] / len(rows))
    return (
        np.vstack((observed, pseudo_design)),
        np.r_[y, pseudo_labels],
        np.r_[np.ones(len(y)), pseudo_weights],
    )


def _logistic_draw(design, y, train, target, unignored, rng):
    if len(np.unique(y)) != 2 or not np.all((y == 0) | (y == 1)):
        raise ValueError("logreg requires both observed binary classes coded 0 and 1")
    x, response, case_weights = _augment_logistic(design, y, train, unignored)
    signed = 2 * response - 1
    coefficients = np.zeros(x.shape[1])
    for _ in range(80):
        eta = x @ coefficients
        variance = case_weights * expit(eta) * expit(-eta)
        information = x.T @ (variance[:, None] * x)
        gradient = x.T @ (case_weights * signed * expit(-signed * eta))
        step = np.linalg.solve(information, gradient)
        if np.max(abs(step)) < 1e-8:
            factor = np.linalg.cholesky(information)
            draw = coefficients + solve_triangular(
                factor.T, rng.standard_normal(len(coefficients)), lower=False,
            )
            probabilities = expit(design[target] @ draw)
            return rng.random(len(probabilities)) < probabilities
        loss = case_weights @ np.logaddexp(0, -signed * eta)
        for discount in 0.5 ** np.arange(30):
            candidate = coefficients + discount * step
            candidate_loss = case_weights @ np.logaddexp(0, -signed * (x @ candidate))
            if candidate_loss <= loss + 1e-12:
                coefficients = candidate
                break
        else:
            break
    raise ValueError("logreg could not fit a finite logistic model")


def _passive_values(function, data, target):
    readonly = data.view()
    readonly.flags.writeable = False
    values = np.asarray(function(readonly))
    if values.shape != (len(data),) or values.dtype.kind not in "buif":
        raise ValueError("passive methods must return one numeric value per row")
    imputed = values[target].astype(float)
    if not np.isfinite(imputed).all():
        raise ValueError("passive method returned missing or infinite values at target rows")
    return imputed


def _initialize(raw, methods, requested, visit, ignore, initial, rng):
    data = raw.copy()
    latest = raw.copy()
    missing = np.isnan(raw)
    for column in dict.fromkeys(visit):
        targets = requested[:, column]
        if initial is not None:
            values = initial[targets, column]
            if not np.isfinite(values).all():
                raise ValueError("initial must supply finite values for requested cells")
        elif callable(methods[column]):
            continue
        else:
            donors = raw[~ignore & ~missing[:, column], column]
            if not len(donors):
                raise ValueError(f"column {column} has no eligible observed initialization values")
            values = rng.choice(donors, size=int(targets.sum()))
        latest[targets, column] = values
        data[missing[:, column] & targets, column] = latest[missing[:, column] & targets, column]
    if initial is None:
        for column in dict.fromkeys(visit):
            if callable(methods[column]):
                targets = requested[:, column]
                latest[targets, column] = _passive_values(methods[column], data, targets)
                missing_targets = targets & missing[:, column]
                data[missing_targets, column] = latest[missing_targets, column]
    return data, latest


def _visit_column(raw, current, column, chosen, method, target, ignore, donors, match_type, rng):
    if callable(method):
        return _passive_values(method, current, target)
    x = current[:, chosen]
    available = np.isfinite(x).all(axis=1)
    if np.any(target & ~available):
        raise ValueError("a requested row has an incomplete predictor")
    train = ~np.isnan(raw[:, column]) & available & ~ignore
    y = raw[train, column]
    if not len(y):
        raise ValueError("no eligible observed outcomes")
    if method == "mean":
        return np.full(target.sum(), y.mean())
    design = _design(x, train)
    if method == "logreg":
        return _logistic_draw(design, y, train, target, available & ~ignore, rng)
    return _continuous_draw(method, design, y, train, target, donors, match_type, rng)


def mice_impute(
    data: ArrayLike,
    *,
    methods: Sequence[ImputationMethod] | None = None,
    predictor_matrix: ArrayLike | None = None,
    visit_sequence: Sequence[int] | None = None,
    where: ArrayLike | None = None,
    ignore: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    m: int = 5,
    iterations: int = 5,
    donors: int = 5,
    match_type: int = 1,
    seed: int | None = None,
) -> ChainedImputations:
    """Impute numeric data through fully conditional sweeps over its columns.

    NaN marks missing values. Methods default to predictive mean matching.
    Specify ``"logreg"`` for columns coded as binary 0/1, and a callable for
    deterministic passive variables. A callable receives the current data and
    returns one value per row. Predictor rows index outcomes and columns index
    input variables. Visit indices are zero-based and can be repeated.
    """
    raw = _numeric_matrix(data, "data")
    n_rows, n_columns = raw.shape
    methods = _methods(methods, n_columns)
    m = _positive_integer(m, "m")
    iterations = _positive_integer(iterations, "iterations", minimum=0)
    donors = _positive_integer(donors, "donors")
    match_type = _positive_integer(match_type, "match_type", minimum=0)
    if match_type > 2:
        raise ValueError("match_type must be 0, 1, or 2")

    observed = ~np.isnan(raw)
    requested = ~observed if where is None else _mask(where, raw.shape, "where")
    excluded = np.zeros(n_rows, dtype=bool) if ignore is None else _mask(ignore, (n_rows,), "ignore")
    if predictor_matrix is None:
        predictors = ~np.eye(n_columns, dtype=bool)
    else:
        indicators = np.asarray(predictor_matrix)
        if indicators.shape != (n_columns, n_columns) or not np.all(
            (indicators == 0) | (indicators == 1)
        ):
            raise ValueError("predictor_matrix must be a square matrix of 0/1 indicators")
        if np.any(np.diag(indicators)):
            raise ValueError("predictor_matrix cannot select a column as its own predictor")
        predictors = indicators.astype(bool)

    active = {j for j, method in enumerate(methods) if method is not None and requested[:, j].any()}
    if visit_sequence is None:
        visit = sorted(active)
    else:
        supplied = tuple(_positive_integer(j, "visit_sequence index", 0) for j in visit_sequence)
        if any(j >= n_columns for j in supplied) or not active.issubset(supplied):
            raise ValueError("visit_sequence must visit every requested active column")
        visit = [j for j in supplied if j in active]

    start = None if initial is None else _numeric_matrix(initial, "initial")
    if start is not None and start.shape != raw.shape:
        raise ValueError("initial must have the same shape as data")
    means = np.full((m, iterations, n_columns), np.nan)
    variances = np.full_like(means, np.nan)
    completed = np.empty((m, n_rows, n_columns))

    for chain, stream in enumerate(np.random.SeedSequence(seed).spawn(m)):
        rng = np.random.default_rng(stream)
        current, latest = _initialize(raw, methods, requested, visit, excluded, start, rng)
        for sweep in range(iterations):
            for column in visit:
                target = requested[:, column]
                try:
                    values = _visit_column(
                        raw, current, column, predictors[column], methods[column], target,
                        excluded, donors, match_type, rng,
                    )
                    if not np.isfinite(values).all():
                        raise ValueError("imputed values are not finite")
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    raise ValueError(
                        f"chain {chain + 1}, sweep {sweep + 1}, column {column}: {error}"
                    ) from error
                latest[target, column] = values
                in_chain = target & ~observed[:, column]
                current[in_chain, column] = latest[in_chain, column]
                means[chain, sweep, column] = values.mean()
                if len(values) > 1:
                    variances[chain, sweep, column] = values.var(ddof=1)
        completed[chain] = latest
    effective = requested & np.isin(np.arange(n_columns), list(active))
    return ChainedImputations(completed, means, variances, effective)
