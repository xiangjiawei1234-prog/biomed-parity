"""Gaussian two-level conditional models for chained imputation."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular


TwoLevelMethod = Literal["2l.pan", "2l.norm"]


@dataclass(frozen=True)
class TwoLevelOptions:
    """Inner sampler controls for multilevel columns in a chained imputation."""

    iterations: int | None = None
    intercept: bool = True
    groupcenter_slope: bool = False
    ridge: float = 1e-4


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _iterations(method, options):
    if method not in ("2l.pan", "2l.norm"):
        raise ValueError("method must be '2l.pan' or '2l.norm'")
    if not isinstance(options, TwoLevelOptions):
        raise ValueError("options must be a TwoLevelOptions")
    if not isinstance(options.intercept, (bool, np.bool_)) or not isinstance(
        options.groupcenter_slope, (bool, np.bool_)
    ):
        raise ValueError("intercept and groupcenter_slope must be boolean")
    if not isinstance(options.ridge, Real) or not np.isfinite(options.ridge) or options.ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if method == "2l.norm" and options.groupcenter_slope:
        raise ValueError("groupcenter_slope applies to type 3 and 4 predictors in '2l.pan'")
    if options.iterations is None:
        return 500 if method == "2l.pan" else 100
    return _positive_integer(options.iterations, "iterations")


def _group_crossproducts(groups, random, response, n_groups):
    n_effects = random.shape[1]
    cross = np.zeros((n_groups, n_effects, n_effects))
    moment = np.zeros((n_groups, n_effects))
    np.add.at(cross, groups, random[:, :, None] * random[:, None, :])
    np.add.at(moment, groups, random * response[:, None])
    return cross, moment


def _group_sum(groups, values, n_groups):
    return np.bincount(groups, weights=values, minlength=n_groups)


def _precision_normal(precision, right, rng):
    factor = np.linalg.cholesky(precision)
    mean = np.linalg.solve(factor.swapaxes(-1, -2), np.linalg.solve(factor, right[..., None]))
    offset = np.linalg.solve(factor.swapaxes(-1, -2), rng.standard_normal(mean.shape))
    return (mean + offset)[..., 0]


def _inverse_wishart(scale, df, rng):
    dimension = len(scale)
    if dimension == 1:
        return scale / rng.chisquare(df)
    location = np.linalg.cholesky(scale)
    triangular = np.zeros_like(scale)
    lower = np.tril_indices(dimension, k=-1)
    triangular[lower] = rng.standard_normal(len(lower[0]))
    np.fill_diagonal(triangular, np.sqrt(rng.chisquare(df - np.arange(dimension))))
    root = solve_triangular(triangular, location.T, lower=True).T
    return root @ root.T


def _ridge(matrix, amount):
    if matrix.shape[-1] == 1 or not amount:
        return matrix
    return matrix + amount * np.eye(matrix.shape[-1]) * np.diagonal(
        matrix, axis1=-2, axis2=-1,
    )[..., None, :]


def _design(x, types, groups, train, n_groups, method, options):
    selected = types > 0
    codes = types[selected]
    values = x[:, selected].copy()
    # Codes 3/4 append a cluster mean as a fixed effect. With centering,
    # the original column becomes its within-cluster deviation.
    aggregated = np.flatnonzero((codes == 3) | (codes == 4))
    means = []
    for column in aggregated:
        valid = np.isfinite(values[:, column])
        count = np.bincount(groups[valid], minlength=n_groups)
        group_mean = np.divide(
            _group_sum(groups[valid], values[valid, column], n_groups),
            count,
            out=np.full(n_groups, np.nan),
            where=count > 0,
        )
        mean = group_mean[groups]
        means.append(mean)
        if options.groupcenter_slope:
            values[:, column] -= mean

    z = values[:, np.isin(codes, (2, 4))]
    if options.intercept:
        z = np.column_stack((np.ones(len(x)), z))
    if not z.shape[1]:
        raise ValueError("the two-level model needs a random intercept or at least one random slope")
    if method == "2l.norm":
        return z, None

    fixed = np.column_stack((values, *means))
    # The fixed effects have a flat prior, so reparameterize fixed columns
    # for numerical conditioning while leaving the random-effect prior scale.
    center = fixed[train].mean(axis=0) if options.intercept else np.zeros(fixed.shape[1])
    scale = np.linalg.norm(fixed[train] - center, axis=0)
    if np.any(~np.isfinite(scale) | (scale == 0)):
        raise ValueError("fixed effects in the two-level model must have full observed rank")
    fixed = (fixed - center) / scale
    if options.intercept:
        fixed = np.column_stack((np.ones(len(x)), fixed))
    return z, fixed


@dataclass(frozen=True)
class _ConditionalData:
    y: np.ndarray
    train_group: np.ndarray
    target_group: np.ndarray
    random: np.ndarray
    target_random: np.ndarray
    fixed: np.ndarray | None
    target_fixed: np.ndarray | None
    count: np.ndarray
    cross: np.ndarray
    response_moment: np.ndarray


def _prepare(y, predictors, types, train, target, method, options):
    group_labels = predictors[:, types == -2]
    if group_labels.shape[1] != 1 or not np.isfinite(group_labels).all():
        raise ValueError("exactly one complete cluster column must be coded -2")
    _, group = np.unique(group_labels[:, 0], return_inverse=True)
    n_groups = int(group.max()) + 1
    if n_groups < 2:
        raise ValueError("the two-level model requires at least two clusters")
    if method == "2l.norm" and np.any(np.isin(types, (1, 3, 4))):
        raise ValueError("'2l.norm' supports random slopes coded 2")
    selected = predictors[:, types > 0]
    complete = np.isfinite(selected).all(axis=1)
    if np.any(target & ~complete):
        raise ValueError("a requested row has an incomplete predictor")
    train = train & complete
    if not train.any():
        raise ValueError("no eligible observed outcomes")
    count = np.bincount(group[train], minlength=n_groups)
    random, fixed = _design(predictors, types, group, train, n_groups, method, options)
    if method == "2l.norm":
        if not np.all(count):
            raise ValueError("'2l.norm' requires an observed outcome in every cluster")
        if n_groups <= 2 * random.shape[1]:
            raise ValueError("'2l.norm' needs more clusters than twice the random effect dimension")
    elif np.linalg.matrix_rank(fixed[train]) != fixed.shape[1]:
        raise ValueError("fixed effects in the two-level model must have full observed rank")

    train_random = random[train]
    train_group = group[train]
    cross, moment = _group_crossproducts(train_group, train_random, y[train], n_groups)
    return _ConditionalData(
        y[train], train_group, group[target], train_random, random[target],
        None if fixed is None else fixed[train], None if fixed is None else fixed[target],
        count, cross, moment,
    )


@dataclass(frozen=True)
class _PanMoments:
    fixed_factor: tuple
    fixed_response: np.ndarray
    fixed_random: np.ndarray
    standard_factor: np.ndarray

    @classmethod
    def from_data(cls, data):
        design = data.fixed
        dimension = design.shape[1]
        cross = np.zeros((len(data.count), dimension, data.random.shape[1]))
        np.add.at(cross, data.train_group, design[:, :, None] * data.random[:, None, :])
        factor = cho_factor(design.T @ design, lower=True)
        covariance_draw = solve_triangular(factor[0].T, np.eye(dimension), lower=False)
        return cls(factor, design.T @ data.y, cross, covariance_draw)


def _pan_draw(data, iterations, rng, moments):
    n_groups, q, _ = data.cross.shape
    identity = np.eye(q)
    beta = cho_solve(moments.fixed_factor, moments.fixed_response)
    residual = data.y - data.fixed @ beta
    variance = (1 + residual @ residual) / (1 + len(data.y))
    psi = identity
    # This Gibbs sampler conditions on observed rows, integrating out missing
    # responses during the inner sweeps. Its equilibrium posterior matches
    # the data-augmentation sampler with the same priors.
    for _ in range(iterations):
        precision = np.linalg.solve(psi, identity)
        beta_moment = np.einsum("gpq,p->gq", moments.fixed_random, beta)
        random = _precision_normal(
            precision + data.cross / variance,
            (data.response_moment - beta_moment) / variance,
            rng,
        )
        beta_mean = cho_solve(
            moments.fixed_factor, moments.fixed_response
            - np.einsum("gpq,gq->p", moments.fixed_random, random),
        )
        beta = beta_mean + np.sqrt(variance) * (moments.standard_factor @
                                                rng.standard_normal(len(beta)))
        residual = data.y - data.fixed @ beta - np.einsum(
            "ij,ij->i", data.random, random[data.train_group],
        )
        variance = (1 + residual @ residual) / rng.chisquare(1 + len(residual))
        psi = _inverse_wishart(identity + random.T @ random, n_groups + q, rng)

    return (data.target_fixed @ beta + np.einsum(
        "ij,ij->i", data.target_random, random[data.target_group],
    ) + np.sqrt(variance) * rng.standard_normal(len(data.target_group)))


def _norm_draw(data, iterations, rng, ridge):
    n_groups, q, _ = data.cross.shape
    identity = np.eye(q)
    mean = np.zeros(q)
    covariance = identity
    precision = np.ones(n_groups)
    scale, shape = 1.0, 1.0
    for _ in range(iterations):
        inverse_covariance = np.linalg.solve(covariance, identity)
        design_precision = _ridge(
            data.cross * precision[:, None, None] + inverse_covariance, ridge,
        )
        right = data.response_moment * precision[:, None] + inverse_covariance @ mean
        effect_covariance = np.linalg.solve(
            design_precision, np.broadcast_to(identity, design_precision.shape),
        )
        effect = (effect_covariance @ right[..., None])[..., 0]
        effect += (np.linalg.cholesky(_ridge(effect_covariance, ridge)) @
                   rng.standard_normal((n_groups, q, 1)))[..., 0]
        mean_covariance = np.linalg.solve(_ridge(inverse_covariance, ridge), identity)
        mean = effect.mean(axis=0) + np.linalg.cholesky(mean_covariance / n_groups) @ (
            rng.standard_normal(q)
        )
        centered = effect - mean
        covariance = _inverse_wishart(_ridge(centered.T @ centered, ridge), n_groups - q - 1, rng)
        residual = data.y - np.einsum("ij,ij->i", data.random, effect[data.train_group])
        squares = _group_sum(data.train_group, residual**2, n_groups)
        precision = rng.gamma((data.count + 1 / shape) / 2, 2 / (squares + scale / shape))
        log_mean_precision = np.log(precision).mean()
        average_precision = precision.mean()
        scale = rng.gamma(n_groups / (2 * shape) + 1, 2 * shape / (n_groups * average_precision))
        ratio = scale * average_precision
        gamma_rate = (n_groups / 2) * (
            ratio - 1 - np.log(ratio) + np.log(average_precision) - log_mean_precision
        )
        shape = max(gamma_rate, np.finfo(float).tiny) / rng.gamma(n_groups / 2 - 1)

    return (np.einsum("ij,ij->i", data.target_random, effect[data.target_group])
            + rng.standard_normal(len(data.target_group)) / np.sqrt(precision[data.target_group]))


def _inputs(y, predictors, types, observed, where):
    y = np.asarray(y)
    x = np.asarray(predictors)
    codes = np.asarray(types)
    if x.ndim != 2 or not x.shape[0] or x.dtype.kind not in "buif" or np.isinf(x).any():
        raise ValueError("predictors must be a nonempty numeric design with no infinities")
    if y.shape != (len(x),) or y.dtype.kind not in "buif" or np.isinf(y).any():
        raise ValueError("y must be a numeric outcome per row with no infinities")
    if (codes.shape != (x.shape[1],) or codes.dtype.kind not in "buif"
            or not np.isin(codes, (-2, 0, 1, 2, 3, 4)).all()):
        raise ValueError("types must code each predictor as -2, 0, 1, 2, 3 or 4")
    train = np.isfinite(y) if observed is None else np.asarray(observed)
    if train.shape != y.shape or train.dtype.kind != "b" or not np.isfinite(y[train]).all():
        raise ValueError("observed must identify finite outcomes with a boolean vector")
    target = ~train if where is None else np.asarray(where)
    if target.shape != y.shape or target.dtype.kind != "b":
        raise ValueError("where must be a boolean vector matching y")
    return y.astype(float), x.astype(float), codes, train, target


def impute_two_level_normal(
    y: ArrayLike,
    predictors: ArrayLike,
    types: ArrayLike,
    *,
    method: TwoLevelMethod = "2l.pan",
    observed: ArrayLike | None = None,
    where: ArrayLike | None = None,
    options: TwoLevelOptions = TwoLevelOptions(),
    m: int = 5,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Gaussian conditional imputations from independent Gibbs runs.

    Type -2 identifies the cluster, 1 a fixed effect, and 2 a random slope
    with a population mean. With 2l.pan, 3/4 also add a fixed cluster mean.
    2l.norm models heterogeneous residual variances and uses type 2 slopes.
    Returns draws × requested rows, in original row order.
    """
    m = _positive_integer(m, "m")
    iterations = _iterations(method, options)
    y, x, codes, train, target = _inputs(y, predictors, types, observed, where)
    if not target.any():
        return np.empty((m, 0))
    data = _prepare(y, x, codes, train, target, method, options)
    if method == "2l.pan":
        moments = _PanMoments.from_data(data)
        return np.array([_pan_draw(data, iterations, np.random.default_rng(stream), moments)
                         for stream in np.random.SeedSequence(seed).spawn(m)])
    return np.array([_norm_draw(data, iterations, np.random.default_rng(stream), options.ridge)
                     for stream in np.random.SeedSequence(seed).spawn(m)])


def _draw_for_chain(y, predictors, types, observed, target, method, options, rng):
    iterations = _iterations(method, options)
    data = _prepare(y, predictors, types, observed, target, method, options)
    if method == "2l.pan":
        return _pan_draw(data, iterations, rng, _PanMoments.from_data(data))
    return _norm_draw(data, iterations, rng, options.ridge)
