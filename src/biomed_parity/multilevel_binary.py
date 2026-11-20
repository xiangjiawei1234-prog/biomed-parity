"""Two-level binary imputation with a Laplace logistic random-intercept fit."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from .multilevel_imputation import _inputs, _positive_integer


@dataclass(frozen=True)
class TwoLevelBinaryOptions:
    """Sampling convention for random effects in clusters with observed data."""

    random_effects: Literal["laplace", "eb", "marginal"] = "laplace"


@dataclass(frozen=True)
class BinaryRandomInterceptFit:
    """Laplace GLMM fit and posterior mode information for observed clusters."""

    coefficients: NDArray[np.float64]
    coefficient_covariance: NDArray[np.float64]
    random_variance: float
    groups: NDArray[np.float64]
    random_modes: NDArray[np.float64]
    conditional_variances: NDArray[np.float64]
    log_likelihood: float


@dataclass(frozen=True)
class _LaplaceLogistic:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    successes: np.ndarray
    failures: np.ndarray

    def grouped(self, values):
        return np.bincount(self.groups, weights=values, minlength=len(self.successes))

    def mode(self, linear, scale):
        lower = -scale * self.failures
        upper = scale * self.successes
        u = np.zeros(len(self.successes))
        for _ in range(80):
            eta = linear + scale * u[self.groups]
            p = expit(eta)
            weight = p * expit(-eta)
            residual = np.where(self.y == 1, -expit(-eta), p)
            group_residual = self.grouped(residual)
            group_weight = self.grouped(weight)
            gradient = u + scale * group_residual
            hessian = 1 + scale**2 * group_weight
            stable = abs(gradient) <= 2e-12 * (1 + abs(u))
            if stable.all():
                return u, eta, p, weight, residual, group_residual, group_weight, hessian
            lower = np.where(gradient < 0, u, lower)
            upper = np.where(gradient > 0, u, upper)
            candidate = u - gradient / hessian
            candidate = np.where((candidate < lower) | (candidate > upper),
                                 (lower + upper) / 2, candidate)
            u = np.where(stable, u, candidate)
        raise ArithmeticError("random-intercept posterior modes did not converge")

    def objective(self, parameters):
        beta, scale = parameters[:-1], parameters[-1]
        values = self.mode(self.x @ beta, scale)
        u, eta, p, weight, residual, group_residual, group_weight, hessian = values
        objective = (
            np.logaddexp(0, (1 - 2 * self.y) * eta).sum()
            + 0.5 * (u @ u + np.log(hessian).sum())
        )
        weighted_x = weight[:, None] * self.x
        hessian_x = np.column_stack([self.grouped(column) for column in weighted_x.T])
        u_beta = -scale * hessian_x / hessian[:, None]
        u_scale = -(group_residual + scale * u * group_weight) / hessian
        curvature = 0.5 * scale**2 / hessian[self.groups] * weight * (1 - 2 * p)
        beta_gradient = self.x.T @ residual + (
            self.x + scale * u_beta[self.groups]
        ).T @ curvature
        scale_gradient = (
            u[self.groups] @ residual + scale * np.sum(group_weight / hessian)
            + (u + scale * u_scale)[self.groups] @ curvature
        )
        return objective, np.r_[beta_gradient, scale_gradient]


def _fixed_covariance(model, parameters):
    scale = parameters[-1]
    if scale == 0:
        eta = model.x @ parameters[:-1]
        weight = expit(eta) * expit(-eta)
        information = model.x.T @ (model.x * weight[:, None])
        return np.linalg.inv(information)
    curvature = np.empty((len(parameters), len(parameters)))
    for column, value in enumerate(parameters):
        step = np.cbrt(np.finfo(float).eps) * max(1, abs(value))
        if column == len(parameters) - 1:
            step = min(step, scale / 3)
        lower, upper = parameters.copy(), parameters.copy()
        lower[column] -= step
        upper[column] += step
        curvature[:, column] = (model.objective(upper)[1] - model.objective(lower)[1]) / (2 * step)
    covariance = np.linalg.inv((curvature + curvature.T) / 2)[:-1, :-1]
    np.linalg.cholesky(covariance)
    return covariance


def _fit_binary(y, x, groups):
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("2l.bin requires both observed binary classes coded 0 and 1")
    levels, labels = np.unique(groups, return_inverse=True)
    if not 2 <= len(levels) < len(y):
        raise ValueError("2l.bin requires at least two observed clusters and replication within clusters")
    center, spread = x.mean(axis=0), x.std(axis=0)
    if np.any(spread == 0):
        raise ValueError("2l.bin fixed effects must have full observed rank")
    design = np.column_stack((np.ones(len(y)), (x - center) / spread))
    size = design.shape[1]
    if np.linalg.matrix_rank(design) != size:
        raise ValueError("2l.bin fixed effects must have full observed rank")
    successes = np.bincount(labels, weights=y, minlength=len(levels))
    failures = np.bincount(labels, minlength=len(levels)) - successes
    model = _LaplaceLogistic(design, y, labels, successes, failures)
    initial = np.r_[logit(y.mean()), np.zeros(size - 1)]
    options = {"maxiter": 400, "ftol": 1e-13, "gtol": 1e-8}

    def fixed_only(beta):
        value, gradient = model.objective(np.r_[beta, 0])
        return value, gradient[:-1]

    baseline = minimize(fixed_only, initial, jac=True, method="L-BFGS-B", options=options)
    if not baseline.success:
        raise ValueError("2l.bin fixed-effect fit did not converge")
    parameters, objective = np.r_[baseline.x, 0.0], float(baseline.fun)
    for start in (0.6, 2.0):
        candidate = minimize(
            model.objective, np.r_[baseline.x, start], jac=True, method="L-BFGS-B",
            bounds=[(None, None)] * size + [(0, None)], options=options,
        )
        if candidate.success and candidate.fun < objective - 1e-9:
            parameters, objective = candidate.x, float(candidate.fun)
    if np.max(abs(parameters[:-1])) > 40:
        raise ValueError("2l.bin fixed effects do not have a stable finite fit")

    scale = parameters[-1]
    covariance = _fixed_covariance(model, parameters)
    mode, *_, hessian = model.mode(design @ parameters[:-1], scale)
    conversion = np.eye(size)
    conversion[0, 1:] = -center / spread
    conversion[1:, 1:] = np.diag(1 / spread)
    return BinaryRandomInterceptFit(
        conversion @ parameters[:-1], conversion @ covariance @ conversion.T,
        float(scale**2), levels, scale * mode, scale**2 / hessian, -objective,
    )


def _prepare_binary(y, x, types, observed, target, options):
    if not isinstance(options, TwoLevelBinaryOptions) or options.random_effects not in (
        "laplace", "eb", "marginal",
    ):
        raise ValueError("options must specify laplace, eb, or marginal random effects")
    if not np.isin(types, [-2, 0, 1]).all():
        raise ValueError("2l.bin supports cluster code -2 and fixed predictors coded 1")
    group = x[:, types == -2]
    if group.shape[1] != 1 or not np.isfinite(group).all():
        raise ValueError("exactly one complete cluster column must be coded -2")
    fixed = x[:, types == 1]
    complete = np.isfinite(fixed).all(axis=1)
    if np.any(target & ~complete):
        raise ValueError("a requested row has an incomplete predictor")
    use = observed & complete
    fit = _fit_binary(y[use], fixed[use], group[use, 0])
    return fit, fixed[target], group[target, 0]


def _binary_draws(fit, target, groups, options, rng):
    beta_factor = np.linalg.cholesky(fit.coefficient_covariance)
    beta = fit.coefficients + beta_factor @ rng.standard_normal(len(fit.coefficients))
    random_scale = fit.random_modes @ fit.random_modes + fit.random_variance
    marginal = random_scale / rng.chisquare(len(fit.groups) + 1)
    labels, inverse = np.unique(groups, return_inverse=True)
    means = np.zeros(len(labels))
    variances = np.full(len(labels), marginal)
    if options.random_effects != "marginal":
        observed = np.isin(labels, fit.groups)
        matching = np.searchsorted(fit.groups, labels[observed])
        means[observed] = fit.random_modes[matching]
        if options.random_effects == "laplace":
            variances[observed] = fit.conditional_variances[matching]
    random_intercept = means + np.sqrt(variances) * rng.standard_normal(len(labels))
    linear = beta[0] + target @ beta[1:] + random_intercept[inverse]
    return rng.random(len(target)) < expit(linear)


def impute_two_level_binary(
    y: ArrayLike,
    predictors: ArrayLike,
    types: ArrayLike,
    *,
    observed: ArrayLike | None = None,
    where: ArrayLike | None = None,
    options: TwoLevelBinaryOptions = TwoLevelBinaryOptions(),
    m: int = 5,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Impute 0/1 responses with a normally distributed cluster intercept.

    Specify the cluster with code -2, fixed predictors with 1, and unused
    columns with 0. Draw conditional or marginal random intercepts, with one
    shared draw per cluster within each completed dataset.
    """
    m = _positive_integer(m, "m")
    y, predictors, types, observed, target = _inputs(y, predictors, types, observed, where)
    if not target.any():
        return np.empty((m, 0))
    fit, x, groups = _prepare_binary(y, predictors, types, observed, target, options)
    return np.asarray([
        _binary_draws(fit, x, groups, options, np.random.default_rng(stream))
        for stream in np.random.SeedSequence(seed).spawn(m)
    ], dtype=float)


def _draw_binary_for_chain(y, predictors, types, observed, target, options, rng):
    fit, x, groups = _prepare_binary(y, predictors, types, observed, target, options)
    return _binary_draws(fit, x, groups, options, rng).astype(float)
