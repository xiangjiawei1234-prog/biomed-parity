"""Laplace likelihood for a zero-inflated Poisson random-intercept model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit


@dataclass(frozen=True)
class ZeroInflatedPoissonResult:
    count_coefficients: tuple[float, ...]
    inflation_coefficients: tuple[float, ...]
    random_intercept_sd: float
    log_likelihood: float
    converged: bool
    iterations: int
    message: str


@dataclass(frozen=True)
class _ZipData:
    counts: NDArray[np.float64]
    groups: NDArray[np.intp]
    count_design: NDArray[np.float64]
    inflation_design: NDArray[np.float64]
    offset: NDArray[np.float64]
    group_count: int

    def likelihood(
        self, count_coefficients: NDArray, inflation_coefficients: NDArray, sd: float
    ) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            fixed_count = np.einsum("ij,j->i", self.count_design, count_coefficients) + self.offset
            logit_zi = np.einsum("ij,j->i", self.inflation_design, inflation_coefficients)
        if not np.isfinite(fixed_count).all() or not np.isfinite(logit_zi).all():
            raise ValueError("linear predictors exceed the supported numerical range")
        log_zi = -np.logaddexp(0, -logit_zi)
        log_count = -np.logaddexp(0, logit_zi)
        zero = self.counts == 0
        constants = gammaln(1 + self.counts)

        def conditional(mode):
            eta = fixed_count + sd * mode[self.groups]
            # At higher points exp(eta) would overflow. The bounded value only
            # affects trial evaluations far outside the supported fit range.
            mu = np.exp(np.minimum(eta, 340.0))
            active = expit(-logit_zi[zero] - mu[zero])
            ll = log_count + self.counts * eta - mu - constants
            ll[zero] = np.logaddexp(log_zi[zero], ll[zero])
            gradient = self.counts - mu
            gradient[zero] *= active
            hessian = -mu
            hessian[zero] = gradient[zero] * (1 - mu[zero] * (1 - active))
            return ll, gradient, hessian

        mode = np.zeros(self.group_count)
        if sd == 0:
            if (fixed_count > 340).any():
                raise ValueError("count predictor exceeds the supported numerical range")
            return float(conditional(mode)[0].sum())

        max_eta = np.full(self.group_count, -np.inf)
        np.maximum.at(max_eta, self.groups, fixed_count)
        lower = -np.maximum(5.0, (max_eta + 8) / sd)
        upper = np.maximum(5.0, sd * np.bincount(self.groups, weights=self.counts) + 1)
        for _ in range(100):
            ll, gradient, hessian = conditional(mode)
            score = sd * np.bincount(self.groups, weights=gradient) - mode
            curvature = 1 - sd**2 * np.bincount(self.groups, weights=hessian)
            converged = np.abs(score) <= 1e-9
            if converged.all() and (curvature > 0).all():
                if (fixed_count + sd * mode[self.groups] > 340).any():
                    raise ValueError("count predictor exceeds the supported numerical range")
                return float(ll.sum() - 0.5 * (np.sum(mode**2) + np.log(curvature).sum()))

            lower = np.where((score > 0) & ~converged, mode, lower)
            upper = np.where((score <= 0) & ~converged, mode, upper)
            with np.errstate(divide="ignore", invalid="ignore"):
                proposal = mode + score / curvature
            use_newton = (curvature > 0) & (lower < proposal) & (proposal < upper)
            proposal = np.where(use_newton, proposal, (lower + upper) / 2)
            mode = np.where(converged & (curvature > 0), mode, proposal)

        raise ValueError("could not locate a stable conditional random-intercept mode")


def _matrix(value: ArrayLike | None, n: int, name: str) -> NDArray[np.float64]:
    matrix = np.ones((n, 1)) if value is None else np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != n or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional design matrix with {n} rows")
    if matrix.dtype.kind not in "biuf" or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return matrix.astype(float, copy=False)


def _prepare(counts, groups, count_design, inflation_design, offset) -> _ZipData:
    counts = np.asarray(counts)
    groups = np.asarray(groups)
    if counts.ndim != 1 or groups.shape != counts.shape or counts.size == 0:
        raise ValueError("counts and groups must be nonempty, equal-length vectors")
    if (
        counts.dtype.kind not in "iuf" or not np.isfinite(counts).all()
        or (counts < 0).any() or (counts != np.floor(counts)).any()
    ):
        raise ValueError("counts must contain finite nonnegative integers")
    if groups.dtype.kind in "biufUS":
        if groups.dtype.kind == "f" and not np.isfinite(groups).all():
            raise ValueError("groups must have a nonmissing label in every row")
        _, codes = np.unique(groups, return_inverse=True)
    elif groups.dtype.kind == "O" and all(isinstance(label, (str, int, np.integer)) for label in groups):
        assigned = {}
        codes = np.array([assigned.setdefault(label, len(assigned)) for label in groups], dtype=np.intp)
    else:
        raise ValueError("groups must contain strings or finite numeric labels")
    offset = np.zeros(counts.size) if offset is None else np.asarray(offset)
    if offset.shape != counts.shape or offset.dtype.kind not in "iuf" or not np.isfinite(offset).all():
        raise ValueError("offset must be a vector of finite real numbers with one value per row")

    return _ZipData(
        counts.astype(float, copy=False), codes,
        _matrix(count_design, counts.size, "count_design"),
        _matrix(inflation_design, counts.size, "inflation_design"),
        offset.astype(float, copy=False), int(codes.max()) + 1,
    )


def _coefficients(value: ArrayLike, count: int, name: str) -> NDArray[np.float64]:
    values = np.asarray(value)
    if values.shape != (count,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a vector of {count} finite real numbers")
    return values.astype(float, copy=False)


def zip_random_intercept_loglikelihood(
    counts: ArrayLike,
    groups: ArrayLike,
    count_coefficients: ArrayLike,
    inflation_coefficients: ArrayLike,
    random_intercept_sd: float,
    *,
    count_design: ArrayLike | None = None,
    inflation_design: ArrayLike | None = None,
    offset: ArrayLike | None = None,
) -> float:
    """Evaluate the ZIP log-likelihood using the first-order Laplace approximation.

    The count predictor has a Gaussian group intercept. The inflation predictor
    supplies logits of structural-zero probabilities.
    """
    if not np.isfinite(random_intercept_sd) or random_intercept_sd < 0:
        raise ValueError("random_intercept_sd must be finite and nonnegative")
    data = _prepare(counts, groups, count_design, inflation_design, offset)
    beta = _coefficients(count_coefficients, data.count_design.shape[1], "count_coefficients")
    gamma = _coefficients(
        inflation_coefficients, data.inflation_design.shape[1], "inflation_coefficients"
    )
    return data.likelihood(beta, gamma, random_intercept_sd)


def fit_zip_random_intercept(
    counts: ArrayLike,
    groups: ArrayLike,
    *,
    count_design: ArrayLike | None = None,
    inflation_design: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    max_iterations: int = 300,
) -> ZeroInflatedPoissonResult:
    """Fit a ZIP model with a random intercept in the Poisson mean."""
    data = _prepare(counts, groups, count_design, inflation_design, offset)
    positive = data.counts > 0
    if not positive.any() or positive.all() or data.group_count < 2:
        raise ValueError("fitting requires zeros, positive counts, and at least two groups")
    p, q = data.count_design.shape[1], data.inflation_design.shape[1]
    if np.linalg.matrix_rank(data.count_design) < p or np.linalg.matrix_rank(data.inflation_design) < q:
        raise ValueError("both fixed-effect design matrices must have full column rank")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    initial_beta = np.linalg.lstsq(
        data.count_design[positive], np.log(data.counts[positive]) - data.offset[positive], rcond=None
    )[0]
    estimated_excess = np.clip(np.mean(~positive) - np.exp(-data.counts.mean()), 0.02, 0.9)
    initial_gamma = np.linalg.lstsq(
        data.inflation_design, np.full(data.counts.size, logit(estimated_excess)), rcond=None
    )[0]
    initial = np.concatenate((initial_beta, initial_gamma, [np.log(0.5)]))

    def objective(parameters):
        try:
            value = data.likelihood(parameters[:p], parameters[p:-1], float(np.exp(parameters[-1])))
        except ValueError:
            return 1e30
        return -value if np.isfinite(value) else 1e30

    lower_log_sd, upper_log_sd = -12.0, 3.0
    result = minimize(
        objective, initial, method="L-BFGS-B",
        bounds=[(None, None)] * (p + q) + [(lower_log_sd, upper_log_sd)],
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-6},
    )
    at_boundary = (
        result.x[-1] <= lower_log_sd + 1e-6 or result.x[-1] >= upper_log_sd - 1e-6
    )
    message = (
        "random-intercept SD reached a numerical fitting boundary" if at_boundary
        else str(result.message)
    )
    sd = float(np.exp(result.x[-1]))
    return ZeroInflatedPoissonResult(
        count_coefficients=tuple(float(beta) for beta in result.x[:p]),
        inflation_coefficients=tuple(float(gamma) for gamma in result.x[p:-1]),
        random_intercept_sd=sd,
        log_likelihood=data.likelihood(result.x[:p], result.x[p:-1], sd),
        converged=bool(result.success and not at_boundary),
        iterations=int(result.nit),
        message=message,
    )
