"""Laplace likelihoods for zero-inflated count random-intercept models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, logit


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
class ZeroInflatedNegativeBinomialResult:
    """NB2 fit with log-shape dispersion coefficients."""

    count_coefficients: tuple[float, ...]
    inflation_coefficients: tuple[float, ...]
    dispersion_coefficients: tuple[float, ...]
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
        self, count_coefficients: NDArray, inflation_coefficients: NDArray, sd: float,
        log_shape: NDArray | None = None,
    ) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            fixed_count = np.einsum("ij,j->i", self.count_design, count_coefficients) + self.offset
            logit_zi = np.einsum("ij,j->i", self.inflation_design, inflation_coefficients)
        if not np.isfinite(fixed_count).all() or not np.isfinite(logit_zi).all():
            raise ValueError("linear predictors exceed the supported numerical range")
        log_zi = -np.logaddexp(0, -logit_zi)
        log_count = -np.logaddexp(0, logit_zi)
        zero = self.counts == 0
        if log_shape is None:
            constants = -gammaln(1 + self.counts)
        else:
            if not np.isfinite(log_shape).all() or (abs(log_shape) > 35).any():
                raise ValueError("log dispersion predictor exceeds the supported numerical range")
            shape = np.exp(log_shape)
            count_size = self.counts + shape
            constants = -betaln(self.counts + 1, shape) - np.log(count_size)

        def poisson(eta):
            # At higher points exp(eta) would overflow. The bounded value only
            # affects trial evaluations far outside the supported fit range.
            mu = np.exp(np.minimum(eta, 340.0))
            return constants + self.counts * eta - mu, self.counts - mu, -mu

        def negative_binomial(eta):
            log_odds = eta - log_shape
            q = expit(log_odds)
            chance = expit(-log_odds)
            probability = (constants - self.counts * np.logaddexp(0, -log_odds)
                           - shape * np.logaddexp(0, log_odds))
            return probability, self.counts * chance - shape * q, -count_size * q * chance

        count_distribution = poisson if log_shape is None else negative_binomial

        def conditional(mode):
            eta = fixed_count + sd * mode[self.groups]
            ll, gradient, hessian = count_distribution(eta)
            active = expit(ll[zero] - logit_zi[zero])
            ll += log_count
            ll[zero] = np.logaddexp(log_zi[zero], ll[zero])
            active_gradient = gradient[zero]
            gradient[zero] = active * active_gradient
            hessian[zero] = active * (
                hessian[zero] + (1 - active) * active_gradient**2
            )
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
    p, q, initial = _count_model_start(data, max_iterations)
    lower_log_sd, upper_log_sd = -12.0, 3.0

    def objective(parameters):
        try:
            value = data.likelihood(parameters[:p], parameters[p:-1], float(np.exp(parameters[-1])))
        except ValueError:
            return 1e30
        return -value if np.isfinite(value) else 1e30

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


def _count_model_start(data, max_iterations):
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
    return p, q, initial


def zinb2_random_intercept_loglikelihood(
    counts: ArrayLike,
    groups: ArrayLike,
    count_coefficients: ArrayLike,
    inflation_coefficients: ArrayLike,
    dispersion_coefficients: ArrayLike,
    random_intercept_sd: float,
    *,
    count_design: ArrayLike | None = None,
    inflation_design: ArrayLike | None = None,
    dispersion_design: ArrayLike | None = None,
    offset: ArrayLike | None = None,
) -> float:
    """Evaluate the Laplace likelihood with variance mu + mu**2 / shape."""
    if not np.isfinite(random_intercept_sd) or random_intercept_sd < 0:
        raise ValueError("random_intercept_sd must be finite and nonnegative")
    data = _prepare(counts, groups, count_design, inflation_design, offset)
    dispersion = _matrix(dispersion_design, len(data.counts), "dispersion_design")
    beta = _coefficients(count_coefficients, data.count_design.shape[1], "count_coefficients")
    gamma = _coefficients(
        inflation_coefficients, data.inflation_design.shape[1], "inflation_coefficients",
    )
    delta = _coefficients(dispersion_coefficients, dispersion.shape[1], "dispersion_coefficients")
    log_shape = np.einsum("ij,j->i", dispersion, delta)
    return data.likelihood(beta, gamma, random_intercept_sd, log_shape)


def fit_zinb2_random_intercept(
    counts: ArrayLike,
    groups: ArrayLike,
    *,
    count_design: ArrayLike | None = None,
    inflation_design: ArrayLike | None = None,
    dispersion_design: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    max_iterations: int = 400,
) -> ZeroInflatedNegativeBinomialResult:
    """Fit a ZINB2 model with log-mean, logit-inflation and log-shape predictors."""
    data = _prepare(counts, groups, count_design, inflation_design, offset)
    dispersion = _matrix(dispersion_design, len(data.counts), "dispersion_design")
    p, q, initial = _count_model_start(data, max_iterations)
    r = dispersion.shape[1]
    if np.linalg.matrix_rank(dispersion) != r:
        raise ValueError("dispersion_design must have full column rank")
    split = p + q

    def likelihood(parameters):
        log_shape = np.einsum("ij,j->i", dispersion, parameters[split:-1])
        return data.likelihood(
            parameters[:p], parameters[p:split], float(np.exp(parameters[-1])), log_shape,
        )

    def objective(parameters):
        try:
            value = likelihood(parameters)
        except ValueError:
            return 1e30
        return -value if np.isfinite(value) else 1e30

    lower_log_sd, upper_log_sd = -12.0, 3.0
    fits = []
    for shape in (0.7, 4.0, 15.0):
        initial_dispersion = np.linalg.lstsq(
            dispersion, np.full(len(dispersion), np.log(shape)), rcond=None,
        )[0]
        start = np.r_[initial[:-1], initial_dispersion, initial[-1]]
        fits.append(minimize(
            objective, start, method="L-BFGS-B",
            bounds=[(None, None)] * (p + q + r) + [(lower_log_sd, upper_log_sd)],
            options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-6},
        ))
    result = min(fits, key=lambda fit: fit.fun)
    at_boundary = result.x[-1] <= lower_log_sd + 1e-6 or result.x[-1] >= upper_log_sd - 1e-6
    return ZeroInflatedNegativeBinomialResult(
        count_coefficients=tuple(map(float, result.x[:p])),
        inflation_coefficients=tuple(map(float, result.x[p:split])),
        random_intercept_sd=float(np.exp(result.x[-1])), log_likelihood=likelihood(result.x),
        converged=bool(result.success and not at_boundary), iterations=int(result.nit),
        message=("random-intercept SD reached a numerical fitting boundary"
                 if at_boundary else str(result.message)),
        dispersion_coefficients=tuple(map(float, result.x[split:-1])),
    )
