"""Multilevel Gaussian meta-regression with known sampling covariance."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, eigh, qr, solve_triangular
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, f, norm, t


Array = NDArray[np.float64]


@dataclass(frozen=True)
class MetaCoefficientTests:
    covariance: Array
    standard_errors: Array
    statistics: Array
    p_values: Array
    confidence_intervals: Array
    degrees_of_freedom: Array


@dataclass(frozen=True)
class MetaWaldTest:
    statistic: float
    numerator_df: int
    denominator_df: float | None
    p_value: float


@dataclass(frozen=True)
class _MetaFitInputs:
    sampling: Array
    groups: dict[str, NDArray[np.intp]]
    kernels: dict[str, Array]
    fixed_variances: dict[str, float]
    intercept: bool
    test: str
    dfs: str | Array
    reestimation_supported: bool = True


@dataclass(frozen=True)
class MetaMultilevelResult:
    """Fitted coefficients, covariance components, and model-based inference."""

    coefficients: Array
    inference: MetaCoefficientTests
    variance_components: dict[str, float]
    variance_intervals: dict[str, tuple[float, float]] | None
    marginal_covariance: Array
    residuals: Array
    log_likelihood: float
    heterogeneity: MetaWaldTest
    omnibus: MetaWaldTest
    method: str
    confidence_level: float
    _design: Array = field(repr=False)
    _tested: NDArray[np.intp] = field(repr=False)
    _random_variance: Array | None = field(default=None, repr=False)
    _estimated_variances: int | None = field(default=None, repr=False)
    _fit_inputs: _MetaFitInputs | None = field(default=None, repr=False)


@dataclass(frozen=True)
class MetaRobustResult:
    """Cluster-robust coefficient and joint Wald tests."""

    inference: MetaCoefficientTests
    omnibus: MetaWaldTest
    cluster_count: int
    correction: str


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain finite real numbers")
    array = raw.astype(float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def _codes(labels, size, name):
    """Factorize scalar labels or tuples of labels."""
    try:
        rows = tuple(labels) if not isinstance(labels, (str, bytes)) else ()
    except TypeError as exc:
        raise ValueError(f"{name} must have one label per effect") from exc
    if len(rows) != size:
        raise ValueError(f"{name} must have one label per effect")
    levels = {}
    codes = np.empty(size, dtype=np.intp)
    for i, label in enumerate(rows):
        # A row of nested labels describes a single joint level.
        if isinstance(label, (list, np.ndarray)):
            label = tuple(label)
        parts = label if isinstance(label, tuple) else (label,)
        if not parts:
            raise ValueError(f"{name} contains an empty group label")
        for part in parts:
            if not isinstance(part, (str, Real)) or (
                isinstance(part, Real) and not np.isfinite(part)
            ):
                raise ValueError(f"{name} must contain finite numeric or string labels")
        codes[i] = levels.setdefault(label, len(levels))
    return codes


def _covariance(value, size, name, *, vector=False):
    matrix = _real_array(value, name)
    if vector and matrix.shape == (size,):
        if np.any(matrix < 0):
            raise ValueError(f"{name} variances must be nonnegative")
        return np.diag(matrix)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must be square with one row per effect")
    scale = max(np.max(np.abs(matrix)), np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) / 2
    if np.linalg.eigvalsh(matrix / scale)[0] < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _design_matrix(moderators, size, intercept):
    design = np.empty((size, 0)) if moderators is None else _real_array(moderators, "moderators")
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or len(design) != size:
        raise ValueError("moderators must be a vector or matrix with one row per effect")
    if intercept:
        design = np.column_stack((np.ones(size), design))
    if not 0 < design.shape[1] < size:
        raise ValueError("the design must have at least one coefficient and positive residual degrees of freedom")
    column_scale = np.max(np.abs(design), axis=0)
    if np.any(column_scale == 0) or np.linalg.matrix_rank(design / column_scale) != design.shape[1]:
        raise ValueError("moderators and intercept must form a full-rank design")
    return design, column_scale


def _gls(y, design, decomposition):
    factor = decomposition[0]
    y_w = solve_triangular(factor, y, lower=True, check_finite=False)
    x_w = solve_triangular(factor, design, lower=True, check_finite=False)
    basis, r = qr(x_w, mode="economic", check_finite=False)
    beta = solve_triangular(r, np.einsum("ij,i->j", basis, y_w), check_finite=False)
    return beta, y_w - np.einsum("ij,j->i", x_w, beta), basis, r


@dataclass(frozen=True)
class _VarianceObjective:
    y: Array
    sampling: Array
    components: Array
    design: Array | None

    def covariance(self, variances):
        return self.sampling + np.einsum("i,ijk->jk", variances, self.components)

    def evaluate(self, variances, *, gradient):
        try:
            factor = cho_factor(self.covariance(variances), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(variances)
        lower = factor[0]
        residual = (
            solve_triangular(lower, self.y, lower=True, check_finite=False)
            if self.design is None else _gls(self.y, self.design, factor)[1]
        )
        nll = float(np.log(lower.diagonal()).sum() + 0.5 * (residual @ residual))
        if not gradient:
            return nll, np.zeros_like(variances)
        precision = cho_solve(factor, np.eye(len(self.y)), check_finite=False)
        score_residual = solve_triangular(
            lower, residual, lower=True, trans="T", check_finite=False,
        )
        score_matrix = precision - np.outer(score_residual, score_residual)
        score = 0.5 * np.einsum("jk,ijk->i", score_matrix, self.components)
        return nll, score

    def fit(self, fixed, *, initial=None, multistart=True):
        free = np.isnan(fixed)
        if not np.any(free):
            return fixed.copy(), self.evaluate(fixed, gradient=False)[0]

        point = fixed.copy()
        component_scale = np.trace(self.components[free], axis1=1, axis2=2) / len(self.y)

        def objective(position):
            point[free] = position / component_scale
            value, gradient = self.evaluate(point, gradient=True)
            return value, gradient[free] / component_scale

        dimension = int(free.sum())
        starts = [np.full(dimension, 0.2)]
        if initial is not None:
            starts.insert(0, initial[free] * component_scale)
        if multistart:
            starts = [np.zeros(dimension), *starts, np.ones(dimension)]
            if 1 < dimension <= 4:
                starts.extend(np.full(dimension, 0.03) + row for row in np.eye(dimension))
        best = None
        for start in starts:
            if not np.isfinite(objective(start)[0]):
                continue
            result = minimize(
                objective, start, method="L-BFGS-B", jac=True,
                bounds=[(0.0, None)] * dimension,
                options={"ftol": 5e-15, "gtol": 2e-9, "maxiter": 600, "maxls": 50},
            )
            projected = np.where(result.x == 0.0, np.minimum(result.jac, 0), result.jac)
            if np.isfinite(result.fun) and np.max(np.abs(projected)) < 2e-5:
                if best is None or result.fun < best.fun:
                    best = result
        if best is None:
            raise ValueError("variance component optimization did not converge to a positive-definite fit")
        point[free] = best.x / component_scale
        return point, float(best.fun)

    def confidence_intervals(self, estimates, fixed, best_value, level):
        cutoff = best_value + chi2.isf(1 - level, 1) / 2
        intervals = {}
        for i in np.flatnonzero(np.isnan(fixed)):
            trial = fixed.copy()

            def difference(value):
                trial[i] = value
                _, nll = self.fit(trial, initial=estimates, multistart=False)
                return nll - cutoff

            estimate = estimates[i]
            lower = 0.0
            if estimate > 0:
                try:
                    outside = difference(0.0) > 0
                except ValueError:
                    outside = True
                if outside:
                    def lower_difference(value):
                        return np.inf if value == 0 else difference(value)

                    lower = brentq(lower_difference, 0, estimate, xtol=min(1e-9, 1e-4 * estimate))
            upper = max(2 * estimate, 1.0)
            for _ in range(60):
                if difference(upper) > 0:
                    break
                upper *= 2
            else:
                raise ValueError("could not bracket the upper variance profile limit")
            intervals[i] = (lower, brentq(difference, estimate, upper, xtol=1e-9))
        return intervals


def _degrees_of_freedom(dfs, design, groups):
    size, count = design.shape
    if isinstance(dfs, str):
        if dfs == "residual":
            return np.full(count, size - count, dtype=float)
        if dfs != "contain":
            raise ValueError("dfs must be 'residual', 'contain', or positive numbers")
        levels = np.full(count, size)
        for codes in groups:
            group_count = codes.max() + 1
            minima = np.full((group_count, count), np.inf)
            maxima = np.full((group_count, count), -np.inf)
            np.minimum.at(minima, codes, design)
            np.maximum.at(maxima, codes, design)
            levels = np.where(np.all(minima == maxima, axis=0), np.minimum(levels, group_count), levels)
        return np.maximum(levels - count, 1).astype(float)
    degrees = _real_array(dfs, "dfs")
    if degrees.shape not in ((), (count,)) or np.any(degrees <= 0):
        raise ValueError("dfs must supply one positive value or one per coefficient")
    return np.broadcast_to(degrees, (count,)).copy()


def _coefficient_tests(beta, covariance, degrees, level):
    standard_error = np.sqrt(covariance.diagonal())
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = beta / standard_error
    distribution = t(degrees) if degrees is not None else norm
    p = 2 * distribution.sf(np.abs(statistic))
    radius = distribution.isf((1 - level) / 2) * standard_error
    return MetaCoefficientTests(
        covariance, standard_error, statistic, p, np.column_stack((beta - radius, beta + radius)),
        np.full(len(beta), np.nan) if degrees is None else degrees,
    )


def _wald(beta, covariance, tested, denominator_df=None, *, htz_scale=None):
    count = len(tested)
    estimate = beta[tested]
    try:
        subset = covariance[np.ix_(tested, tested)]
        factor = cho_factor(subset, lower=True, check_finite=True)
        raw = float(estimate @ cho_solve(factor, estimate, check_finite=False))
    except (np.linalg.LinAlgError, ValueError):
        return MetaWaldTest(np.nan, count, denominator_df, np.nan)
    if denominator_df is None:
        return MetaWaldTest(raw, count, None, float(chi2.sf(raw, count)))
    statistic = raw / count if htz_scale is None else raw * htz_scale
    return MetaWaldTest(statistic, count, denominator_df, float(f.sf(statistic, count, denominator_df)))


def _variance_objective(y, sampling, components, x_scaled, method):
    size, count = x_scaled.shape
    basis, _ = qr(x_scaled, mode="full", check_finite=False)
    q = basis[:, count:]
    residual_projection = q.T.dot(y)
    scale = max(float(np.max(sampling.diagonal())), float(np.mean(residual_projection**2)))
    if not np.isfinite(scale) or not scale > 0:
        raise ValueError("effects or sampling covariance exceed the supported numerical range")
    if method == "REML":
        projected = np.stack([q.T.dot(kernel).dot(q) for kernel in components]) if len(components) else (
            np.empty((0, size - count, size - count))
        )
        return _VarianceObjective(
            residual_projection / np.sqrt(scale), q.T.dot(sampling / scale).dot(q),
            projected, None,
        ), scale
    return _VarianceObjective(y / np.sqrt(scale), sampling / scale, components, x_scaled), scale


def _require_estimable(projected, components):
    if not len(projected):
        return
    matrices = projected.reshape((len(projected), -1))
    maxima = np.max(np.abs(matrices), axis=1, keepdims=True)
    base_norms = np.max(np.abs(components), axis=(1, 2))[:, None]
    if (
        np.any(maxima <= 1e-12 * base_norms) or np.any(maxima == 0)
        or np.linalg.matrix_rank(matrices / maxima, tol=1e-9) != len(matrices)
    ):
        raise ValueError("estimated variance components are not identifiable under this design")


def meta_multilevel(
    effects: ArrayLike,
    sampling_covariance: ArrayLike,
    *,
    moderators: ArrayLike | None = None,
    intercept: bool = True,
    random_intercepts: Mapping[str, ArrayLike] | None = None,
    covariance_components: Mapping[str, ArrayLike] | None = None,
    fixed_variances: Mapping[str, float] | None = None,
    method: Literal["REML", "ML"] = "REML",
    test: Literal["z", "t", "knha", "adhoc"] = "z",
    dfs: Literal["residual", "contain"] | ArrayLike = "residual",
    confidence_level: float = 0.95,
    tested_coefficients: ArrayLike | None = None,
    profile: bool = False,
) -> MetaMultilevelResult:
    """Fit Gaussian meta-regression with a full known sampling covariance.

    Named random-intercept terms take one label per effect. Supply a tuple of
    labels for a nested level, e.g. (study, endpoint) plus a separate study term.
    Additional positive-semidefinite kernels allow known random-effect
    correlation structures. Variance components not in fixed_variances are
    estimated on the nonnegative scale. Set profile for likelihood-ratio
    intervals for each estimated variance component.
    """
    y = _real_array(effects, "effects")
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("effects must be a vector with at least two observations")
    if method not in ("REML", "ML") or test not in ("z", "t", "knha", "adhoc"):
        raise ValueError("method must be REML or ML and test must be z, t, knha, or adhoc")
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    size = len(y)
    design, column_scale = _design_matrix(moderators, size, intercept)
    count = design.shape[1]
    x_scaled = design / column_scale
    if tested_coefficients is None:
        tested = np.arange(int(intercept and count > 1), count)
    else:
        tested = np.asarray(tested_coefficients)
        if (
            tested.ndim != 1 or not len(tested) or tested.dtype.kind not in "iu"
            or np.any(tested < 0) or np.any(tested >= count) or len(np.unique(tested)) != len(tested)
        ):
            raise ValueError("tested_coefficients must contain unique zero-based coefficient indices")
        tested = tested.astype(np.intp)

    sampling = _covariance(sampling_covariance, size, "sampling_covariance", vector=True)
    intercept_terms = random_intercepts or {}
    extra = covariance_components or {}
    names = tuple(intercept_terms) + tuple(extra)
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("variance components must have unique nonempty string names")
    labels = [_codes(value, size, name) for name, value in intercept_terms.items()]
    kernels = [(code[:, None] == code).astype(float) for code in labels]
    kernels.extend(_covariance(value, size, name) for name, value in extra.items())
    components = np.asarray(kernels, dtype=float).reshape((-1, size, size))
    fixed_variances = fixed_variances or {}
    if set(fixed_variances) - set(names):
        raise ValueError("fixed_variances names must match a specified variance component")
    fixed = np.full(len(names), np.nan)
    for i, name in enumerate(names):
        if name in fixed_variances:
            value = fixed_variances[name]
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError("fixed variances must be finite and nonnegative")
            fixed[i] = value

    objective, scale = _variance_objective(y, sampling, components, x_scaled, method)
    y_scaled = y / np.sqrt(scale)
    known = sampling / scale
    free = np.isnan(fixed)
    _require_estimable(objective.components[free], components[free])

    fitted, nll = objective.fit(fixed / scale)
    if not np.isfinite(nll):
        raise ValueError("model covariance must be positive definite")
    marginal_scaled = known + np.einsum("i,ijk->jk", fitted, components)
    try:
        decomposition = cho_factor(marginal_scaled, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fitted marginal covariance must be positive definite") from exc
    beta_scaled, res_scaled, _, r = _gls(y_scaled, x_scaled, decomposition)
    beta = beta_scaled * np.sqrt(scale) / column_scale
    residual = y - design @ beta
    inverse_r = solve_triangular(r, np.eye(count), check_finite=False)
    inverse_r *= np.sqrt(scale) / column_scale[:, None]
    covariance = inverse_r @ inverse_r.T
    if test in ("knha", "adhoc"):
        rss = float(res_scaled @ res_scaled)
        adjustment = 0.0 if rss <= np.finfo(float).eps else rss / (size - count)
        covariance *= max(1.0, adjustment) if test == "adhoc" else adjustment
    degrees = None if test == "z" else _degrees_of_freedom(dfs, design, labels)
    inference = _coefficient_tests(beta, covariance, degrees, confidence_level)
    omnibus = _wald(beta, covariance, tested, None if degrees is None else float(degrees[tested].min()))

    try:
        fixed_decomp = cho_factor(known, lower=True, check_finite=False)
        fixed_residual = _gls(y_scaled, x_scaled, fixed_decomp)[1]
        qe = float(fixed_residual @ fixed_residual)
    except np.linalg.LinAlgError:
        qe = np.nan
    heterogeneity = MetaWaldTest(qe, size - count, None, float(chi2.sf(qe, size - count)))
    intervals = None
    if profile:
        intervals = {
            names[i]: (scale * low, scale * high)
            for i, (low, high) in objective.confidence_intervals(
                fitted, fixed / scale, nll, confidence_level
            ).items()
        }
    log_likelihood = -nll - 0.5 * len(objective.y) * (np.log(2 * np.pi) + np.log(scale))
    random_variance = (fitted * scale) @ np.diagonal(components, axis1=1, axis2=2)
    fit_inputs = _MetaFitInputs(
        sampling, dict(zip(intercept_terms, labels, strict=True)),
        {name: components[i].copy() for i, name in enumerate(names) if name in extra},
        {name: float(fixed[i]) for i, name in enumerate(names) if not free[i]},
        intercept, test, dfs if isinstance(dfs, str) or degrees is None else degrees,
    )
    return MetaMultilevelResult(
        beta, inference, dict(zip(names, map(float, fitted * scale), strict=True)), intervals,
        marginal_scaled * scale, residual, float(log_likelihood), heterogeneity, omnibus,
        method, float(confidence_level), design, tested, random_variance, int(free.sum()), fit_inputs,
    )


def _whitened_clusters(fit, codes):
    covariance = fit.marginal_covariance
    off_cluster = codes[:, None] != codes
    if np.any(np.abs(covariance[off_cluster]) > 1e-10 * np.max(np.abs(covariance))):
        raise ValueError("CR2 requires marginal covariance block-diagonal by cluster")
    factor = cho_factor(covariance, lower=True, check_finite=False)[0]
    y = solve_triangular(factor, fit.residuals, lower=True, check_finite=False)
    x = solve_triangular(factor, fit._design, lower=True, check_finite=False)
    basis, upper = qr(x, mode="economic", check_finite=False)
    bread = solve_triangular(upper, basis.T, check_finite=False)
    n, p = x.shape
    groups = int(codes.max()) + 1
    influences = np.zeros((n, groups, p))
    scores = np.empty((groups, p))
    for group in range(groups):
        keep = codes == group
        rows = basis[keep]
        complement = np.eye(len(rows)) - rows.dot(rows.T)
        local_factor = np.tril(factor[np.ix_(keep, keep)])
        gram = local_factor.T.dot(local_factor)
        # In the original observation space clubSandwich uses a symmetric
        # adjustment. Its transformation to whitened space is S^(-1/2) G,
        # with S = G (I - H_gg) G and G = L_g' L_g.
        moment = gram.dot(complement).dot(gram)
        eigenvalues, vectors = eigh(moment, check_finite=False)
        inverse_sqrt = np.zeros_like(eigenvalues)
        regular = eigenvalues > 1e-12 * max(eigenvalues[-1], np.finfo(float).tiny)
        inverse_sqrt[regular] = 1 / np.sqrt(eigenvalues[regular])
        adjusted = gram.dot(vectors * inverse_sqrt).dot(vectors.T.dot(bread[:, keep].T))
        influences[keep, group, :] = adjusted
        scores[group] = y[keep] @ adjusted
    flattened = influences.reshape((n, -1))
    flattened -= basis.dot(basis.T.dot(flattened))
    return scores.T @ scores, influences, bread @ bread.T


def _satterthwaite(influences):
    p = influences.shape[-1]
    dfs = np.empty(p)
    for i in range(p):
        rows = influences[:, :, i]
        gram = rows.T @ rows
        squared_sum = np.sum(gram**2)
        dfs[i] = np.trace(gram)**2 / squared_sum if squared_sum > 0 else np.nan
    return dfs


def _hotelling_test(fit, covariance, influences, target):
    tested = fit._tested
    count = len(tested)
    lower = cho_factor(target[np.ix_(tested, tested)], lower=True, check_finite=False)[0]
    selected = influences[:, :, tested]
    normalized = solve_triangular(
        lower, selected.reshape((-1, count)).T, lower=True, check_finite=False,
    ).T.reshape(selected.shape)
    grams = [normalized[:, :, i].T @ normalized[:, :, i] for i in range(count)]
    variance = 0.0
    for i in range(count):
        for j in range(i + 1):
            cross = normalized[:, :, i].T @ normalized[:, :, j]
            covariance_term = np.sum(grams[i] * grams[j] + cross * cross.T)
            variance += covariance_term if i == j else 2 * covariance_term
    if variance <= 0:
        return MetaWaldTest(np.nan, count, np.nan, np.nan)
    shape = float(count * (count + 1) / variance)
    denominator_df = shape - count + 1
    if denominator_df <= 0:
        return MetaWaldTest(np.nan, count, denominator_df, np.nan)
    return _wald(fit.coefficients, covariance, tested, denominator_df,
                 htz_scale=denominator_df / (shape * count))


def meta_cluster_robust(
    fit: MetaMultilevelResult,
    clusters: ArrayLike,
    *,
    correction: Literal["CR0", "CR1", "CR2"] = "CR2",
) -> MetaRobustResult:
    """Test a fitted model using a cluster-robust sandwich estimate.

    CR0/CR1 follow metafor's cluster residual degrees of freedom. CR2 uses
    the fitted covariance as working target, with Satterthwaite coefficient
    tests and the approximate Hotelling T² joint test. Its working covariance
    must be block-diagonal at the stated cluster level.
    """
    if correction not in ("CR0", "CR1", "CR2"):
        raise ValueError("correction must be 'CR0', 'CR1', or 'CR2'")
    n, p = fit._design.shape
    codes = _codes(clusters, n, "clusters")
    cluster_count = int(codes.max()) + 1
    if cluster_count < 2:
        raise ValueError("at least two clusters are required for robust inference")
    if correction == "CR2":
        covariance, influences, target = _whitened_clusters(fit, codes)
        degrees = _satterthwaite(influences)
        omnibus = _hotelling_test(fit, covariance, influences, target)
    else:
        denominator_df = cluster_count - p
        if denominator_df <= 0:
            raise ValueError("number of clusters must exceed the number of coefficients for CR0/CR1")
        decomposition = cho_factor(fit.marginal_covariance, lower=True, check_finite=False)
        weighted = cho_solve(decomposition, fit._design, check_finite=False)
        bread = fit.inference.covariance @ weighted.T
        scores = np.zeros((cluster_count, p))
        np.add.at(scores, codes, bread.T * fit.residuals[:, None])
        covariance = scores.T @ scores
        if correction == "CR1":
            covariance *= cluster_count / denominator_df
        unavailable = covariance.diagonal() < 100 * np.finfo(float).eps
        covariance[unavailable] = np.nan
        covariance[:, unavailable] = np.nan
        degrees = np.full(p, denominator_df, dtype=float)
        omnibus = _wald(fit.coefficients, covariance, fit._tested, denominator_df)

    inference = _coefficient_tests(fit.coefficients, covariance, degrees, fit.confidence_level)
    return MetaRobustResult(inference, omnibus, cluster_count, correction)
