"""Case deletion and leverage diagnostics for multilevel meta-regression."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .meta_multilevel import MetaMultilevelResult, _codes, _gls, meta_multilevel


@dataclass(frozen=True)
class MetaInfluence:
    """Case diagnostics in first occurrence order and residuals in input row order."""

    labels: tuple
    case_index: NDArray[np.intp]
    cooks_distance: NDArray[np.float64]
    dfbetas: NDArray[np.float64]
    deleted_coefficients: NDArray[np.float64]
    deleted_variances: NDArray[np.float64]
    variance_names: tuple[str, ...]
    deleted_residuals: NDArray[np.float64]
    residual_standard_errors: NDArray[np.float64]
    studentized_residuals: NDArray[np.float64]
    cluster_chi_squared: NDArray[np.float64]
    cluster_df: NDArray[np.intp]
    failed_cases: tuple


def _inverse_information(design, decomposition):
    _, _, _, upper = _gls(np.zeros(len(design)), design, decomposition)
    inverse = solve_triangular(upper, np.eye(design.shape[1]), check_finite=False)
    return inverse @ inverse.T


def meta_hatvalues(fit: MetaMultilevelResult, *, matrix: bool = False) -> NDArray[np.float64]:
    """Return the GLS leverage diagonal or the full fitted-value operator."""
    if not isinstance(fit, MetaMultilevelResult):
        raise ValueError("fit must be a MetaMultilevelResult")
    if not isinstance(matrix, (bool, np.bool_)):
        raise ValueError("matrix must be boolean")
    design = fit._design / np.max(abs(fit._design), axis=0)
    decomposition = cho_factor(fit.marginal_covariance, lower=True)
    weighted_design = cho_solve(decomposition, design)
    unweighted = design @ _inverse_information(design, decomposition)
    if matrix:
        return unweighted @ weighted_design.T
    return np.einsum("ij,ij->i", unweighted, weighted_design)


def _fixed_gls(effects, design, covariance, test):
    n, p = design.shape
    scale = np.max(abs(design), axis=0, initial=0)
    if n <= p or np.any(scale == 0):
        raise ValueError("deletion leaves too few observations or a constant zero design column")
    scaled = design / scale
    if np.linalg.matrix_rank(scaled) < p:
        raise ValueError("deletion leaves a rank-deficient meta-regression")
    beta, residual, _, upper = _gls(effects, scaled, cho_factor(covariance, lower=True))
    inverse = solve_triangular(upper, np.eye(p), check_finite=False) / scale[:, None]
    fixed_covariance = inverse @ inverse.T
    if test in ("knha", "adhoc"):
        rss = float(residual @ residual)
        adjust = 0.0 if rss <= np.finfo(float).eps else rss / (n - p)
        fixed_covariance *= max(adjust, 1.0) if test == "adhoc" else adjust
    return beta / scale, fixed_covariance


def _marginal_covariance(inputs, variances):
    covariance = inputs.sampling.copy()
    for name, labels in inputs.groups.items():
        covariance += variances[name] * (labels[:, None] == labels)
    for name, kernel in inputs.kernels.items():
        covariance += variances[name] * kernel
    return covariance


def _delete(fit, inputs, effects, keep, reestimate):
    if reestimate and fit._estimated_variances:
        subset = np.ix_(keep, keep)
        model = meta_multilevel(
            effects[keep], inputs.sampling[subset],
            moderators=fit._design[keep, int(inputs.intercept):], intercept=inputs.intercept,
            random_intercepts={name: labels[keep] for name, labels in inputs.groups.items()},
            covariance_components={name: kernel[subset] for name, kernel in inputs.kernels.items()},
            fixed_variances=inputs.fixed_variances, method=fit.method, test=inputs.test, dfs=inputs.dfs,
        )
        marginals = _marginal_covariance(inputs, model.variance_components)
        _, covariance = _fixed_gls(effects, fit._design, marginals, inputs.test)
        return model.coefficients, model.inference.covariance, model.variance_components, marginals, covariance

    beta, covariance = _fixed_gls(
        effects[keep], fit._design[keep], fit.marginal_covariance[np.ix_(keep, keep)],
        "z" if inputs is None else inputs.test,
    )
    return beta, covariance, fit.variance_components, fit.marginal_covariance, fit.inference.covariance


def _cases(cluster, size):
    if cluster is None:
        return tuple(range(size)), np.arange(size, dtype=np.intp)
    try:
        source = tuple(cluster) if not isinstance(cluster, (str, bytes)) else ()
    except TypeError as exc:
        raise ValueError("cluster must have one label per effect") from exc
    indices = _codes(source, size, "cluster")
    first = np.unique(indices, return_index=True)[1]
    labels = tuple(tuple(source[i]) if isinstance(source[i], (list, np.ndarray)) else source[i]
                   for i in first)
    return labels, indices


def meta_influence(
    fit: MetaMultilevelResult,
    *,
    cluster: ArrayLike | None = None,
    reestimate: bool = True,
    cook_coefficients: ArrayLike | None = None,
) -> MetaInfluence:
    """Delete each row or group and compute Cook distance and deleted residuals.

    By default variance components are re-estimated for each successful
    deletion. Set reestimate=False for conditional diagnostics holding the
    original marginal covariance fixed. An inestimable deletion returns
    NaN diagnostics while preserving the case and observation alignment.
    """
    if not isinstance(fit, MetaMultilevelResult):
        raise ValueError("fit must be a MetaMultilevelResult")
    if not isinstance(reestimate, (bool, np.bool_)):
        raise ValueError("reestimate must be boolean")
    inputs = fit._fit_inputs
    if reestimate and inputs is None:
        raise ValueError("re-estimation requires a fit created by meta_multilevel")
    design = fit._design
    n, p = design.shape
    labels, codes = _cases(cluster, n)
    if cook_coefficients is None:
        tested = np.arange(p)
    else:
        tested = np.asarray(cook_coefficients)
        if (tested.ndim != 1 or not len(tested) or tested.dtype.kind not in "iu"
                or np.any(tested < 0) or np.any(tested >= p) or len(set(tested)) != len(tested)):
            raise ValueError("cook_coefficients must contain unique in-range coefficient indices")
    cook_decomposition = cho_factor(fit.inference.covariance[np.ix_(tested, tested)], lower=True)
    effects = np.einsum("ij,j->i", design, fit.coefficients) + fit.residuals
    variance_names = tuple(fit.variance_components)
    k = len(labels)
    cook, chi_squared = np.full(k, np.nan), np.full(k, np.nan)
    dfbetas, betas = np.full((k, p), np.nan), np.full((k, p), np.nan)
    deleted_variance = np.full((k, len(variance_names)), np.nan)
    residual, se = np.full(n, np.nan), np.full(n, np.nan)
    failure = []
    for i, label in enumerate(labels):
        omitted = codes == i
        try:
            beta, coef_cov, variances, marginal, full_cov = _delete(fit, inputs, effects, ~omitted, reestimate)
            omitted_x = design[omitted]
            error = effects[omitted] - np.einsum("ij,j->i", omitted_x, beta)
            covariance = marginal[np.ix_(omitted, omitted)] + omitted_x @ coef_cov @ omitted_x.T
            factor = cho_factor(covariance, lower=True)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            failure.append(label)
            continue
        delta = fit.coefficients - beta
        betas[i] = beta
        deleted_variance[i] = [variances[name] for name in variance_names]
        with np.errstate(divide="ignore", invalid="ignore"):
            dfbetas[i] = delta / np.sqrt(np.diag(full_cov))
        cook[i] = delta[tested] @ cho_solve(cook_decomposition, delta[tested])
        residual[omitted], se[omitted] = error, np.sqrt(np.diag(covariance))
        chi_squared[i] = error @ cho_solve(factor, error)
    return MetaInfluence(
        labels, codes, cook, dfbetas, betas, deleted_variance, variance_names, residual, se,
        residual / se, chi_squared, np.bincount(codes), tuple(failure),
    )
