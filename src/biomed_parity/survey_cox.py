"""Survey-weighted Cox regression with first-stage Taylor linearization."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ._cox import _RiskSets, _fit_likelihood
from ._survival_inputs import _encode_labels, _event_vector, _fix_near_ties, _numeric_vector


@dataclass(frozen=True)
class SurveyCoxResult:
    coefficients: NDArray[np.float64]
    covariance: NDArray[np.float64]
    naive_covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    hazard_ratios: NDArray[np.float64]
    lower_hazard_ratios: NDArray[np.float64]
    upper_hazard_ratios: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    log_likelihood: float
    null_log_likelihood: float
    design_df: int
    residual_df: int


def _survey_covariance(linearized, strata, psu, fpc):
    dimension = linearized.shape[1]
    covariance = np.zeros((dimension, dimension))
    design_df = 0
    for stratum in np.unique(strata):
        mask = strata == stratum
        fractions = fpc[mask]
        fraction = float(fractions[0])
        if np.any(fractions != fraction):
            raise ValueError("fpc sampling fraction must be constant within each stratum")
        clusters, indices = np.unique(psu[mask], return_inverse=True)
        count = len(clusters)
        design_df += count - 1
        if fraction == 1:
            continue
        if count < 2:
            raise ValueError("a sampling stratum has only one non-certainty PSU")
        totals = np.zeros((count, dimension))
        np.add.at(totals, indices, linearized[mask])
        centered = totals - totals.mean(axis=0)
        covariance += ((1 - fraction) * count / (count - 1)) * (centered.T @ centered)
    return covariance, design_df


def svycoxph(
    time: ArrayLike,
    status: ArrayLike,
    covariates: ArrayLike,
    weights: ArrayLike,
    *,
    psu: ArrayLike | None = None,
    strata: ArrayLike | None = None,
    model_strata: ArrayLike | None = None,
    fpc: ArrayLike | None = None,
    ties: str = "efron",
    confidence_level: float = 0.95,
) -> SurveyCoxResult:
    """Fit a right-censored weighted Cox model with one-stage design covariance.

    Use one row per observation and strictly positive sampling weights. Supply
    sampling stratum and PSU IDs separately from baseline-hazard model strata.
    If supplied, fpc contains first-stage sampling fractions by observation.
    """
    times = _numeric_vector(time, "time")
    event = _event_vector(status)
    weight = _numeric_vector(weights, "weights")
    count = len(times)
    x = np.asarray(covariates)
    if x.ndim == 1:
        x = x[:, None]
    if (
        x.ndim != 2 or len(x) != count or x.shape[1] == 0
        or x.dtype.kind not in "buif" or not np.isfinite(x).all()
    ):
        raise ValueError("covariates must contain finite numeric columns aligned with time")
    if not (len(event) == len(weight) == count):
        raise ValueError("time, status and weights must all have the same length")
    if np.any(times < 0) or not event.any():
        raise ValueError("nonnegative follow-up and at least one event are required")
    if np.any(weight <= 0):
        raise ValueError("sampling weights must be strictly positive")
    if ties not in ("efron", "breslow"):
        raise ValueError("ties must be 'efron' or 'breslow'")
    if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    sampling_codes = np.zeros(count, dtype=int) if strata is None else _encode_labels(
        strata, None, count, "strata"
    )[1]
    psu_codes = np.arange(count) if psu is None else _encode_labels(psu, None, count, "psu")[1]
    model_codes = np.zeros(count, dtype=int) if model_strata is None else _encode_labels(
        model_strata, None, count, "model_strata"
    )[1]
    fractions = np.zeros(count) if fpc is None else _numeric_vector(fpc, "fpc")
    if len(fractions) != count or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("fpc must contain one sampling fraction in [0, 1] per observation")

    # Center and scale columns to keep Newton steps comparable across units.
    scales = x.std(axis=0)
    if np.any(scales == 0):
        raise ValueError("the Cox model requires nonconstant covariates and no intercept column")
    scaled = (x - x.mean(axis=0)) / scales
    if np.linalg.matrix_rank(scaled) < scaled.shape[1]:
        raise ValueError("covariates must form a full-rank design")
    weight = weight / weight.mean()
    times = _fix_near_ties(times)
    sets = [
        _RiskSets.build(times, event, scaled, weight, np.flatnonzero(model_codes == group))
        for group in np.unique(model_codes)
    ]
    dimension = x.shape[1]
    beta, likelihood, null, naive = _fit_likelihood(sets, dimension, ties)
    scores = np.zeros_like(scaled)
    for subset in sets:
        scores[subset.order] = subset.evaluate(beta, ties, linearize=True)
    coefficients = beta / scales
    linearized = (scores @ naive) / scales
    covariance, design_df = _survey_covariance(linearized, sampling_codes, psu_codes, fractions)
    standard_errors = np.sqrt(np.diag(covariance))
    critical = norm.isf((1 - confidence_level) / 2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pvalues = 2 * norm.sf(np.abs(coefficients / standard_errors))
        hazard_ratios = np.exp(coefficients)
        lower = np.exp(coefficients - critical * standard_errors)
        upper = np.exp(coefficients + critical * standard_errors)
    return SurveyCoxResult(
        coefficients, covariance, naive / np.outer(scales, scales), standard_errors,
        hazard_ratios, lower, upper, pvalues, float(likelihood), float(null),
        design_df, design_df - dimension + 1,
    )
