"""Estimate a structured outcome-level covariance in Gaussian meta-analysis."""

from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .meta_multilevel import (
    Array, MetaMultilevelResult, _codes, _covariance, _degrees_of_freedom, _design_matrix, _real_array,
    _require_estimable, _variance_objective, meta_multilevel,
)


@dataclass(frozen=True)
class MetaMultivariateResult:
    """Between-study covariance and model-based fixed-effect inference."""

    outcome_levels: tuple[str | Real, ...]
    structure: str
    random_covariance: Array
    random_correlation: Array
    fit: MetaMultilevelResult


def _outcomes(values, levels, n):
    try:
        labels = tuple(values) if not isinstance(values, (str, bytes)) else ()
        ordered = (
            tuple(dict.fromkeys(labels)) if levels is None
            else tuple(levels) if not isinstance(levels, (str, bytes)) else ()
        )
        lookup = {label: i for i, label in enumerate(ordered)}
        valid = all(
            isinstance(label, str) or isinstance(label, Real) and np.isfinite(label)
            for label in (*labels, *ordered)
        )
        if (
            not valid or len(labels) != n or len(lookup) != len(ordered)
            or lookup.keys() != set(labels)
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome_levels must list each observed outcome exactly once") from exc
    return ordered, np.array([lookup[label] for label in labels], dtype=np.intp)


class _Structure:
    def __init__(self, structure, size):
        self.structure = "ID" if size == 1 else structure
        self.size = size
        self.rows, self.cols = np.tril_indices(size)
        self.lags = np.abs(np.arange(size)[:, None] - np.arange(size))
        self.common_scale = self.structure in ("ID", "CS", "AR")
        self.correlated = self.structure in ("CS", "HCS", "AR", "HAR")
        self.autoregressive = self.structure in ("AR", "HAR")
        self.scale_count = 1 if self.common_scale else size
        if self.structure == "UN":
            self.bounds = [(0.0, None) if i == j else (None, None)
                           for i, j in zip(self.rows, self.cols, strict=True)]
        else:
            self.bounds = [(0.0, None)] * self.scale_count
            if self.correlated:
                self.bounds.append((-1.0 if self.autoregressive else -1.0 / (size - 1), 1.0))

    def start(self, scale, correlation):
        sd = scale * np.linspace(0.8, 1.2, self.size)
        if self.structure == "UN":
            initial = np.diag(sd) + correlation * scale * np.tril(np.ones_like(self.lags), -1)
            return initial[self.rows, self.cols]
        start = np.array([scale]) if self.common_scale else sd
        return np.r_[start, max(correlation, self.bounds[-1][0] / 2)] if self.correlated else start

    def evaluate(self, parameters):
        """Return the covariance and its derivatives with respect to the fitted parameters."""
        n = self.size
        if self.structure == "UN":
            lower = np.zeros((n, n))
            lower[self.rows, self.cols] = parameters
            tensor = np.zeros((len(parameters), n, n))
            tensor[np.arange(len(parameters)), self.rows, :] = lower[:, self.cols].T
            return lower @ lower.T, tensor + tensor.transpose(0, 2, 1)

        corr = np.eye(n)
        sd = np.repeat(parameters[0], n) if self.common_scale else parameters[:n]
        derivatives = np.zeros((len(parameters), n, n))
        if self.correlated:
            rho = parameters[-1]
            off_diag = self.lags > 0
            powers = self.lags[off_diag] if self.autoregressive else np.ones(off_diag.sum(), dtype=int)
            corr[off_diag] = rho**powers
            derivatives[-1, off_diag] = powers * rho**(powers - 1) * np.outer(sd, sd)[off_diag]
        if self.common_scale:
            derivatives[0] = 2 * parameters[0] * corr
        else:
            diagonal = np.arange(n)
            derivatives[diagonal, diagonal, :] = corr * sd
            derivatives[:n] += derivatives[:n].transpose(0, 2, 1).copy()
        return corr * np.outer(sd, sd), derivatives


def meta_multivariate(
    effects: ArrayLike,
    sampling_covariance: ArrayLike,
    studies: ArrayLike,
    outcomes: ArrayLike,
    *,
    outcome_levels: ArrayLike | None = None,
    structure: Literal["UN", "CS", "HCS", "ID", "DIAG", "AR", "HAR"] = "UN",
    moderators: ArrayLike | None = None,
    intercept: bool = True,
    method: Literal["ML", "REML"] = "REML",
    test: Literal["z", "t", "knha", "adhoc"] = "z",
    dfs: Literal["residual", "contain"] | ArrayLike = "residual",
    confidence_level: float = 0.95,
    tested_coefficients: ArrayLike | None = None,
) -> MetaMultivariateResult:
    """Fit one correlated outcome random-effect term within independent studies.

    Effects in the same study and outcome share a random effect. The full
    sampling covariance is supplied separately and may have cross-study
    entries. AR and HAR use the ordered levels, equally spaced by design.
    By default outcomes are ordered by first occurrence in the inputs.
    """
    y = _real_array(effects, "effects")
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("effects must be a vector with at least two observations")
    if structure not in ("UN", "CS", "HCS", "ID", "DIAG", "AR", "HAR"):
        raise ValueError("structure must be UN, CS, HCS, ID, DIAG, AR, or HAR")
    if method not in ("ML", "REML"):
        raise ValueError("method must be ML or REML")
    size = len(y)
    design, design_scale = _design_matrix(moderators, size, intercept)
    sampling = _covariance(sampling_covariance, size, "sampling_covariance", vector=True)
    study = _codes(studies, size, "studies")
    levels, outcome = _outcomes(outcomes, outcome_levels, size)
    model = _Structure(structure, len(levels))
    within_study = study[:, None] == study
    if model.autoregressive:
        observed_lags = model.lags[outcome[:, None], outcome][within_study]
        if not np.any(observed_lags % 2):
            raise ValueError("the AR correlation sign is not identifiable without an observed odd lag")
    rows, cols = model.rows, model.cols
    outcome_high = np.maximum(outcome[:, None], outcome)
    outcome_low = np.minimum(outcome[:, None], outcome)
    components = (
        within_study & (outcome_high[None, :, :] == rows[:, None, None])
        & (outcome_low[None, :, :] == cols[:, None, None])
    ).astype(float)
    objective, scale = _variance_objective(y, sampling, components, design / design_scale, method)

    def loss(parameters):
        covariance, jacobian = model.evaluate(parameters)
        value, score = objective.evaluate(covariance[rows, cols], gradient=True)
        return value, jacobian[:, rows, cols] @ score

    _, derivative = model.evaluate(model.start(0.7, 0.35))
    linearization = derivative[:, rows, cols]
    _require_estimable(
        np.einsum("ij,jkl->ikl", linearization, objective.components),
        np.einsum("ij,jkl->ikl", linearization, components),
    )
    best = None
    for start in (model.start(0.4, 0), model.start(0.8, 0.35), model.start(1.1, -0.2)):
        result = minimize(loss, start, method="L-BFGS-B", jac=True, bounds=model.bounds,
                          options={"ftol": 5e-15, "gtol": 2e-9, "maxiter": 800, "maxls": 50})
        score = result.jac.copy()
        for index, (lo, hi) in enumerate(model.bounds):
            if result.x[index] == lo:
                score[index] = min(score[index], 0)
            elif result.x[index] == hi:
                score[index] = max(score[index], 0)
        if np.isfinite(result.fun) and np.max(np.abs(score)) < 2e-5:
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise ValueError("outcome covariance optimization did not converge")

    random_covariance = model.evaluate(best.x)[0] * scale
    sd = np.sqrt(random_covariance.diagonal())
    denominator = np.outer(sd, sd)
    correlation = np.divide(
        random_covariance, denominator, out=np.full_like(random_covariance, np.nan),
        where=denominator > 0,
    )
    random_kernel = within_study * random_covariance[outcome[:, None], outcome]
    degrees = dfs
    if test != "z" and isinstance(dfs, str) and dfs == "contain":
        study_outcome = _codes(zip(study, outcome, strict=True), size, "study outcomes")
        degrees = _degrees_of_freedom(dfs, design, [study, study_outcome])
    fit = meta_multilevel(
        y, sampling, moderators=moderators, intercept=intercept, method=method, test=test, dfs=degrees,
        confidence_level=confidence_level, tested_coefficients=tested_coefficients,
        covariance_components={"outcomes": random_kernel}, fixed_variances={"outcomes": 1.0},
    )
    fit = replace(
        fit, variance_components={}, _estimated_variances=len(best.x),
        _fit_inputs=replace(fit._fit_inputs, reestimation_supported=False),
    )
    return MetaMultivariateResult(levels, structure, random_covariance, correlation, fit)
