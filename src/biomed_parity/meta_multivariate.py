"""Estimate a structured outcome-level covariance in Gaussian meta-analysis."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .meta_multilevel import (
    Array, MetaMultilevelResult, _codes, _covariance, _degrees_of_freedom, _design_matrix, _real_array,
    _random_components, _require_estimable, _variance_objective, meta_multilevel,
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
    def __init__(self, structure, levels, outcome, within_study):
        size = len(levels)
        self.structure = "ID" if size == 1 else structure
        self.size = size
        self.rows, self.cols = np.tril_indices(size)
        self.lags = np.abs(np.arange(size)[:, None] - np.arange(size))
        self.common_scale = self.structure in ("ID", "CS", "AR", "CAR")
        self.correlated = self.structure in ("CS", "HCS", "AR", "HAR", "CAR")
        self.autoregressive = self.structure in ("AR", "HAR")
        if structure == "CAR":
            times = _real_array(levels, "CAR outcome times")
            if len(np.unique(times)) != size:
                raise ValueError("CAR outcome times must be distinct at floating-point precision")
            if size > 1:
                with np.errstate(over="ignore"):
                    distances = np.abs(times[:, None] - times)
                    observed = distances[outcome[:, None], outcome][within_study]
                    positive = observed[observed > 0]
                    self.lags = distances / (np.min(positive) if len(positive) else 1.0)
                if not np.isfinite(self.lags).all():
                    raise ValueError("CAR outcome times exceed the supported numerical range")
        self.scale_count = 1 if self.common_scale else size
        if self.structure == "UN":
            self.bounds = [(0.0, None) if i == j else (None, None)
                           for i, j in zip(self.rows, self.cols, strict=True)]
        else:
            self.bounds = [(0.0, None)] * self.scale_count
            if self.structure == "CAR":
                self.bounds.append((0.0, None))
            elif self.correlated:
                self.bounds.append((-1.0 if self.autoregressive else -1.0 / (size - 1), 1.0))

    def start(self, scale, correlation):
        sd = scale * np.linspace(0.8, 1.2, self.size)
        if self.structure == "UN":
            initial = np.diag(sd) + correlation * scale * np.tril(np.ones_like(self.lags), -1)
            return initial[self.rows, self.cols]
        start = np.array([scale]) if self.common_scale else sd
        if self.structure == "CAR":
            return np.r_[start, 0.7 - correlation]
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
        if self.structure == "CAR":
            off_diag = self.lags > 0
            distance = self.lags[off_diag]
            with np.errstate(over="ignore", under="ignore"):
                corr[off_diag] = np.exp(-parameters[-1] * distance)
                derivatives[-1, off_diag] = -distance * corr[off_diag] * parameters[0]**2
        elif self.correlated:
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
    structure: Literal["UN", "CS", "HCS", "ID", "DIAG", "AR", "HAR", "CAR"] = "UN",
    moderators: ArrayLike | None = None,
    intercept: bool = True,
    random_intercepts: Mapping[str, ArrayLike] | None = None,
    covariance_components: Mapping[str, ArrayLike] | None = None,
    fixed_variances: Mapping[str, float] | None = None,
    method: Literal["ML", "REML"] = "REML",
    test: Literal["z", "t", "knha", "adhoc"] = "z",
    dfs: Literal["residual", "contain"] | ArrayLike = "residual",
    confidence_level: float = 0.95,
    tested_coefficients: ArrayLike | None = None,
) -> MetaMultivariateResult:
    """Fit one correlated outcome random-effect term within independent studies.

    Effects in the same study and outcome share a random effect. The full
    sampling covariance is supplied separately and may have cross-study
    entries. Independent random-intercept terms and scaled known covariance
    kernels can be included as in meta_multilevel. AR and HAR use the ordered levels.
    CAR uses the absolute differences between numeric outcome times.
    By default outcomes are ordered by first occurrence in the inputs.
    """
    y = _real_array(effects, "effects")
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("effects must be a vector with at least two observations")
    if structure not in ("UN", "CS", "HCS", "ID", "DIAG", "AR", "HAR", "CAR"):
        raise ValueError("structure must be UN, CS, HCS, ID, DIAG, AR, HAR, or CAR")
    if method not in ("ML", "REML"):
        raise ValueError("method must be ML or REML")
    size = len(y)
    design, design_scale = _design_matrix(moderators, size, intercept)
    sampling = _covariance(sampling_covariance, size, "sampling_covariance", vector=True)
    study = _codes(studies, size, "studies")
    levels, outcome = _outcomes(outcomes, outcome_levels, size)
    within_study = study[:, None] == study
    model = _Structure(structure, levels, outcome, within_study)
    if model.autoregressive:
        observed_lags = model.lags[outcome[:, None], outcome][within_study]
        if not np.any(observed_lags % 2):
            raise ValueError("the AR correlation sign is not identifiable without an observed odd lag")
    rows, cols = model.rows, model.cols
    outcome_high = np.maximum(outcome[:, None], outcome)
    outcome_low = np.minimum(outcome[:, None], outcome)
    outcome_components = (
        within_study & (outcome_high[None, :, :] == rows[:, None, None])
        & (outcome_low[None, :, :] == cols[:, None, None])
    ).astype(float)
    names, groups, extra, fixed = _random_components(
        size, random_intercepts, covariance_components, fixed_variances
    )
    components = np.concatenate((outcome_components, extra)) if names else outcome_components
    objective, scale = _variance_objective(y, sampling, components, design / design_scale, method)
    count = len(model.bounds)
    cell_count = len(rows)
    free = np.isnan(fixed)
    scales = fixed / scale
    bounds = model.bounds + [(0.0, None)] * int(free.sum())

    def loss(parameters):
        covariance, jacobian = model.evaluate(parameters[:count])
        additional = scales.copy()
        additional[free] = parameters[count:]
        value, score = objective.evaluate(np.r_[covariance[rows, cols], additional], gradient=True)
        return value, np.r_[jacobian[:, rows, cols] @ score[:cell_count], score[cell_count:][free]]

    _, derivative = model.evaluate(model.start(0.7, 0.35))
    linearization = derivative[:, rows, cols]

    def tangent_kernels(kernels):
        tangent = np.einsum("ij,jkl->ikl", linearization, kernels[:cell_count])
        return np.concatenate((tangent, kernels[cell_count:][free])) if np.any(free) else tangent

    _require_estimable(tangent_kernels(objective.components), tangent_kernels(components))
    best = None
    starts = [(0.4, 0, 0.2), (0.8, 0.35, 0), (1.1, -0.2, 0.8)]
    for sd, rho, extra_start in starts:
        start = np.r_[model.start(sd, rho), np.full(free.sum(), extra_start)]
        result = minimize(loss, start, method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"ftol": 5e-15, "gtol": 2e-9, "maxiter": 800, "maxls": 50})
        score = result.jac.copy()
        for index, (lo, hi) in enumerate(bounds):
            if result.x[index] == lo:
                score[index] = min(score[index], 0)
            elif result.x[index] == hi:
                score[index] = max(score[index], 0)
        if np.isfinite(result.fun) and np.max(np.abs(score)) < 2e-5:
            if best is None or result.fun < best.fun:
                best = result
    best_value = np.inf if best is None else best.fun
    fitted = None if best is None else model.evaluate(best.x[:count])[0]
    if best is not None:
        scales[free] = best.x[count:]
    if model.structure == "CAR":
        # Zero correlation is the infinite-decay limit. Compare the exact
        # independent fit as a candidate instead of relying on an arbitrary cap.
        diagonal = objective.components[:cell_count][rows == cols].sum(axis=0)[None]
        independent = replace(
            objective, components=np.concatenate((diagonal, objective.components[cell_count:]))
        )
        try:
            variance, value = independent.fit(np.r_[np.nan, fixed / scale])
        except ValueError:
            value = np.inf
        if np.isfinite(value) and value <= best_value + 1e-12:
            fitted = np.eye(model.size) * variance[0]
            scales = variance[1:]
    if fitted is None:
        raise ValueError("outcome covariance optimization did not converge")

    random_covariance = fitted * scale
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
        degrees = _degrees_of_freedom(dfs, design, [study, study_outcome, *groups.values()])
    outcome_term = "outcomes"
    while outcome_term in names:
        outcome_term = "_" + outcome_term
    kernels = {name: extra[i] for i, name in enumerate(names) if name not in groups}
    variances = dict(zip(names, map(float, scales * scale), strict=True))
    fit = meta_multilevel(
        y, sampling, moderators=moderators, intercept=intercept, method=method, test=test, dfs=degrees,
        confidence_level=confidence_level, tested_coefficients=tested_coefficients,
        random_intercepts=groups, covariance_components={**kernels, outcome_term: random_kernel},
        fixed_variances={**variances, outcome_term: 1.0},
    )
    fit = replace(
        fit, variance_components=variances, _estimated_variances=len(bounds),
        _fit_inputs=replace(fit._fit_inputs, reestimation_supported=False),
    )
    return MetaMultivariateResult(levels, structure, random_covariance, correlation, fit)
