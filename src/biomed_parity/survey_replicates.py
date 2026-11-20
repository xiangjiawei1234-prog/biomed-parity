"""Design-based means and totals from prespecified survey replicate weights."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.blas import dger, dnrm2


@dataclass(frozen=True)
class SurveyReplicateResult:
    """Estimates indexed by outcome and replicate estimates indexed by replicate first."""

    estimate: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    replicate_estimates: NDArray[np.float64]
    variance_center: NDArray[np.float64]
    design_df: float


@dataclass(frozen=True, eq=False)
class SurveyReplicateDesign:
    """Prespecified replicate weights, constructed by :func:`svrepdesign`."""

    sampling_weights: NDArray[np.float64]
    replicate_weights: NDArray[np.float64]
    type: str
    scale: float
    rscales: NDArray[np.float64]
    combined_weights: bool
    mse: bool
    _df_override: float | None

    @property
    def analysis_weights(self) -> NDArray[np.float64]:
        """Replicate weights on the scale used in the estimators."""
        if self.combined_weights:
            return self.replicate_weights
        return self.replicate_weights * self.sampling_weights[:, None]

    @cached_property
    def design_df(self) -> float:
        """Numerical rank of analysis weights minus one, or the user-supplied value."""
        if self._df_override is not None:
            return self._df_override
        return _replicate_rank(self.analysis_weights) - 1

    def subset(self, mask: ArrayLike) -> SurveyReplicateDesign:
        """Restrict to a survey domain while keeping the variance scaling and all replicates."""
        mask = np.asarray(mask)
        if mask.shape != self.sampling_weights.shape or mask.dtype.kind != "b" or not mask.any():
            raise ValueError("mask must be Boolean with one entry per row and at least one selected row")
        return SurveyReplicateDesign(
            _readonly(self.sampling_weights[mask]), _readonly(self.replicate_weights[mask]),
            self.type, self.scale, self.rscales, self.combined_weights, self.mse, self._df_override,
        )

    def mean(self, values: ArrayLike, *, na_rm: bool = False) -> SurveyReplicateResult:
        """Estimate weighted means of one or more numeric outcomes."""
        return self._summarize(values, na_rm=na_rm, mean=True)

    def total(self, values: ArrayLike, *, na_rm: bool = False) -> SurveyReplicateResult:
        """Estimate weighted totals of one or more numeric outcomes."""
        return self._summarize(values, na_rm=na_rm, mean=False)

    def _summarize(self, values, *, na_rm, mean):
        data = np.asarray(values, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] != self.sampling_weights.size or data.shape[1] == 0:
            raise ValueError("values must have one row per sampled unit and at least one outcome")
        if np.isinf(data).any():
            raise ValueError("values must not contain infinities")
        observed = ~np.isnan(data).any(axis=1)
        design = self
        if not observed.all():
            if not na_rm:
                raise ValueError("values contain missing data; use na_rm=True to remove incomplete rows")
            if not observed.any():
                raise ValueError("no complete rows remain after removing missing data")
            design = self.subset(observed)
            data = data[observed]

        analysis_weights = design.analysis_weights
        estimate = design.sampling_weights @ data
        replicates = analysis_weights.T @ data
        valid = np.ones(replicates.shape[0], dtype=bool)
        if mean:
            denominator = design.sampling_weights.sum()
            if denominator == 0:
                raise ValueError("sampling weights must sum to a nonzero value for a mean")
            estimate /= denominator
            replicate_sums = analysis_weights.sum(axis=0)
            valid = replicate_sums != 0
            replicates[~valid] = np.nan
            np.divide(replicates, replicate_sums[:, None], out=replicates, where=valid[:, None])
            if not valid.any():
                raise ValueError("all replicate weight sums are zero for this mean")
            if not valid.all():
                warnings.warn(
                    f"{np.count_nonzero(~valid)} replicates with zero weight sums were discarded",
                    RuntimeWarning, stacklevel=3,
                )

        usable_replicates = replicates[valid]
        if not np.isfinite(estimate).all() or not np.isfinite(usable_replicates).all():
            raise ValueError("weighted estimates must be finite")
        rscales = design.rscales[valid]
        center = estimate
        if not design.mse:
            active = usable_replicates[rscales > 0]
            center = active.mean(axis=0) if active.size else np.full(estimate.shape, np.nan)
        deviations = (usable_replicates - center) * np.sqrt(rscales[:, None])
        covariance = design.scale * (deviations.T @ deviations)
        return SurveyReplicateResult(
            estimate, covariance, np.sqrt(np.diag(covariance)), replicates, center, design.design_df,
        )


def _readonly(array):
    array.setflags(write=False)
    return array


def _replicate_rank(weights):
    # R's rank tolerance is relative to each original column norm. Process
    # columns in input order, skipping dependencies before adding a direction.
    residual = np.array(weights, order="F", copy=True)
    original_norms = [dnrm2(column) for column in residual.T]
    rank = 0
    for index, original_norm in enumerate(original_norms):
        column = residual[:, index]
        norm = dnrm2(column)
        if norm == 0 or norm < 1e-5 * original_norm:
            continue
        rank += 1
        remaining = residual[:, index + 1:]
        if rank == weights.shape[0] or not remaining.size:
            break
        direction = column / norm
        dger(-1, direction, direction @ remaining, a=remaining, overwrite_a=True)
    return rank


def _scalar(value, name):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _variance_scales(repweights, type, combined, scale, rscales, rho, bootstrap_average, fpc, fpctype):
    n_replicates = repweights.shape[1]
    if type == "BRR":
        scale = 1 / n_replicates
    elif type == "Fay":
        rho = _scalar(rho, "rho")
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1) for Fay replicate weights")
        scale = 1 / (n_replicates * (1 - rho) ** 2)
    elif type in ("ACS", "successive-difference", "JK2"):
        scale = 1 if type == "JK2" else 4 / n_replicates
        rscales = None
    elif type == "bootstrap" and scale is None:
        average = 1 if bootstrap_average is None else _scalar(bootstrap_average, "bootstrap_average")
        if n_replicates < 2 or average <= 0:
            raise ValueError("default bootstrap scale requires at least two replicates and a positive bootstrap_average")
        scale = average / (n_replicates - 1)
    elif type == "JK1" and scale is None:
        largest = repweights[:, 0].max()
        if not combined and largest <= 0:
            raise ValueError("provide scale for JK1 when the first replicate has no positive weights")
        scale = (n_replicates - 1) / n_replicates if combined else 1 / largest

    if type == "JKn" and rscales is None:
        if combined:
            raise ValueError("rscales is required when using combined JKn weights")
        largest = repweights.max(axis=0)
        if np.any(largest <= 0):
            raise ValueError("provide rscales for JKn when a replicate has no positive weights")
        rscales = 1 / largest

    scale = 1.0 if scale is None else _scalar(scale, "scale")
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    rscales = np.array(1 if rscales is None else rscales, dtype=float, copy=True)
    if rscales.shape in ((), (1,)):
        rscales = np.full(n_replicates, rscales.item())
    if rscales.shape != (n_replicates,) or not np.isfinite(rscales).all() or np.any(rscales < 0):
        raise ValueError("rscales must be finite and nonnegative with one entry or one per replicate")

    if fpc is not None:
        if type not in ("JK1", "JKn", "other"):
            raise ValueError("fpc is only available for JK1, JKn and other replicate weights")
        if fpctype not in ("fraction", "correction"):
            raise ValueError("fpctype must be 'fraction' or 'correction' when fpc is supplied")
        fpc = np.asarray(fpc, dtype=float)
        if fpc.shape != (n_replicates,) or not np.isfinite(fpc).all() or np.any((fpc < 0) | (fpc > 1)):
            raise ValueError("fpc must have one value in [0, 1] per replicate")
        rscales *= 1 - fpc if fpctype == "fraction" else fpc
    return scale, rscales


def svrepdesign(
    repweights: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    type: str = "BRR",
    combined_weights: bool = True,
    rho: float | None = None,
    bootstrap_average: float | None = None,
    scale: float | None = None,
    rscales: ArrayLike | None = None,
    fpc: ArrayLike | None = None,
    fpctype: str | None = None,
    mse: bool | None = None,
    degf: float | None = None,
) -> SurveyReplicateDesign:
    """Build a survey design from a row-by-replicate matrix of published weights.

    The full-sample sampling weights default to one per row. Set
    ``combined_weights=False`` for a matrix of factors that must be multiplied
    by the sampling weights. Supply the survey's published scaling options when
    applicable. The default variance center is the mean of the positive-scale
    replicates, or the full-sample estimate for ``mse=True`` and by default for ACS.
    """
    if type not in ("BRR", "Fay", "JK1", "JKn", "bootstrap", "ACS", "successive-difference", "JK2", "other"):
        raise ValueError("type must be BRR, Fay, JK1, JKn, bootstrap, ACS, successive-difference, JK2 or other")
    if not isinstance(combined_weights, (bool, np.bool_)) or (
        mse is not None and not isinstance(mse, (bool, np.bool_))
    ):
        raise ValueError("combined_weights and mse must be Boolean")
    repweights = np.array(repweights, dtype=float, copy=True)
    if repweights.ndim != 2 or not repweights.size or not np.isfinite(repweights).all():
        raise ValueError("repweights must be a finite matrix with at least one row and one replicate")
    n_rows = repweights.shape[0]
    weights = np.ones(n_rows) if weights is None else np.array(weights, dtype=float, copy=True)
    if weights.shape != (n_rows,) or not np.isfinite(weights).all():
        raise ValueError("weights must have one finite value per row of repweights")
    if degf is not None:
        degf = _scalar(degf, "degf")

    scale, rscales = _variance_scales(
        repweights, type, combined_weights, scale, rscales, rho, bootstrap_average, fpc, fpctype,
    )
    return SurveyReplicateDesign(
        _readonly(weights), _readonly(repweights), type, scale, _readonly(rscales), bool(combined_weights),
        type == "ACS" if mse is None else bool(mse), degf,
    )
