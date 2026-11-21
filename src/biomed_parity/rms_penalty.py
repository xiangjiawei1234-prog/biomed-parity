"""Grid tuning for quadratically penalized binary logistic regression."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._logistic import _information
from .rms_logistic import PenalizedLogisticFit, _finite_array, fit_penalized_logistic

_TERM_TYPES = ("simple", "nonlinear", "interaction", "nonlinear.interaction")
PenaltyCriterion = Literal["aicc", "aic", "bic"]


@dataclass(frozen=True)
class LogisticPenaltyTrace:
    """One score per grid row and the fit with the highest selected score."""

    penalty_names: tuple[str, ...]
    penalties: NDArray[np.float64]
    effective_df: NDArray[np.float64]
    likelihood_ratio: NDArray[np.float64]
    aic: NDArray[np.float64]
    bic: NDArray[np.float64]
    aicc: NDArray[np.float64]
    criterion: PenaltyCriterion
    best_index: int
    objective: float
    best_fit: PenalizedLogisticFit
    adjusted_covariance: NDArray[np.float64]
    coefficients: NDArray[np.float64] | None

    @property
    def best_penalty(self) -> float | dict[str, float]:
        values = self.penalties[self.best_index]
        if self.penalty_names == ("penalty",):
            return float(values[0])
        return dict(zip(self.penalty_names, values.tolist(), strict=True))


def _penalty_sequence(values):
    values = np.atleast_1d(_finite_array(values, "penalties"))
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("each penalty sequence must contain nonnegative values")
    return values


def _penalty_grid(penalties, term_types, p, include_zero, complex_more):
    if not isinstance(penalties, Mapping):
        if term_types is not None:
            raise ValueError("term_types applies to named penalty grids")
        values = _penalty_sequence(penalties)
        if include_zero:
            values = np.r_[0, values[values > 0]]
        return ("penalty",), values[:, None], values

    names = tuple(penalties)
    if not names or any(name not in _TERM_TYPES for name in names):
        raise ValueError(f"named penalties must use term types {_TERM_TYPES}")
    axes = [_penalty_sequence(penalties[name]) for name in names]
    if any(not len(axis) for axis in axes):
        raise ValueError("each named penalty sequence must be nonempty")
    term_types = ("simple",) * p if term_types is None else tuple(term_types)
    if len(term_types) != p or any(term not in _TERM_TYPES for term in term_types):
        raise ValueError("term_types must name one recognized type per slope column")

    # R expand.grid increments its first input vector fastest.
    grid = np.asarray([row[::-1] for row in product(*axes[::-1])])
    if complex_more:
        grid = grid[np.all(grid[:, 1:] >= grid[:, :-1], axis=1)]
    factors = {"simple": np.zeros(len(grid))}
    factors.update(zip(names, grid.T, strict=True))
    for previous, term in zip(_TERM_TYPES, _TERM_TYPES[1:]):
        factors.setdefault(term, factors[previous])
    return names, grid, np.column_stack([factors[term] for term in term_types])


def trace_logistic_penalties(
    x: ArrayLike,
    y: ArrayLike,
    penalties: ArrayLike | Mapping[str, ArrayLike],
    *,
    term_types: Sequence[str] | None = None,
    penalty_matrix: ArrayLike | None = None,
    categorical_blocks: Sequence[Sequence[int]] = (),
    criterion: PenaltyCriterion = "aicc",
    include_zero: bool = True,
    complex_more: bool = True,
    keep_coefficients: bool = False,
) -> LogisticPenaltyTrace:
    """Tune binary logistic penalties on a scalar or named term-type grid.

    Scalar grids include the zero penalty by default. Named axes form a
    Cartesian grid in input order. Unspecified term types inherit the
    preceding type's factor, starting with zero for simple terms.
    The criterion is maximized in the likelihood-ratio convention of
    rms::pentrace. Exact ties select the first grid row.
    """
    x = _finite_array(x, "x")
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must have at least one slope column")
    if not isinstance(criterion, str) or criterion not in ("aicc", "aic", "bic"):
        raise ValueError("criterion must be 'aicc', 'aic', or 'bic'")
    flags = (include_zero, complex_more, keep_coefficients)
    if not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValueError("include_zero, complex_more, and keep_coefficients must be boolean")

    n, p = x.shape
    names, grid, factors = _penalty_grid(penalties, term_types, p, include_zero, complex_more)
    if not len(grid):
        raise ValueError("the penalty grid contains no candidates")
    df, lr, aic, bic, aicc = np.empty((5, len(grid)))
    objectives = {"aicc": aicc, "aic": aic, "bic": bic}[criterion]
    coefficients = np.empty((len(grid), p + 1)) if keep_coefficients else None
    best_fit = None
    objective = -np.inf
    for i, factor in enumerate(factors):
        fit = fit_penalized_logistic(
            x, y, penalty=factor, penalty_matrix=penalty_matrix, categorical_blocks=categorical_blocks,
        )
        df[i], lr[i] = fit.effective_df, fit.likelihood_ratio
        aic[i] = lr[i] - 2 * df[i]
        bic[i] = lr[i] - np.log(n) * df[i]
        remaining = n - df[i] - 1
        aicc[i] = lr[i] - 2 * df[i] * (1 + (df[i] + 1) / remaining) if remaining > 0 else -np.inf
        if coefficients is not None:
            coefficients[i] = fit.coefficients
        if objectives[i] > objective:
            best_index, objective, best_fit = i, objectives[i], fit
    if best_fit is None:
        raise ValueError(f"no penalty candidate has a finite {criterion} score")

    design = np.column_stack((np.ones(n), x))
    design_times_covariance = np.einsum("ij,jk->ik", design, best_fit.covariance)
    adjusted = _information(design_times_covariance, best_fit.linear_predictor)
    return LogisticPenaltyTrace(
        names, grid, df, lr, aic, bic, aicc, criterion, best_index, float(objective),
        best_fit, adjusted, coefficients,
    )
