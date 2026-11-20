"""Delete-one-PSU jackknife replicate weights for one-stage surveys."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ._survival_inputs import _encode_labels, _numeric_vector
from .survey_replicates import SurveyReplicateDesign, svrepdesign


def _codes(values, n, name):
    if np.shape(values) != (n,):
        raise ValueError(f"{name} must have one label per sampled row")
    return _encode_labels(values, None, n, name)


def _first_appearance(ids):
    return tuple(dict.fromkeys(ids))


def _corrections(fpc, fpctype, labels, row_groups, psu_counts, n):
    if fpctype not in ("fraction", "population", "correction"):
        raise ValueError("fpctype must be 'fraction', 'population' or 'correction'")
    if fpc is None:
        return np.ones(len(labels)), np.zeros(len(labels), dtype=bool)
    if isinstance(fpc, Mapping):
        if set(fpc) != set(labels):
            raise ValueError("fpc must give a value for each stratum label")
        values = _numeric_vector([fpc[label] for label in labels], "fpc")
    else:
        values = _numeric_vector(np.atleast_1d(fpc), "fpc")
        if values.size == 1:
            values = np.repeat(values, len(labels))
        elif values.size == n:
            by_row = values
            values = np.array([by_row[rows[0]] for rows in row_groups])
            if any(np.any(by_row[rows] != value) for rows, value in zip(row_groups, values, strict=True)):
                raise ValueError("fpc must be constant within each stratum")
        elif values.size != len(labels):
            raise ValueError("fpc must be scalar or have one value per stratum or sampled row")

    if fpctype == "population":
        if np.any(values < psu_counts):
            raise ValueError("fpc population size must be at least the number of sampled PSUs in its stratum")
        return (values - psu_counts) / values, np.zeros(len(labels), dtype=bool)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("fpc fractions and corrections must be in [0, 1]")
    if fpctype == "correction":
        return values, np.zeros(len(labels), dtype=bool)
    # In survey's explicit fraction path, fpc=0 takes a special branch for a
    # single-PSU stratum and retains a replicate with variance multiplier zero.
    return 1 - values, values == 0


def _jkn_factors(n_rows, labels, row_groups, psu_codes, psu_counts, corrections, zero_fraction, lonely_psu):
    retained = []
    n_strata = len(labels)
    for i, (label, count, correction) in enumerate(zip(labels, psu_counts, corrections, strict=True)):
        if correction == 0:
            continue
        adjust = count == 1 and not zero_fraction[i]
        if adjust:
            if lonely_psu == "fail":
                raise ValueError(f"stratum {label!r} has only one noncertainty PSU")
            if lonely_psu != "adjust":
                continue
            if n_strata < 2:
                raise ValueError("adjusting a lonely PSU requires at least two strata")
        retained.append((i, count, correction, adjust))

    n_replicates = sum(count for _, count, _, _ in retained)
    if n_replicates == 0:
        raise ValueError("no jackknife replicates remain after omitting fully sampled or lonely strata")
    factors = np.ones((n_rows, n_replicates))
    rscales = np.zeros(n_replicates)
    column = 0
    for group, count, correction, adjust in retained:
        if adjust:
            factors[:, column] = n_strata / (n_strata - 1)
            factors[row_groups[group], column] = 0
            rscales[column] = (n_strata - 1) / n_strata
        elif count == 1:
            factors[row_groups[group], column] = 0
        else:
            deleted = np.arange(count)
            factors[np.ix_(row_groups[group], column + deleted)] = (
                psu_codes[group][:, None] != deleted
            ) * (count / (count - 1))
            rscales[column:column + count] = correction * (count - 1) / count
        column += count
    scale = psu_counts.sum() / n_replicates if lonely_psu == "average" else 1.0
    return factors, float(scale), rscales


def as_svrepdesign(
    weights: ArrayLike,
    *,
    psu: ArrayLike | None = None,
    strata: ArrayLike | None = None,
    type: str = "auto",
    fpc: ArrayLike | Mapping[str | float, float] | None = None,
    fpctype: str = "fraction",
    lonely_psu: str = "fail",
    mse: bool = False,
) -> SurveyReplicateDesign:
    """Create JK1 or JKn replicate factors for an existing one-stage design.

    Supply a sampling weight per row. With no ``psu``, every row is a PSU.
    PSU labels are nested within strata. ``type="auto"`` uses JK1 when
    ``strata`` is omitted and JKn when it is supplied. Columns follow
    first appearance of strata and then of PSUs within each stratum.

    ``fpc`` is a scalar, a vector in first-appearance stratum order, a
    stratum-label mapping, or a row-aligned vector constant within strata.
    It describes first-stage sampling fractions by default. ``fpctype`` can
    instead denote PSU population sizes or direct variance corrections.
    Exhaustively sampled strata contribute their original weight to all
    retained JKn replicates. Use ``lonely_psu`` to select survey's policy
    for noncertainty strata with a single PSU.
    """
    weights = _numeric_vector(weights, "weights")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative with at least one positive value")
    if type not in ("auto", "JK1", "JKn"):
        raise ValueError("type must be 'auto', 'JK1' or 'JKn'")
    if lonely_psu not in ("fail", "certainty", "remove", "adjust", "average"):
        raise ValueError("lonely_psu must be 'fail', 'certainty', 'remove', 'adjust' or 'average'")
    if type == "auto":
        type = "JK1" if strata is None else "JKn"
    if (type == "JK1") != (strata is None):
        raise ValueError("JK1 requires an unstratified design and JKn requires strata")

    n = weights.size
    cluster_codes = np.arange(n) if psu is None else _codes(psu, n, "psu")[1]
    levels, stratum_codes = ((None,), np.zeros(n, dtype=int)) if strata is None else _codes(strata, n, "strata")
    order = _first_appearance(stratum_codes)
    labels = tuple(levels[level] for level in order)
    row_groups = [np.flatnonzero(stratum_codes == level) for level in order]
    local_codes = []
    for rows in row_groups:
        cluster_order = _first_appearance(cluster_codes[rows])
        index = {cluster: i for i, cluster in enumerate(cluster_order)}
        local_codes.append(np.array([index[cluster] for cluster in cluster_codes[rows]]))
    counts = np.array([codes.max() + 1 for codes in local_codes])
    corrections, zero_fraction = _corrections(fpc, fpctype, labels, row_groups, counts, n)

    if type == "JK1":
        count = counts[0]
        if count < 2:
            raise ValueError("JK1 requires at least two sampled PSUs")
        if corrections[0] == 0:
            raise ValueError("no JK1 replicates remain when all PSUs were sampled")
        factors = (local_codes[0][:, None] != np.arange(count)) * (count / (count - 1))
        scale = corrections[0] * (count - 1) / count
        rscales = None
    else:
        factors, scale, rscales = _jkn_factors(
            n, labels, row_groups, local_codes, counts, corrections, zero_fraction, lonely_psu,
        )
    return svrepdesign(factors, weights, type=type, combined_weights=False, scale=scale, rscales=rscales, mse=mse)
