"""Two-phase survey means and totals under nested simple random sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._survival_inputs import _encode_labels

Label = str | float


@dataclass(frozen=True)
class SurveyTwoPhaseResult:
    """Point estimates, covariance contributions, and phase-two row weights."""

    estimate: NDArray[np.float64]
    covariance: NDArray[np.float64]
    phase1_covariance: NDArray[np.float64]
    phase2_covariance: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    weights: NDArray[np.float64]
    phase1_probabilities: NDArray[np.float64]
    phase2_probabilities: NDArray[np.float64]
    categories: tuple[Label, ...] | None


@dataclass(frozen=True)
class _BlockTerm:
    codes: NDArray[np.intp]
    coefficients: NDArray[np.float64]


@dataclass(frozen=True)
class _CovarianceTerms:
    diagonal: NDArray[np.float64]
    blocks: tuple[_BlockTerm, ...] = ()

    def crossproduct(self, rows):
        diagonal_extra = self.diagonal.copy()
        block_sum = np.zeros((rows.shape[1], rows.shape[1]))
        for block in self.blocks:
            sums = np.zeros((block.coefficients.size, rows.shape[1]))
            np.add.at(sums, block.codes, rows)
            block_sum += sums.T @ (block.coefficients[:, None] * sums)
            diagonal_extra -= block.coefficients[block.codes]
        return block_sum + rows.T @ (diagonal_extra[:, None] * rows)

    def elementwise_product(self, other):
        blocks = []
        for left in self.blocks:
            for right in other.blocks:
                pairs, codes = _pairs(left.codes, right.codes)
                coefficients = left.coefficients[pairs[:, 0]] * right.coefficients[pairs[:, 1]]
                blocks.append(_BlockTerm(codes, coefficients))
        return _CovarianceTerms(self.diagonal * other.diagonal, tuple(blocks))


def _pairs(left, right):
    pairs, inverse = np.unique(np.column_stack((left, right)), axis=0, return_inverse=True)
    return pairs, inverse.ravel()


def _labels(values, n, name):
    if values is None:
        return (0,), np.zeros(n, dtype=np.intp)
    if np.shape(values) != (n,):
        raise ValueError(f"{name} must have one label per phase-one row")
    return _encode_labels(values, None, n, name)


def _off_diagonal(probability, count):
    # Same-stratum, distinct-unit SRS coefficient. A singleton has no such pair.
    return np.divide(probability - 1, count - 1, out=np.zeros_like(probability), where=count > 1)


def _phase1_sampling(strata, population, selected):
    levels, codes = _labels(strata, selected.size, "phase1_strata")
    if population is None:
        if strata is not None:
            raise ValueError("supply phase1_population for stratified phase-one SRS")
        probabilities = np.ones(selected.sum())
        return probabilities, _CovarianceTerms(probabilities.copy())

    if isinstance(population, Mapping):
        if set(population) != set(levels):
            raise ValueError("phase1_population must include each phase-one stratum")
        sizes = [population[label] for label in levels]
    elif isinstance(population, Real) and len(levels) == 1:
        sizes = [population]
    else:
        raise ValueError("phase1_population must give a population size for each phase-one stratum")
    sizes = np.asarray(sizes)
    sample_sizes = np.bincount(codes)
    if (
        sizes.dtype.kind not in "iuf" or not np.isfinite(sizes).all()
        or np.any(sizes != np.floor(sizes)) or np.any(sizes < sample_sizes)
    ):
        raise ValueError("phase-one population sizes must be integers at least as large as their sample")
    probability = sample_sizes / sizes
    codes = codes[selected]
    probabilities = probability[codes]
    off = _off_diagonal(probability, sample_sizes)
    return probabilities, _CovarianceTerms(1 - probabilities, (_BlockTerm(codes, off),))


def _phase2_sampling(strata, cluster, selected):
    _, strata_code = _labels(strata, selected.size, "phase2_strata")
    if cluster is None:
        frame_count = np.bincount(strata_code)
        selected_code = strata_code[selected]
        sample_count = np.bincount(selected_code, minlength=frame_count.size)
        if np.any(sample_count == 0):
            raise ValueError("each phase-two stratum must contain a sampled unit")
        p = sample_count / frame_count
        off = _off_diagonal(p, sample_count)
        return p[selected_code], _CovarianceTerms(1 - p[selected_code], (_BlockTerm(selected_code, off),))

    _, cluster_label = _labels(cluster, selected.size, "phase2_psu")
    clusters, cluster_code = _pairs(strata_code, cluster_label)
    cluster_stratum = clusters[:, 0]
    frame_psu_count = np.bincount(cluster_stratum)
    selected_psu = np.unique(cluster_code[selected])
    sample_psu_count = np.bincount(cluster_stratum[selected_psu], minlength=frame_psu_count.size)
    if np.any(sample_psu_count == 0):
        raise ValueError("each phase-two stratum must contain a sampled cluster")

    p_psu = sample_psu_count / frame_psu_count
    b_psu = _off_diagonal(p_psu, sample_psu_count)
    frame_ssu_count = np.bincount(cluster_code)
    sample_ssu_count = np.bincount(cluster_code[selected], minlength=frame_ssu_count.size)
    p_ssu = sample_ssu_count / frame_ssu_count
    b_ssu = _off_diagonal(p_ssu, sample_ssu_count)
    p_cluster = p_psu[cluster_stratum]
    probabilities = (p_cluster * p_ssu)[cluster_code[selected]]
    # Within sampled clusters, stage-two terms enter with the stage-one inclusion probability.
    same_cluster = 1 - p_cluster + p_cluster * b_ssu
    blocks = (
        _BlockTerm(strata_code[selected], b_psu),
        _BlockTerm(cluster_code[selected], same_cluster - b_psu[cluster_stratum]),
    )
    return probabilities, _CovarianceTerms(1 - probabilities, blocks)


def _outcome(values, selected, categories):
    values = np.asarray(values)
    if values.ndim not in (1, 2) or values.shape[0] not in (selected.size, selected.sum()):
        raise ValueError("values must have one row per phase-one or sampled phase-two unit")
    if values.shape[0] == selected.size:
        values = values[selected]
    if categories is not None and values.ndim != 1:
        raise ValueError("categories require a one-dimensional outcome")
    if values.ndim == 1 and (categories is not None or all(isinstance(x, str) for x in values)):
        if categories is None:
            categories = tuple(sorted(set(values)))
        else:
            categories = tuple(categories)
        combined = (*values, *categories)
        labels_valid = (
            all(isinstance(x, str) for x in combined)
            or all(isinstance(x, Real) and np.isfinite(x) for x in combined)
        )
        if (
            not labels_valid or not categories or len(set(categories)) != len(categories)
            or not set(values).issubset(categories)
        ):
            raise ValueError("categories must be unique finite labels and include every sampled outcome label")
        return (values[:, None] == np.asarray(categories, dtype=object)).astype(float), categories
    numeric = values.dtype.kind in "biuf" or (
        values.dtype.kind == "O" and all(isinstance(x, Real) for x in values.flat)
    )
    if not numeric:
        raise ValueError("sampled numeric values must be finite")
    values = values.astype(float)
    if not np.isfinite(values).all():
        raise ValueError("sampled numeric values must be finite")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] == 0:
        raise ValueError("provide at least one numeric outcome")
    return values, None


def survey_twophase(
    values: ArrayLike,
    phase2: ArrayLike,
    *,
    phase1_population: int | Mapping[Label, int] | None = None,
    phase1_strata: ArrayLike | None = None,
    phase2_strata: ArrayLike | None = None,
    phase2_psu: ArrayLike | None = None,
    statistic: str = "mean",
    categories: Sequence[Label] | None = None,
) -> SurveyTwoPhaseResult:
    """Estimate means or totals for a nested two-phase SRS design.

    Supply a selection mask and any strata or clusters on the full phase-one
    sample frame. Phase one is iid when population sizes are omitted. A finite
    phase-one population is a single size or a mapping of strata to sizes.
    Phase two uses the phase-one sample to infer its finite population sizes.
    Providing phase-two PSUs selects a two-stage phase-two design: clusters
    are selected within strata and individual units are selected within PSUs.

    Outcomes can have one row per phase-one unit or contain only phase-two
    rows in mask order. Strings are indicator-coded. Explicit categories
    also allow numeric labels and determine column order. Mean estimates
    normalize inverse-probability weights. Totals use their original scale.
    """
    selected = np.asarray(phase2)
    if (
        selected.ndim != 1 or selected.dtype.kind not in "biuf"
        or not np.isin(selected, [0, 1]).all() or not selected.any()
    ):
        raise ValueError("phase2 must be a one-dimensional selection mask with at least one selected unit")
    if statistic not in ("mean", "total"):
        raise ValueError("statistic must be 'mean' or 'total'")
    selected = selected.astype(bool)
    data, levels = _outcome(values, selected, categories)
    p1, terms1 = _phase1_sampling(phase1_strata, phase1_population, selected)
    p2, terms2 = _phase2_sampling(phase2_strata, phase2_psu, selected)
    weights = 1 / (p1 * p2)

    if statistic == "mean":
        normalized_weights = weights / weights.sum()
        estimate = normalized_weights @ data
        rows = normalized_weights[:, None] * (data - estimate)
    else:
        estimate = weights @ data
        rows = weights[:, None] * data

    # The composite Horvitz–Thompson pair factor is D1 + D2 - D1 * D2.
    shared = terms1.elementwise_product(terms2)
    covariance1 = terms1.crossproduct(rows) - shared.crossproduct(rows)
    covariance2 = terms2.crossproduct(rows)
    covariance = covariance1 + covariance2
    with np.errstate(invalid="ignore"):
        standard_errors = np.sqrt(np.diag(covariance))
    return SurveyTwoPhaseResult(estimate, covariance, covariance1, covariance2, standard_errors, weights, p1, p2, levels)
