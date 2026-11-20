"""Sampling covariance matrices for dependent effects in meta-analysis."""

from dataclasses import dataclass
from numbers import Real
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from .meta_multilevel import _codes, _real_array


class MetaCovarianceWarning(UserWarning):
    """A study correlation matrix is not positive definite."""


@dataclass(frozen=True)
class NamedCorrelation:
    """Matrix with rows and columns in the supplied label order."""

    levels: tuple
    matrix: ArrayLike


def _labels(values, size, name, default):
    labels = np.full(size, default, dtype=object) if values is None else np.asarray(values, dtype=object)
    if labels.shape != (size,):
        raise ValueError(f"{name} must have one label per effect")
    _codes(labels, size, name)
    return labels


def _vector(values, size, name):
    array = _real_array(values, name)
    if array.shape != (size,):
        raise ValueError(f"{name} must have one finite value per effect")
    return array


def _construct_correlation(correlation, labels, name):
    if isinstance(correlation, NamedCorrelation):
        levels = tuple(correlation.levels)
        _codes(levels, len(levels), name)
        index = {level: i for i, level in enumerate(levels)}
        matrix = _real_array(correlation.matrix, name)
        if (
            len(index) != len(levels) or matrix.shape != (len(levels), len(levels))
            or not all(label in index for label in labels)
            or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
            or np.any(np.abs(matrix) > 1)
            or not np.allclose(np.diag(matrix), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError(f"{name} needs a symmetric unit-diagonal matrix that covers all labels")
        codes = np.array([index[label] for label in labels], dtype=int)
        return (matrix + matrix.T) / 2, codes

    if not isinstance(correlation, Real) or not -1 <= correlation <= 1:
        raise ValueError(f"{name} must be a correlation in [-1, 1] or a NamedCorrelation")
    return float(correlation), _codes(labels, len(labels), name)


def _from_levels(matrix, codes, rows):
    local = codes[rows]
    if np.ndim(matrix) == 0:
        return np.where(local[:, None] == local, 1, matrix)
    return matrix[np.ix_(local, local)]


def _correlation_terms(obs, type, rho, size):
    observation_labels = _labels(obs, size, "obs", 1)
    construct_labels = _labels(type, size, "type", 1)
    if obs is None and type is None:
        return (0.0, np.zeros(size, dtype=int)), (0.0, np.zeros(size, dtype=int))
    if rho is None:
        raise ValueError("rho is required when obs or type is specified")
    if obs is not None and type is not None:
        try:
            observation_rho, construct_rho = rho
        except (TypeError, ValueError) as exc:
            raise ValueError("rho must give both within- and between-construct correlations") from exc
    else:
        observation_rho = rho if obs is not None else 0.0
        construct_rho = rho if type is not None else 0.0
    return (
        _construct_correlation(observation_rho, observation_labels, "within-construct rho"),
        _construct_correlation(construct_rho, construct_labels, "between-construct rho"),
    )


def _temporal(phi, t1, t2, shared):
    with np.errstate(over="ignore"):
        distance = np.abs(t1 - t2)
    distance = np.where(shared, distance, 0)
    if phi < 0 and np.any(distance != np.floor(distance)):
        raise ValueError("negative phi requires integral time differences in shared arms")
    return np.where(shared, np.power(phi, distance), 0)


@dataclass(frozen=True)
class _Contrasts:
    groups: tuple
    times: tuple
    weights: tuple
    phi: float
    variance_root: np.ndarray

    def correlation(self, rows):
        first_group, second_group = (group[rows] for group in self.groups)
        first_time, second_time = (time[rows] for time in self.times)
        first, second = (weight[rows] for weight in self.weights)

        def kernel(g1, g2, t1, t2, w1, w2):
            return _temporal(
                self.phi, t1[:, None], t2, g1[:, None] == g2,
            ) * (w1[:, None] * w2)

        cross = kernel(first_group, second_group, first_time, second_time, first, second)
        sampling_covariance = (
            kernel(first_group, first_group, first_time, first_time, first, first)
            + kernel(second_group, second_group, second_time, second_time, second, second)
            - cross - cross.T
        )
        return sampling_covariance / np.outer(self.variance_root[rows], self.variance_root[rows])


def _contrasts(n, grp1, grp2, time1, time2, phi, w1, w2):
    if grp2 is not None and grp1 is None:
        raise ValueError("grp2 requires grp1")
    if time2 is not None and (time1 is None or grp2 is None):
        raise ValueError("time2 requires time1 and grp2")
    if w2 is not None and (w1 is None or grp2 is None):
        raise ValueError("w2 requires w1 and grp2")
    if time1 is not None and (not isinstance(phi, Real) or not -1 <= phi <= 1):
        raise ValueError("specifying time requires phi in [-1, 1]")
    phi = 1.0 if time1 is None else float(phi)
    g1, g2 = _labels(grp1, n, "grp1", 1), _labels(grp2, n, "grp2", 2)
    t1 = np.ones(n) if time1 is None else _vector(time1, n, "time1")
    t2 = t1 if time2 is None else _vector(time2, n, "time2")
    w1 = np.ones(n) if w1 is None else _vector(w1, n, "w1")
    w2 = w1 if w2 is None else _vector(w2, n, "w2")
    if np.any((w1 <= 0) | (w2 <= 0)):
        raise ValueError("arm weights must be positive")
    # Scale both arm precisions per effect, since the scale cancels in correlation.
    unit = np.minimum(w1, w2)
    first, second = np.sqrt(unit / w1), np.sqrt(unit / w2)
    own_correlation = _temporal(phi, t1, t2, g1 == g2)
    variance = (first - second) ** 2 + 2 * first * second * (1 - own_correlation)
    if np.any(variance <= 0):
        raise ValueError("a within-effect contrast has zero model variance")
    return _Contrasts((g1, g2), (t1, t2), (first, second), phi, np.sqrt(variance))


def _blocks(cluster, subgroup, size):
    studies = _labels(cluster, size, "cluster", 1)
    cohorts = _labels(subgroup, size, "subgroup", 1)
    groups = {}
    for row, key in enumerate(zip(studies, cohorts, strict=True)):
        groups.setdefault(key, []).append(row)
    return {key: np.array(rows, dtype=int) for key, rows in groups.items()}


def _given_correlations(rvars, n, columns):
    matrix = np.asarray(rvars, dtype=float)
    if matrix.shape != (n, columns):
        raise ValueError("rvars must have one row per effect and as many columns as the largest block")
    return matrix


def _given_block(rvars, rows):
    lower = np.tril(rvars[rows, :len(rows)])
    if (not np.isfinite(lower).all() or (abs(lower) > 1).any()
            or not np.allclose(np.diag(lower), 1, atol=1e-12, rtol=0)):
        raise ValueError("rvars lower blocks must contain correlations and a unit diagonal")
    return lower + np.tril(lower, -1).T


def _nearest_correlation(matrix):
    """Project onto the PSD cone and the unit diagonal with Dykstra correction."""
    current = matrix.copy()
    correction = np.zeros_like(current)
    converged = False
    for _ in range(100):
        residual = current - correction
        eigenvalues, basis = np.linalg.eigh(residual)
        positive = eigenvalues > 1e-6 * eigenvalues[-1]
        projection = (basis[:, positive] * eigenvalues[positive]) @ basis[:, positive].T
        correction = projection - residual
        np.fill_diagonal(projection, 1)
        change = np.linalg.norm(projection - current, ord=np.inf)
        tolerance = 1e-7 * np.linalg.norm(current, ord=np.inf)
        current = projection
        if change <= tolerance:
            converged = True
            break

    eigenvalues, basis = np.linalg.eigh(current)
    floor = 1e-8 * eigenvalues[-1]
    if eigenvalues[0] < floor:
        current = (basis * np.maximum(eigenvalues, floor)) @ basis.T
        root = np.sqrt(np.diag(current))
        current /= np.outer(root, root)
    current = (current + current.T) * 0.5
    np.fill_diagonal(current, 1)
    return current, converged


def vcalc(
    vi: ArrayLike,
    cluster: ArrayLike,
    *,
    subgroup: ArrayLike | None = None,
    obs: ArrayLike | None = None,
    type: ArrayLike | None = None,
    time1: ArrayLike | None = None,
    time2: ArrayLike | None = None,
    grp1: ArrayLike | None = None,
    grp2: ArrayLike | None = None,
    w1: ArrayLike | None = None,
    w2: ArrayLike | None = None,
    rho: Real | NamedCorrelation | tuple | None = None,
    phi: float | None = None,
    rvars: ArrayLike | None = None,
    checkpd: bool = True,
    nearpd: bool = False,
    sparse: bool = False,
) -> NDArray[np.float64] | csr_matrix:
    """Construct dependent effect sampling covariance in the original row order.

    Effects within a study/subgroup covary through common groups, close
    assessment times and common outcomes or scales. Supply scalar or named
    matrix correlations for obs or type, and two correlations (within,
    between) when both are supplied. Alternatively specify the lower
    triangle of each group's correlation in rvars, one row per effect.
    Set nearpd=True to project indefinite blocks to a positive correlation.
    """
    variances = _real_array(vi, "vi")
    if variances.ndim != 1 or not len(variances) or np.any(variances < 0):
        raise ValueError("vi must contain one or more nonnegative sampling variances")
    if not all(isinstance(flag, (bool, np.bool_)) for flag in (checkpd, nearpd, sparse)):
        raise ValueError("checkpd, nearpd and sparse must be boolean")
    n = len(variances)
    if cluster is None:
        raise ValueError("cluster must be supplied")
    blocks = _blocks(cluster, subgroup, n)
    roots = np.sqrt(variances)
    if rvars is not None:
        if any(value is not None for value in (obs, type, time1, time2, grp1, grp2, w1, w2, rho, phi)):
            raise ValueError("rvars cannot be combined with structured correlation inputs")
        rvars = _given_correlations(rvars, n, max(map(len, blocks.values())))
    else:
        observation, construct = _correlation_terms(obs, type, rho, n)
        contrasts = _contrasts(n, grp1, grp2, time1, time2, phi, w1, w2)

    covariance = np.zeros((n, n)) if not sparse else None
    flat_values, row_indices, column_indices = [], [], []
    for key, rows in blocks.items():
        locations = np.ix_(rows, rows)
        if rvars is None:
            within = _from_levels(*observation, rows)
            between = _from_levels(*construct, rows)
            kinds = construct[1][rows]
            correlation = contrasts.correlation(rows) * np.where(kinds[:, None] == kinds, within, between)
            correlation = (correlation + correlation.T) * 0.5
            np.fill_diagonal(correlation, 1)
        else:
            correlation = _given_block(rvars, rows)
        if (checkpd or nearpd) and np.linalg.eigvalsh(correlation)[0] <= np.finfo(float).eps:
            if nearpd:
                correlation, converged = _nearest_correlation(correlation)
                if not converged:
                    warn(f"nearest-correlation projection did not converge in cluster {key!r}",
                         MetaCovarianceWarning, stacklevel=2)
            else:
                warn(f"sampling correlation is not positive definite in cluster {key!r}",
                     MetaCovarianceWarning, stacklevel=2)
        block = correlation * np.outer(roots[rows], roots[rows])
        np.fill_diagonal(block, variances[rows])
        if sparse:
            row, col = block.nonzero()
            flat_values.append(block[row, col])
            row_indices.append(rows[row])
            column_indices.append(rows[col])
        else:
            covariance[locations] = block
    if not sparse:
        return covariance
    coordinates = (np.concatenate(row_indices), np.concatenate(column_indices))
    return csr_matrix((np.concatenate(flat_values), coordinates), shape=(n, n))
