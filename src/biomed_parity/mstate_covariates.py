"""Build transition-specific numerical columns from counting-process records."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from ._contrasts import _factor_contrasts
from .mstate_prep import MultiStateRecords


def _missing(value):
    return value is None or (isinstance(value, Real) and np.isnan(value))


def _valid_labels(values):
    return (
        all(isinstance(value, str) for value in values)
        or all(isinstance(value, (Real, np.bool_)) and np.isfinite(value) for value in values)
    )


def _label_name(level):
    if isinstance(level, (bool, np.bool_)):
        return "TRUE" if level else "FALSE"
    if isinstance(level, Real) and not isinstance(level, int):
        return str(level).removesuffix(".0")
    return str(level)


def _coefficient_names(names):
    """Convert contrast names to the lexical forms used in R data frames."""
    converted = []
    for label in names:
        name = "".join(char if char.isalnum() or char in "._" else "." for char in label)
        if not (name[0].isalpha() or (name[0] == "." and not name[1:2].isdigit())):
            name = "X" + name
        converted.append(name)
    unique = set(converted)
    seen = set()
    for index, name in enumerate(converted):
        if name in seen:
            suffix = 1
            while f"{name}.{suffix}" in unique:
                suffix += 1
            converted[index] = f"{name}.{suffix}"
            unique.add(converted[index])
        seen.add(name)
    return converted


def _contrast_matrix(values, name, levels, ordered, long_names):
    missing = np.fromiter((_missing(value) for value in values), dtype=bool, count=len(values))
    observed = values[~missing].tolist()
    if not _valid_labels(observed):
        raise ValueError(f"{name!r} must contain finite numbers or factor labels of one kind")
    logical = bool(observed) and all(isinstance(value, (bool, np.bool_)) for value in observed)
    if levels is None and (ordered or logical or (observed and isinstance(observed[0], str))):
        levels = (False, True) if logical else sorted(set(observed))
    if levels is None:
        numeric = np.full((len(values), 1), np.nan)
        numeric[~missing, 0] = observed
        return numeric, [name], missing
    if isinstance(levels, str):
        raise ValueError(f"{name!r} must declare levels as a sequence")
    levels = tuple(levels)
    if len(levels) < 2 or not _valid_labels(levels) or len(set(levels)) != len(levels):
        raise ValueError(f"{name!r} must declare at least two distinct finite factor levels")
    level_codes = {level: index for index, level in enumerate(levels)}
    if not all(value in level_codes for value in observed):
        raise ValueError(f"{name!r} contains an undeclared factor level")
    codes = np.fromiter((level_codes[value] for value in observed), dtype=np.intp, count=len(observed))
    basis = _factor_contrasts(len(levels), ordered)
    contrasts = np.full((len(values), basis.shape[1]), np.nan)
    contrasts[~missing] = basis[codes]
    if long_names:
        if ordered:
            suffixes = [".L", ".Q", ".C"] + [f".{degree}" for degree in range(4, len(levels))]
        else:
            suffixes = [_label_name(level) for level in levels[1:]]
        names = _coefficient_names([name + suffix for suffix in suffixes[:basis.shape[1]]])
    elif basis.shape[1] == 1:
        names = [name]
    else:
        names = [f"{name}{number}" for number in range(1, len(levels))]
    return contrasts, names, missing


def expand_covariates(
    records: MultiStateRecords,
    covariates: str | Sequence[str],
    *,
    factor_levels: Mapping[str, Sequence[str | float]] | None = None,
    ordered: Sequence[str] = (),
    append: bool = True,
    long_names: bool = True,
) -> MultiStateRecords | dict[str, NDArray[np.float64]]:
    """Expand each contrast over the declared transitions of a long-format table.

    Numeric variables are copied on their matching transition and zero on
    other transitions. Missing values propagate across all their columns.
    Categorical levels can be supplied explicitly to set the contrast order.
    Column suffixes use one-based transition numbers. With append=False,
    return just the expanded columns in contrast-major order.
    """
    if not isinstance(records, MultiStateRecords):
        raise ValueError("records must be MultiStateRecords from msprep")
    if not isinstance(append, (bool, np.bool_)) or not isinstance(long_names, (bool, np.bool_)):
        raise ValueError("append and long_names must be boolean")
    names = (covariates,) if isinstance(covariates, str) else tuple(covariates)
    if not names or not all(isinstance(name, str) and name and name in records.covariates for name in names):
        raise ValueError("covariates must name one or more columns in records")
    if len(set(names)) != len(names):
        raise ValueError("covariates must not repeat a column")
    factors = {} if factor_levels is None else factor_levels
    if not isinstance(factors, Mapping) or set(factors).difference(names):
        raise ValueError("factor_levels must map selected covariates to their levels")
    if isinstance(ordered, str) or set(ordered).difference(names):
        raise ValueError("ordered must name selected covariates")

    indicators = records.transition[:, None] == np.arange(len(records.transitions))
    expanded = {}
    for name in names:
        contrasts, base_names, missing = _contrast_matrix(
            records.covariates[name], name, factors.get(name), name in ordered, long_names,
        )
        for base, contrast in zip(base_names, contrasts.T, strict=True):
            columns = np.where(indicators, contrast[:, None], 0.0)
            columns[missing] = np.nan
            for edge, values in enumerate(columns.T, 1):
                term = f"{base}.{edge}"
                if term in expanded or (append and term in records.covariates):
                    raise ValueError(f"expanded column name {term!r} already exists")
                expanded[term] = values.copy()
    if not append:
        return expanded
    return replace(records, covariates=records.covariates | expanded)
