"""Reverse Cox weights and cumulative-hazard influence for IPCW."""

from dataclasses import dataclass

import numpy as np

from ._cox import _RiskSets, _fit_likelihood


def _reverse_sum(values):
    return np.cumsum(values[::-1], axis=0)[::-1]


@dataclass(frozen=True)
class _CensoringCoxStratum:
    rows: np.ndarray
    knots: np.ndarray
    time_index: np.ndarray
    censored: np.ndarray
    design: np.ndarray
    relative_risk: np.ndarray
    inverse_risk_set: np.ndarray
    hazard_over_risk: np.ndarray
    risk_mean: np.ndarray
    cumulative_hazard: np.ndarray
    cumulative_covariate_hazard: np.ndarray
    score_hazard: np.ndarray
    score_covariate_hazard: np.ndarray

    @classmethod
    def fit(cls, times, censored, design, beta, rows):
        times = times[rows]
        knots, index = np.unique(times, return_inverse=True)
        censor = censored[rows]
        covariates = design[rows]
        log_risk = covariates @ beta
        risk = np.exp(log_risk - max(log_risk))
        if not np.all(risk > 0):
            raise ValueError("reverse Cox fitted risks are outside the supported range")

        total_mass = np.bincount(index, weights=risk, minlength=len(knots))
        risk_set = _reverse_sum(total_mass)
        censor_count = np.bincount(index[censor], minlength=len(knots))
        censor_mass = np.bincount(index[censor], weights=risk[censor], minlength=len(knots))
        covariate_mass = np.zeros((len(knots), design.shape[1]))
        np.add.at(covariate_mass, index, covariates * risk[:, None])
        risk_mean = _reverse_sum(covariate_mass) / risk_set[:, None]

        hazard = censor_count / risk_set
        for knot in np.flatnonzero(censor_count > 1):
            count = censor_count[knot]
            denominators = risk_set[knot] - (np.arange(count) / count) * censor_mass[knot]
            hazard[knot] = np.reciprocal(denominators).sum()
        hazard_mean = hazard[:, None] * risk_mean
        # iidCox linearizes beta on the distinct censoring-time grid.
        score_increment = (censor_count > 0) / risk_set
        return cls(
            rows, knots, index, censor, covariates, risk, 1 / risk_set,
            hazard / risk_set, risk_mean, np.r_[0.0, np.cumsum(hazard)],
            np.vstack((np.zeros(design.shape[1]), np.cumsum(hazard_mean, axis=0))),
            np.cumsum(score_increment),
            np.cumsum(score_increment[:, None] * risk_mean, axis=0),
        )

    def coefficient_scores(self):
        hazard = self.score_hazard[self.time_index]
        expected_covariate = self.score_covariate_hazard[self.time_index]
        return (
            self.censored[:, None] * (self.design - self.risk_mean[self.time_index])
            - self.relative_risk[:, None] * (hazard[:, None] * self.design - expected_covariate)
        )

    def query_indices(self, observed_event, horizon):
        at_horizon = np.searchsorted(self.knots, horizon, side="right")
        return np.where(observed_event[self.rows], self.time_index, at_horizon)

    def sampling_influence(self, sensitivity, indices):
        weighted = sensitivity * self.relative_risk
        end_mass = np.bincount(indices, weights=weighted, minlength=len(self.knots) + 1)
        tail = _reverse_sum(end_mass[1:])
        event_contribution = self.censored * (tail * self.inverse_risk_set)[self.time_index]
        drift = self.relative_risk * np.cumsum(tail * self.hazard_over_risk)[self.time_index]
        derivative = weighted @ (
            self.cumulative_hazard[indices, None] * self.design
            - self.cumulative_covariate_hazard[indices]
        )
        return event_contribution - drift, derivative


@dataclass(frozen=True)
class _CensoringCox:
    strata: tuple[_CensoringCoxStratum, ...]
    coefficient_influence: np.ndarray

    @classmethod
    def fit(cls, times, censored, covariates, strata):
        size = len(times)
        subsets = tuple(np.flatnonzero(strata == level) for level in np.unique(strata))
        scale = covariates.std(axis=0)
        if np.any(scale == 0):
            raise ValueError("censoring_covariates must be nonconstant and omit an intercept")
        design = (covariates - covariates.mean(axis=0)) / scale
        dimension = design.shape[1]
        if dimension and np.linalg.matrix_rank(design) != dimension:
            raise ValueError("censoring_covariates must have full column rank")
        if dimension and censored.any():
            fits = [_RiskSets.build(times, censored, design, np.ones(size), rows) for rows in subsets]
            beta, _, _, inverse_information = _fit_likelihood(fits, dimension, "efron")
        else:
            beta = np.zeros(dimension)
            inverse_information = np.zeros((dimension, dimension))

        baselines = tuple(
            _CensoringCoxStratum.fit(times, censored, design, beta, rows) for rows in subsets
        )
        coefficient_influence = np.empty((size, dimension))
        for baseline in baselines:
            coefficient_influence[baseline.rows] = baseline.coefficient_scores() @ inverse_information
        return cls(baselines, coefficient_influence)

    def weights_at(self, observed_event, after_horizon, horizon):
        known = observed_event | after_horizon
        weights = np.zeros(len(known))
        indices = []
        for baseline in self.strata:
            query = baseline.query_indices(observed_event, horizon)
            indices.append(query)
            mask = known[baseline.rows]
            cumulative = baseline.cumulative_hazard[query[mask]]
            with np.errstate(over="ignore"):
                weights[baseline.rows[mask]] = np.exp(baseline.relative_risk[mask] * cumulative)
        if not np.all(np.isfinite(weights)):
            raise ValueError("reverse Cox censoring survival is too small at an evaluated time")
        return weights, indices

    def sampling_influence(self, sensitivity, query_indices):
        direct = np.zeros(len(sensitivity))
        gradient = np.zeros(self.coefficient_influence.shape[1])
        for baseline, query in zip(self.strata, query_indices, strict=True):
            direct[baseline.rows], derivative = baseline.sampling_influence(
                sensitivity[baseline.rows], query,
            )
            gradient += derivative
        return direct + self.coefficient_influence @ gradient
