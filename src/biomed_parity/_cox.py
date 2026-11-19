"""Shared partial likelihood for grouped right-censored Cox models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class _RiskSets:
    design: np.ndarray
    status: np.ndarray
    weights: np.ndarray
    starts: np.ndarray
    order: np.ndarray

    @classmethod
    def build(cls, times, status, design, weights, indices):
        order = indices[np.argsort(-times[indices], kind="stable")]
        sorted_times = times[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_times)) + 1, len(order)]
        return cls(design[order], status[order], weights[order], starts, order)

    def evaluate(self, beta, ties, *, linearize=False):
        eta = self.design @ beta
        log_risk = eta + np.log(self.weights)
        shift = log_risk.max()
        risk = np.exp(log_risk - shift)
        dimension = len(beta)
        likelihood = risk_sum = 0.0
        risk_first = np.zeros(dimension)
        risk_second = np.zeros((dimension, dimension))
        score = np.zeros(dimension)
        information = np.zeros_like(risk_second)
        if linearize:
            score_rows = np.zeros_like(self.design)
            hazard = np.zeros(len(self.starts) - 1)
            hazard_mean = np.zeros((len(hazard), dimension))
        for index, (left, right) in enumerate(zip(self.starts[:-1], self.starts[1:])):
            rows = self.design[left:right]
            row_risk = risk[left:right]
            risk_sum += row_risk.sum()
            risk_first += row_risk @ rows
            risk_second += rows.T @ (row_risk[:, None] * rows)
            death = self.status[left:right]
            number_deaths = int(death.sum())
            if not number_deaths:
                continue
            death_weight = self.weights[left:right][death]
            death_rows = rows[death]
            death_risk = row_risk[death]
            total_weight = death_weight.sum()
            likelihood += death_weight @ eta[left:right][death]
            score += death_weight @ death_rows

            if ties == "efron" and number_deaths > 1:
                fractions = np.arange(number_deaths) / number_deaths
                denominators = risk_sum - fractions * death_risk.sum()
                means = (
                    risk_first - fractions[:, None] * (death_risk @ death_rows)
                ) / denominators[:, None]
                death_second = death_rows.T @ (death_risk[:, None] * death_rows)
                seconds = (
                    risk_second - fractions[:, None, None] * death_second
                ) / denominators[:, None, None]
            else:
                fractions = np.zeros(1)
                denominators = np.array([risk_sum])
                means = (risk_first / risk_sum)[None, :]
                seconds = (risk_second / risk_sum)[None, :, :]

            likelihood -= total_weight * (np.log(denominators).mean() + shift)
            average_mean = means.mean(axis=0)
            score -= total_weight * average_mean
            information += total_weight * (seconds.mean(axis=0) - means.T @ means / len(means))

            if linearize:
                increments = (total_weight / len(fractions)) / denominators
                hazard[index] = increments.sum()
                hazard_mean[index] = increments @ means
                event_rows = left + np.flatnonzero(death)
                # A tied death participates in only a fraction of its own risk set.
                discounted = increments * fractions
                correction = death_risk[:, None] * (
                    death_rows * discounted.sum() - discounted @ means
                )
                score_rows[event_rows] = (
                    death_weight[:, None] * (death_rows - average_mean) + correction
                )
        if not linearize:
            return likelihood, score, information

        group_hazard = np.cumsum(hazard[::-1])[::-1]
        group_mean = np.cumsum(hazard_mean[::-1], axis=0)[::-1]
        counts = np.diff(self.starts)
        score_rows -= risk[:, None] * (
            self.design * np.repeat(group_hazard, counts)[:, None]
            - np.repeat(group_mean, counts, axis=0)
        )
        return score_rows


def _fit_likelihood(sets, dimension, ties):
    def evaluate(beta):
        likelihood, score, information = sets[0].evaluate(beta, ties)
        for subset in sets[1:]:
            value, gradient, curvature = subset.evaluate(beta, ties)
            likelihood += value
            score += gradient
            information += curvature
        return likelihood, score, information

    beta = np.zeros(dimension)
    value, gradient, information = evaluate(beta)
    null = value
    initial_scale = np.max(np.diag(information))
    for _ in range(50):
        try:
            direction = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Cox information matrix is singular") from exc
        if not np.isfinite(direction).all() or np.any(np.abs(beta) > 60):
            raise ValueError("the Cox fit has no stable finite solution")
        if np.max(np.abs(direction)) <= 1e-9:
            try:
                covariance = np.linalg.solve(information, np.eye(dimension))
            except np.linalg.LinAlgError as exc:
                raise ValueError("Cox information matrix is singular") from exc
            smallest = np.min(np.linalg.eigvalsh(information))
            threshold = max(1e-12 * initial_scale, 1e-11 * np.max(np.diag(information)))
            if smallest <= threshold:
                raise ValueError("Cox information matrix has insufficient rank")
            return beta, value, null, covariance
        for fraction in 2.0 ** -np.arange(30):
            candidate = beta + fraction * direction
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                next_value, next_gradient, next_information = evaluate(candidate)
            if np.isfinite(next_value) and next_value >= value - 1e-12:
                break
        else:
            raise ValueError("the Cox fit failed to improve the partial likelihood")
        beta = candidate
        value, gradient, information = next_value, next_gradient, next_information
    raise ValueError("the Cox fit failed to converge")
