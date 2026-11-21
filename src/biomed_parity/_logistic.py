"""Shared numerical routines for binary logistic likelihoods."""

import numpy as np
from scipy.special import expit, logit


def _deviance(signed, linear, weights=None):
    losses = np.logaddexp(0, -signed * linear)
    if weights is not None:
        losses *= weights
    return 2 * losses.sum()


def _information(design, linear, weights=None):
    curvature = expit(linear) * expit(-linear)
    if weights is not None:
        curvature *= weights
    return np.einsum("ij,ik,i->jk", design, design, curvature)


def _quadratic_penalty(slopes, penalty):
    return np.einsum("i,ij,j->", slopes, penalty, slopes)


def _fit_coefficients(design, y, penalty=None, *, weights=None, offset=0, intercept=None):
    signed = 2 * y - 1
    start = logit(y.mean()) if intercept is None else intercept
    coefficients = np.r_[start, np.zeros(design.shape[1] - 1)]

    def objective(linear, beta):
        deviance = _deviance(signed, linear, weights)
        if penalty is not None:
            deviance += _quadratic_penalty(beta[1:], penalty)
        return deviance

    linear = np.einsum("ij,j->i", design, coefficients) + offset
    deviance = objective(linear, coefficients)
    for _ in range(80):
        residual = signed * expit(-signed * linear)
        if weights is not None:
            residual *= weights
        score = np.einsum("ij,i->j", design, residual)
        information = _information(design, linear, weights)
        if penalty is not None:
            score[1:] -= np.einsum("ij,j->i", penalty, coefficients[1:])
            information[1:, 1:] += penalty
        delta = np.linalg.solve(information, score)
        if np.max(np.abs(delta)) <= 1e-9:
            return coefficients
        for half in range(25):
            updated = coefficients + delta * 0.5**half
            fitted = np.einsum("ij,j->i", design, updated) + offset
            candidate = objective(fitted, updated)
            if candidate <= deviance + np.finfo(float).eps * deviance:
                coefficients, linear, deviance = updated, fitted, candidate
                break
        else:
            break
    raise np.linalg.LinAlgError("logistic fit did not converge to a finite maximum")
