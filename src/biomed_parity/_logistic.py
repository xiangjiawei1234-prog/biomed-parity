"""Shared numerical routines for binary logistic likelihoods."""

import numpy as np
from scipy.special import expit, logit


def _deviance(signed, linear):
    return 2 * np.logaddexp(0, -signed * linear).sum()


def _information(design, linear):
    weights = expit(linear) * expit(-linear)
    return np.einsum("ij,ik,i->jk", design, design, weights)


def _quadratic_penalty(slopes, penalty):
    return np.einsum("i,ij,j->", slopes, penalty, slopes)


def _fit_coefficients(design, y, penalty=None):
    signed = 2 * y - 1
    coefficients = np.r_[logit(y.mean()), np.zeros(design.shape[1] - 1)]

    def objective(linear, beta):
        deviance = _deviance(signed, linear)
        if penalty is not None:
            deviance += _quadratic_penalty(beta[1:], penalty)
        return deviance

    linear = np.einsum("ij,j->i", design, coefficients)
    deviance = objective(linear, coefficients)
    for _ in range(80):
        score = np.einsum("ij,i->j", design, signed * expit(-signed * linear))
        information = _information(design, linear)
        if penalty is not None:
            score[1:] -= np.einsum("ij,j->i", penalty, coefficients[1:])
            information[1:, 1:] += penalty
        delta = np.linalg.solve(information, score)
        if np.max(np.abs(delta)) <= 1e-9:
            return coefficients
        for half in range(25):
            updated = coefficients + delta * 0.5**half
            fitted = np.einsum("ij,j->i", design, updated)
            candidate = objective(fitted, updated)
            if candidate <= deviance + np.finfo(float).eps * deviance:
                coefficients, linear, deviance = updated, fitted, candidate
                break
        else:
            break
    raise np.linalg.LinAlgError("logistic fit did not converge to a finite maximum")
