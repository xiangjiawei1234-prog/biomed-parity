"""Treatment and orthogonal polynomial contrasts on ordered factor levels."""

import numpy as np


def _factor_contrasts(n_levels, ordered):
    if not ordered:
        return np.eye(n_levels)[:, 1:]
    scores = np.linspace(-1, 1, n_levels)
    basis = np.empty((n_levels, n_levels))
    basis[:, 0] = 1 / np.sqrt(n_levels)
    for degree in range(1, n_levels):
        vector = scores * basis[:, degree - 1]
        previous = basis[:, :degree]
        for _ in range(2):
            vector -= previous @ (previous.T @ vector)
        basis[:, degree] = vector / np.linalg.norm(vector)
    return basis[:, 1:]
