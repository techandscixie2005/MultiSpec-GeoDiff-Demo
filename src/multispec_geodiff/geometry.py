from __future__ import annotations

import numpy as np


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    if d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square")
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ (d ** 2) @ h
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1]
    values = values[order][:n_components]
    vectors = vectors[:, order][:, :n_components]
    values = np.clip(values, a_min=0.0, a_max=None)
    return vectors * np.sqrt(values)


def pseudo_distance_matrix(atom_count: int) -> np.ndarray:
    idx = np.arange(atom_count, dtype=float)
    return np.abs(idx[:, None] - idx[None, :]) + np.eye(atom_count)
