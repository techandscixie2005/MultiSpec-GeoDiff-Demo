from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import classical_mds


def predict_pairwise_distance_matrix(graph_features: np.ndarray, spectral_condition: np.ndarray) -> np.ndarray:
    """Stub signature for future 2D-graph-to-distance prediction.

    Pairwise distance is the bridge from a discrete 2D graph to 3D geometry because it is
    invariant to rotation and translation while still constraining the recoverable geometry.
    This demo does not train a predictor; it only returns a stable heuristic matrix shape.
    """
    n = int(graph_features.shape[0])
    base = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    condition_scale = float(np.clip(np.mean(np.abs(spectral_condition)), 0.5, 2.0))
    return base * condition_scale + np.eye(n)


def recover_initial_coordinates(distance_matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    return classical_mds(distance_matrix, n_components=n_components)
