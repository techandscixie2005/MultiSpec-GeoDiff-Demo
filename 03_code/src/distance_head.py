"""Stage-II prototype: 2D graph -> pairwise distance matrix D.

This module is an interface placeholder for future training.
It is not claimed as a completed trained model in the current demo.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PairwiseDistanceHead:
    """Stage-II prototype: predict symmetric pairwise distance matrix D from a 2D molecular graph.

    D_ij = ||r_i - r_j||, where r_i are the true 3D coordinates.
    D is invariant to global rotation and translation.

    Future roadmap:
      - Learned GNN encoder from 2D graph -> D_hat
      - D_hat -> MDS/distance geometry -> R^(0) initial 3D coordinates
      - R^(0) -> TFN-Transformer refinement (Stage-III)

    The current demo keeps this as an importable interface stub.
    """

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._rng = np.random.default_rng(42)

    def forward(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Forward pass placeholder.

        Parameters
        ----------
        node_features : ndarray, shape (n_atoms, feat_dim)
            Per-atom features from a 2D graph encoder.
        edge_features : ndarray, shape (n_atoms, n_atoms, e_feat_dim) or None
            Optional edge features.

        Returns
        -------
        dict with:
          - D_hat: symmetric non-negative matrix (n_atoms, n_atoms)
          - status: str
          - message: str
        """
        n = node_features.shape[0]
        base = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
        scale = float(np.clip(np.mean(np.abs(node_features)), 0.5, 2.0))
        D_hat = base * scale + np.eye(n)
        return {
            "D_hat": D_hat,
            "status": "stub",
            "message": (
                "PairwiseDistanceHead is a Stage-II prototype. "
                "This demo does not train the distance predictor; "
                "a heuristic matrix is returned for interface testing."
            ),
        }
