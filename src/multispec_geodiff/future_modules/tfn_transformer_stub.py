from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParityChannels:
    scalar_even: int
    vector_odd: int
    tensor_even: int
    pseudoscalar_odd: int


class ParityAwareTFNTransformerStub:
    """Conceptual interface for a future parity-aware TFN-Transformer refiner.

    Intended roadmap concepts:
    - tensor inner-product attention using invariant scalar scores
    - relative-coordinate updates for SE(3)-equivariant refinement
    - even/odd parity channels for chirality-sensitive reasoning
    - spectrum-conditioned geometry refinement after distance initialization

    The current demo keeps this as an importable interface only.
    """

    def __init__(self, channels: ParityChannels | None = None):
        self.channels = channels or ParityChannels(32, 16, 8, 4)

    def forward(self, node_scalars: np.ndarray, relative_coordinates: np.ndarray, spectral_condition: np.ndarray, edge_features: np.ndarray | None = None) -> dict[str, Any]:
        return {
            "status": "stub",
            "message": "TFN-Transformer refinement is roadmap-only in this demo.",
            "channels": self.channels,
            "node_shape": tuple(node_scalars.shape),
            "relative_coordinate_shape": tuple(relative_coordinates.shape),
            "spectral_condition_shape": tuple(spectral_condition.shape),
            "edge_shape": None if edge_features is None else tuple(edge_features.shape),
        }
