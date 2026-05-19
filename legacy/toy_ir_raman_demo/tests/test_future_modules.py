from __future__ import annotations

import numpy as np

from multispec_geodiff.future_modules.graph_diffusion_stub import SizeAdaptiveGraphDiffusionStub
from multispec_geodiff.future_modules.pairwise_distance_stub import predict_pairwise_distance_matrix, recover_initial_coordinates
from multispec_geodiff.future_modules.tfn_transformer_stub import ParityAwareTFNTransformerStub


def test_graph_diffusion_stub_reports_supported_operations() -> None:
    stub = SizeAdaptiveGraphDiffusionStub()
    plan = stub.plan_sampling({"modalities": ["ir", "raman"]}, max_steps=5)
    assert plan["status"] == "stub"
    assert "insert_node" in plan["supported_operations"]


def test_pairwise_distance_stub_returns_square_matrix_and_coordinates() -> None:
    graph_features = np.ones((4, 3))
    spectral_condition = np.ones((8,))
    dist = predict_pairwise_distance_matrix(graph_features, spectral_condition)
    coords = recover_initial_coordinates(dist, n_components=3)
    assert dist.shape == (4, 4)
    assert coords.shape == (4, 3)


def test_tfn_transformer_stub_returns_shape_metadata() -> None:
    stub = ParityAwareTFNTransformerStub()
    out = stub.forward(
        node_scalars=np.ones((5, 8)),
        relative_coordinates=np.ones((5, 5, 3)),
        spectral_condition=np.ones((16,)),
    )
    assert out["status"] == "stub"
    assert out["node_shape"] == (5, 8)
