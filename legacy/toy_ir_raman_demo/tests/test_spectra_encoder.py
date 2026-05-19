from __future__ import annotations

import numpy as np

from multispec_geodiff.spectra_encoder import CoordinateAwareSpectralEncoder


def test_build_relative_bias_returns_square_matrix_matching_token_count(sample_processed_spectrum) -> None:
    encoder = CoordinateAwareSpectralEncoder(hidden_dim=8, num_rbf=4, seed=11)

    bias = encoder.build_relative_bias(sample_processed_spectrum.wavenumber)

    assert bias.shape == (10, 10)
    assert np.allclose(np.diag(bias), np.ones(10))
    assert np.allclose(bias, bias.T)


def test_encode_modality_returns_expected_tensor_shapes(sample_processed_spectrum) -> None:
    encoder = CoordinateAwareSpectralEncoder(hidden_dim=8, num_rbf=4, seed=11)

    encoded = encoder.encode_modality(sample_processed_spectrum)

    assert encoded.token_features.shape == (10, 8)
    assert encoded.attended_tokens.shape == (10, 8)
    assert encoded.global_embedding.shape == (8,)
    assert encoded.attention.shape == (10, 10)
    assert encoded.relative_bias.shape == (10, 10)
