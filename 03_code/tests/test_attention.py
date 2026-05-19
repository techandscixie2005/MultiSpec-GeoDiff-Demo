"""Tests for spectral_attention module."""

import numpy as np


def test_attention_output_shape():
    from spectral_attention import SpectralDistanceBiasAttention
    attn = SpectralDistanceBiasAttention(hidden_dim=16)
    tokens = np.random.randn(100, 16).astype(float)
    wavenumber = np.linspace(400, 4000, 100)
    output, attention = attn.forward(tokens, wavenumber)
    assert output.shape == (100, 16), f"Expected (100, 16), got {output.shape}"
    assert attention.shape == (100, 100), f"Expected (100, 100), got {attention.shape}"


def test_attention_is_softmax():
    from spectral_attention import SpectralDistanceBiasAttention
    attn = SpectralDistanceBiasAttention(hidden_dim=16)
    tokens = np.random.randn(50, 16).astype(float)
    wn = np.linspace(400, 4000, 50)
    _, attention = attn.forward(tokens, wn)
    row_sums = attention.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Rows should sum to 1 (softmax)"


def test_attention_nonnegative():
    from spectral_attention import SpectralDistanceBiasAttention
    attn = SpectralDistanceBiasAttention(hidden_dim=16)
    tokens = np.random.randn(50, 16).astype(float)
    wn = np.linspace(400, 4000, 50)
    _, attention = attn.forward(tokens, wn)
    assert np.all(attention >= 0), "Attention weights should be non-negative"


def test_distance_bias():
    from spectral_attention import SpectralDistanceBiasAttention
    attn = SpectralDistanceBiasAttention(hidden_dim=16, bias_sigma=200.0)
    wn = np.array([400, 1000, 2000, 3000, 4000], dtype=float)
    bias = attn.compute_distance_bias(wn)
    assert bias.shape == (5, 5)
    # Diagonal should be 1.0 (zero distance)
    assert np.allclose(np.diag(bias), 1.0)
    # Off-diagonal should be less than 1.0
    assert np.all(bias <= 1.0)
    # Nearby points should have higher bias than far points
    assert bias[0, 1] > bias[0, 4], "Nearby points should have higher bias"


def test_compute_heatmap():
    from spectral_attention import compute_attention_heatmap
    attn = np.random.rand(100, 100)
    heatmap = compute_attention_heatmap(attn, downsample=4)
    assert heatmap.shape == (25, 25)
