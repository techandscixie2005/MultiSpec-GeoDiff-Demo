"""Coordinate-aware spectral encoder for the Stage-I demo.

h_i = W_I * I_i + p(nu_i)

where:
  - I_i is the intensity at wavenumber nu_i
  - p(nu_i) is a Fourier coordinate encoding of the wavenumber
  - W_I is a learned linear projection (deterministic for demo)
"""

from __future__ import annotations

import numpy as np

from spectral_attention import SpectralDistanceBiasAttention


def fourier_coordinate_encoding(wavenumber: np.ndarray, num_freqs: int = 16) -> np.ndarray:
    """Fourier feature encoding for wavenumber positions.

    Parameters
    ----------
    wavenumber : ndarray, shape (n,)
        Wavenumber values.
    num_freqs : int
        Number of frequency bands (default 16, producing 32 features).

    Returns
    -------
    encoding : ndarray, shape (n, 2 * num_freqs)
        Sin/cos features at multiple frequencies.
    """
    wn = np.asarray(wavenumber, dtype=float)
    wn_norm = (wn - 400.0) / (4000.0 - 400.0) * 2.0 - 1.0  # normalize to [-1, 1]
    freqs = 2.0 ** np.arange(num_freqs, dtype=float)
    angles = np.pi * wn_norm[:, None] * freqs[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def intensity_projection(intensity: np.ndarray, dim: int = 16) -> np.ndarray:
    """Project intensity values to a feature vector per point.

    Parameters
    ----------
    intensity : ndarray, shape (n,)
        Normalized intensity values.
    dim : int
        Output feature dimension.

    Returns
    -------
    projected : ndarray, shape (n, dim)
        Per-point intensity features.
    """
    intensity = np.asarray(intensity, dtype=float)
    n = len(intensity)
    rng = np.random.default_rng(42)
    W = rng.normal(0.0, 0.3, size=(1, dim))
    b = rng.normal(0.0, 0.1, size=(1, dim))
    return np.tanh(intensity[:, None] * W + b)


def build_coordinate_embedding(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    hidden_dim: int = 16,
) -> np.ndarray:
    """Build coordinate-aware spectral token embeddings.

    h_i = W_I * I_i + p(nu_i)

    Parameters
    ----------
    wavenumber : ndarray, shape (n,)
        Wavenumber positions.
    intensity : ndarray, shape (n,)
        Normalized intensity values.
    hidden_dim : int
        Token feature dimension.

    Returns
    -------
    tokens : ndarray, shape (n, hidden_dim)
        Per-token embeddings.
    """
    coord_feats = fourier_coordinate_encoding(wavenumber)
    inten_feats = intensity_projection(intensity, dim=hidden_dim)
    rng = np.random.default_rng(7)
    W_proj = rng.normal(0.0, 0.3, size=(coord_feats.shape[1], hidden_dim))
    coord_proj = coord_feats @ W_proj
    tokens = inten_feats + 0.3 * coord_proj
    return np.tanh(tokens)


class CoordinateAwareSpectralEncoder:
    """Encode a preprocessed spectrum into a fixed-length embedding vector.

    The encoder:
      1. Builds coordinate-aware token embeddings
      2. Applies spectral-distance-bias attention
      3. Pools attended tokens to a global embedding

    This is a deterministic demo encoder (no training required).
    """

    def __init__(self, hidden_dim: int = 16, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.attention = SpectralDistanceBiasAttention(
            hidden_dim=hidden_dim, seed=seed
        )

    def encode(self, preprocessed: dict) -> np.ndarray:
        """Encode a preprocessed spectrum to a global embedding.

        Parameters
        ----------
        preprocessed : dict
            Must have keys 'wavenumber' (ndarray) and 'intensity' (ndarray).

        Returns
        -------
        embedding : ndarray, shape (2 * hidden_dim,)
            Global embedding (concatenation of mean and max pooling).
        attended_tokens : ndarray, shape (n, hidden_dim)
        attention : ndarray, shape (n, n)
        """
        wn = preprocessed["wavenumber"]
        intensity = preprocessed["intensity"]
        tokens = build_coordinate_embedding(wn, intensity, self.hidden_dim)
        attended, attn = self.attention.forward(tokens, wn)
        embedding = np.concatenate([
            attended.mean(axis=0),
            attended.max(axis=0),
        ])
        return embedding, attended, attn
