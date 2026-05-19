"""Spectral-distance-bias attention module.

Formula: A_ij = softmax(Q_i K_j^T / sqrt(d) + b(|nu_i - nu_j|))

where b(|nu_i - nu_j|) is a learned bias based on wavenumber distance.
"""

from __future__ import annotations

import numpy as np


class SpectralDistanceBiasAttention:
    """Coordinate-aware attention with spectral distance bias.

    This is a lightweight NumPy implementation for the Stage-I demo.
    A PyTorch equivalent would be used in a trained production model.

    The bias term b(|nu_i - nu_j|) uses an RBF basis over wavenumber differences.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        num_rbf: int = 8,
        bias_sigma: float = 150.0,
        attention_scale: float = 0.8,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.bias_sigma = bias_sigma
        self.attention_scale = attention_scale
        rng = np.random.default_rng(seed)
        self._W_q = rng.normal(0.0, 0.3, size=(hidden_dim, hidden_dim))
        self._W_k = rng.normal(0.0, 0.3, size=(hidden_dim, hidden_dim))
        self._W_v = rng.normal(0.0, 0.3, size=(hidden_dim, hidden_dim))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shift = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x_shift)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def compute_distance_bias(self, wavenumber: np.ndarray) -> np.ndarray:
        """Compute relative-distance bias matrix.

        Parameters
        ----------
        wavenumber : ndarray, shape (n,)
            Wavenumber positions on the common grid (subsampled).

        Returns
        -------
        bias : ndarray, shape (n, n)
            Bias values b(|nu_i - nu_j|) = exp(-(Delta_nu / sigma)^2).
        """
        wn = np.asarray(wavenumber, dtype=float)
        delta = np.abs(wn[:, None] - wn[None, :])
        return np.exp(-((delta / self.bias_sigma) ** 2))

    def forward(
        self, tokens: np.ndarray, wavenumber: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply spectral-distance-bias attention.

        Parameters
        ----------
        tokens : ndarray, shape (n, hidden_dim)
            Input token features (from encoder).
        wavenumber : ndarray, shape (n,)
            Wavenumber positions for each token.

        Returns
        -------
        output : ndarray, shape (n, hidden_dim)
            Contextualized token representations.
        attention : ndarray, shape (n, n)
            Attention weight matrix.
        """
        Q = tokens @ self._W_q
        K = tokens @ self._W_k
        V = tokens @ self._W_v
        scale = np.sqrt(self.hidden_dim)
        bias = self.compute_distance_bias(wavenumber)
        logits = (Q @ K.T) / scale + self.attention_scale * bias
        attention = self._softmax(logits, axis=-1)
        output = attention @ V
        return output, attention


def compute_attention_heatmap(
    attention_matrix: np.ndarray,
    downsample: int = 4,
) -> np.ndarray:
    """Downsample an attention matrix for visualization.

    Parameters
    ----------
    attention_matrix : ndarray, shape (n, n)
        Full attention weight matrix.
    downsample : int
        Downsampling factor (default 4, yielding ~450x450 for 1800 tokens).

    Returns
    -------
    heatmap : ndarray, shape (n//downsample, n//downsample)
        Downsampled attention matrix.
    """
    n = attention_matrix.shape[0]
    target = n // downsample
    if target < 2:
        return attention_matrix
    # Block-average downsample
    trimmed = attention_matrix[: target * downsample, : target * downsample]
    return trimmed.reshape(target, downsample, target, downsample).mean(axis=(1, 3))
