from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spectra_preprocess import ProcessedSpectrum


@dataclass
class EncodedModality:
    modality: str
    token_features: np.ndarray
    attended_tokens: np.ndarray
    global_embedding: np.ndarray
    attention: np.ndarray
    relative_bias: np.ndarray


@dataclass
class MultiModalEncoding:
    ir: EncodedModality | None
    raman: EncodedModality | None
    fused_embedding: np.ndarray
    metadata: dict[str, Any]


class CoordinateAwareSpectralEncoder:
    """Deterministic NumPy encoder for demo-only coordinate-aware spectral attention."""

    def __init__(self, hidden_dim: int = 16, num_rbf: int = 8, bias_sigma: float = 180.0, attention_scale: float = 0.75, conv_kernel_sizes: list[int] | None = None, seed: int = 7):
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.bias_sigma = bias_sigma
        self.attention_scale = attention_scale
        self.conv_kernel_sizes = conv_kernel_sizes or [3, 5, 9]
        rng = np.random.default_rng(seed)
        feature_dim = (3 + len(self.conv_kernel_sizes)) + 1 + (5 + num_rbf)
        self._projection = rng.normal(0.0, 0.35, size=(feature_dim, hidden_dim))
        self._q = rng.normal(0.0, 0.30, size=(hidden_dim, hidden_dim))
        self._k = rng.normal(0.0, 0.30, size=(hidden_dim, hidden_dim))
        self._v = rng.normal(0.0, 0.30, size=(hidden_dim, hidden_dim))
        self._fusion = rng.normal(0.0, 0.25, size=(hidden_dim * 4, hidden_dim * 2))

    @staticmethod
    def _softmax(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _coord_features(self, wavenumber: np.ndarray) -> np.ndarray:
        wn = np.asarray(wavenumber, dtype=float)
        wn_norm = (wn - wn.min()) / max(1e-8, wn.max() - wn.min())
        centers = np.linspace(wn.min(), wn.max(), self.num_rbf)
        rbfs = [np.exp(-((wn - c) ** 2) / (2 * self.bias_sigma**2)) for c in centers]
        base = np.column_stack([
            wn_norm,
            np.sin(2 * np.pi * wn_norm),
            np.cos(2 * np.pi * wn_norm),
            np.sin(4 * np.pi * wn_norm),
            np.cos(4 * np.pi * wn_norm),
        ])
        return np.column_stack([base, *rbfs])

    def _local_conv_features(self, intensity: np.ndarray) -> np.ndarray:
        intensity = np.asarray(intensity, dtype=float)
        features = []
        for kernel_size in self.conv_kernel_sizes:
            kernel = np.ones(kernel_size, dtype=float) / kernel_size
            smooth = np.convolve(intensity, kernel, mode="same")
            features.append(smooth)
        grad = np.gradient(intensity)
        curvature = np.gradient(grad)
        return np.column_stack([intensity, grad, curvature, *features])

    def build_relative_bias(self, wavenumber: np.ndarray) -> np.ndarray:
        wn = np.asarray(wavenumber, dtype=float)
        pairwise = np.abs(wn[:, None] - wn[None, :])
        return np.exp(-((pairwise / self.bias_sigma) ** 2))

    def encode_modality(self, spectrum: ProcessedSpectrum) -> EncodedModality:
        coord = self._coord_features(spectrum.wavenumber)
        local = self._local_conv_features(spectrum.intensity)
        features = np.column_stack([
            local,
            np.abs(local[:, 1:2]),
            coord,
        ])
        tokens = np.tanh(features @ self._projection)
        q = tokens @ self._q
        k = tokens @ self._k
        v = tokens @ self._v
        relative_bias = self.build_relative_bias(spectrum.wavenumber)
        logits = (q @ k.T) / np.sqrt(self.hidden_dim) + self.attention_scale * relative_bias
        attn = self._softmax(logits, axis=-1)
        attended = attn @ v
        global_embedding = np.concatenate([attended.mean(axis=0), attended.max(axis=0)])[: self.hidden_dim]
        return EncodedModality(
            modality=spectrum.modality,
            token_features=tokens,
            attended_tokens=attended,
            global_embedding=global_embedding,
            attention=attn,
            relative_bias=relative_bias,
        )

    def fuse_modalities(self, ir: EncodedModality | None, raman: EncodedModality | None) -> tuple[np.ndarray, dict[str, Any]]:
        if ir is None and raman is None:
            raise ValueError("At least one modality must be provided to the encoder.")
        if ir is None:
            return np.concatenate([raman.global_embedding, raman.global_embedding])[: self.hidden_dim * 2], {"modalities": ["raman"], "fusion_mode": "single_modality"}
        if raman is None:
            return np.concatenate([ir.global_embedding, ir.global_embedding])[: self.hidden_dim * 2], {"modalities": ["ir"], "fusion_mode": "single_modality"}
        ir_vec = ir.global_embedding
        ra_vec = raman.global_embedding
        compat = np.concatenate([ir_vec, ra_vec, np.abs(ir_vec - ra_vec), ir_vec * ra_vec])
        fused = np.tanh(compat @ self._fusion)
        cosine = float(np.dot(ir_vec, ra_vec) / (np.linalg.norm(ir_vec) * np.linalg.norm(ra_vec) + 1e-8))
        return fused, {"modalities": ["ir", "raman"], "fusion_mode": "cross_modal_feature_compatibility", "ir_raman_cosine": cosine}

    def encode(self, ir: ProcessedSpectrum | None, raman: ProcessedSpectrum | None) -> MultiModalEncoding:
        ir_encoded = self.encode_modality(ir) if ir is not None else None
        raman_encoded = self.encode_modality(raman) if raman is not None else None
        fused, metadata = self.fuse_modalities(ir_encoded, raman_encoded)
        return MultiModalEncoding(ir=ir_encoded, raman=raman_encoded, fused_embedding=fused, metadata=metadata)
