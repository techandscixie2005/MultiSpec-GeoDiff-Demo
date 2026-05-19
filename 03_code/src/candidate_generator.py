"""Candidate molecule retrieval for Stage-I.

Current demo uses spectral similarity retrieval as the candidate generator.
Future versions will replace this module with size-adaptive graph diffusion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metrics import cosine_similarity
from spectra_encoder import CoordinateAwareSpectralEncoder


def build_library_embeddings(
    preprocessed_records: list[dict[str, Any]],
    encoder: CoordinateAwareSpectralEncoder,
) -> tuple[np.ndarray, list[str]]:
    """Encode all library spectra into embedding vectors.

    Parameters
    ----------
    preprocessed_records : list[dict]
        List of preprocessed library records.
    encoder : CoordinateAwareSpectralEncoder
        Encoder instance.

    Returns
    -------
    embeddings : ndarray, shape (n_library, 2 * hidden_dim)
        All library embeddings.
    smiles_list : list[str]
        Corresponding SMILES strings.
    """
    embeddings: list[np.ndarray] = []
    smiles_list: list[str] = []
    for rec in preprocessed_records:
        emb, _, _ = encoder.encode(rec)
        embeddings.append(emb)
        smiles_list.append(rec["smiles"])
    return np.array(embeddings), smiles_list


def retrieve_top_m(
    query_embedding: np.ndarray,
    library_embeddings: np.ndarray,
    top_m: int = 10,
) -> tuple[list[int], list[float]]:
    """Retrieve top-M candidates by cosine similarity.

    Parameters
    ----------
    query_embedding : ndarray, shape (emb_dim,)
        Query embedding vector.
    library_embeddings : ndarray, shape (n_library, emb_dim)
        Library embeddings.
    top_m : int
        Number of candidates to retrieve.

    Returns
    -------
    indices : list[int]
        Indices of top-M candidates in the library array.
    scores : list[float]
        Cosine similarity scores.
    """
    scores = [cosine_similarity(query_embedding, lib_emb) for lib_emb in library_embeddings]
    sorted_indices = np.argsort(scores)[::-1][:top_m]
    return sorted_indices.tolist(), [scores[i] for i in sorted_indices]
