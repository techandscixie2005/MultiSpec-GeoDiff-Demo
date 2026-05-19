# MultiSpec-GeoDiff Stage-I Demo Report

## 1. Problem Background

Molecular structure identification from spectroscopic data is a fundamental
inverse problem in chemistry. Given an experimental IR spectrum, the goal is to
identify the molecular structure that produced it. This is a one-to-many
ill-posed problem: multiple molecules can produce similar spectra, making
direct prediction unreliable.

**MultiSpec-GeoDiff** formulates this as a multi-stage pipeline:

1. Encode the query spectrum into a coordinate-aware representation
2. Generate or retrieve candidate molecules
3. Refine candidates via forward-spectrum consistency
4. Output Top-K ranked candidates

## 2. Proposal Connection

This demo implements **Stage-I** of the proposal: spectral encoding, candidate
retrieval, and forward-spectrum reranking on a real experimental IR dataset.
Stages II and III (graph diffusion, pairwise distance prediction, TFN-Transformer
refinement) are present as importable interface stubs only.

## 3. Dataset: 200 NIST Experimental IR Spectra

- **File**: `04_data/IR_nist_200.jsonl`
- **Records**: 200 experimental IR spectra from NIST
- **Format**: JSONL with `smiles` and `value: {x: wavenumbers, y: intensities}`
- **Split**: One record selected by `--query_id` as query; remaining 199 as library

Each spectrum is interpolated to a common grid (400–4000 cm^-1, 1800 points)
and min-max normalized before processing.

## 4. Demo Pipeline

```
query IR spectrum
  → load & validate JSONL
  → interpolate to common grid (400–4000 cm^-1)
  → normalize intensities
  → build coordinate-aware spectral embeddings
  → apply spectral-distance-bias attention
  → global embedding pooling
  → cosine similarity retrieval (Top-M)
  → forward-spectrum similarity reranking (Top-K)
  → visualization outputs & trace log
```

## 5. Core Implemented Ideas

### 5.1 Coordinate-Aware Spectral Encoding

Each point in the spectrum is represented as:

$$h_i = W_I I_i + p(\nu_i)$$

where $I_i$ is the intensity, $p(\nu_i)$ is a Fourier coordinate encoding
of the wavenumber, and $W_I$ is a learned projection. This respects the
physical meaning of the x-axis (wavenumber) rather than treating the
spectrum as a flat sequence.

### 5.2 Spectral Distance Bias Attention

Self-attention with a relative-distance bias term:

$$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt d} + b(|\nu_i - \nu_j|)\right)$$

where $b(|\nu_i - \nu_j|) = \exp(-(\Delta\nu / \sigma)^2)$ encodes the
physical intuition that nearby wavenumbers should have stronger attention
interactions. This is the spectral analogue of coordinate-aware attention
in the full proposal.

### 5.3 Candidate Retrieval

Cosine similarity search over spectral embeddings. The query embedding is
compared against all 199 library embeddings, returning Top-M candidates.

### 5.4 Forward-Spectrum Reranking

Retrieved candidates are reranked by fine-grained spectral similarity:

- Cosine similarity of full spectra
- Peak matching score (position and intensity agreement)
- Weighted final score

This mirrors the forward-spectrum consistency principle from the full proposal:
candidates whose experimental spectrum best matches the query are ranked higher.

## 6. Example Outputs

| Output | Description |
|---|---|
| `05_outputs/topk_candidates.csv` | Ranked candidate table with scores |
| `05_outputs/spectrum_overlay.png` | Query vs Top-K spectra overlay |
| `05_outputs/attention_heatmap.png` | Spectral distance bias attention pattern |
| `05_outputs/molecule_grid.png` | Top-K molecular structures |
| `05_outputs/trace_log.json` | Full execution trace |
| `05_outputs/ablation_results.csv` | Simplified ablation comparison |

## 7. Minimal Ablation

A simplified ablation compares two retrieval strategies:
- **Baseline**: Raw spectral cosine similarity (no encoding)
- **Full**: Coordinate-aware encoding + distance-bias attention + reranking

The ablation report is saved to `05_outputs/ablation_results.csv`.

## 8. Limitations

- 200-molecule library is small; retrieval accuracy is not benchmark-level
- The encoder uses deterministic random projections, not learned parameters
- Only IR modality is used; Raman addition is straightforward
- No graph diffusion (Stage-II) — retrieval is the substitute
- No 3D refinement (Stage-III) — TFN-Transformer is an interface stub only
- Experimental noise and grid resolution affect cross-spectra comparison

## 9. Roadmap

| Stage | Module | Status |
|---|---|---|
| **Stage I** | Spectral encoding & attention | ✅ Implemented |
| | Candidate retrieval | ✅ Implemented |
| | Forward-spectrum reranking | ✅ Implemented |
| **Stage II** | Size-adaptive graph diffusion | 🔜 Stub |
| | Pairwise distance prediction | 🔜 Stub |
| **Stage III** | TFN-Transformer refinement | 🔜 Stub |
| | Parity channels (0e, 0o, 1e, 1o, 2e, 2o) | 🔜 Design only |
