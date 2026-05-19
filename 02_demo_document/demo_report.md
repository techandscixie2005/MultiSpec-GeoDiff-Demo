# MultiSpec-GeoDiff Stage-I Demo Report

## 1. Demo Purpose

This demo validates the first stage of the MultiSpec-GeoDiff proposal:
an experimental IR spectrum → Top-K molecular candidate ranking workflow.
It demonstrates the core ideas of coordinate-aware spectral encoding and
spectral-distance-bias attention on real NIST data, without requiring a
trained graph diffusion or SE(3)-equivariant model.

The demo is a **feasibility workflow demonstration**, not a benchmark result.
Ground truth may not appear in the Top-K output, which is reported honestly.

## 2. Connection to Project Proposal

MultiSpec-GeoDiff formulates molecular structure inversion from spectra as a
multi-stage generation problem:

```
S_multi → (c, C) → G_2D → D → R^(0) → (R, T) → Ŝ → TopK
```

| Stage | Description | Demo status |
|---|---|---|
| **Stage I** | Spectral encoding, candidate retrieval, reranking | ✅ Implemented |
| **Stage II** | Graph diffusion + pairwise distance prediction | 🔜 Stub |
| **Stage III** | TFN-Transformer parity-aware refinement | 🔜 Stub |

This demo covers Stage I fully. Stages II and III are present as importable
interface stubs only (see `03_code/src/distance_head.py` and
`03_code/src/tfn_transformer_stub.py`).

## 3. Dataset: 200 Experimental NIST IR Spectra

- **File**: `04_data/IR_nist_200.jsonl`
- **Records**: 200 experimental IR spectra from the NIST database
- **Format**: JSONL with `smiles` and `value: {x: wavenumbers, y: intensities}`
- **Split**: One record selected by `--query_id` as query; remaining 199 as library
- **Preprocessing**: Each spectrum is interpolated to a common grid
  (400–4000 cm⁻¹, 1800 points) and min-max normalized before processing.
  Peaks are extracted via scipy.signal.find_peaks.

## 4. Stage-I Pipeline

```
query IR spectrum
  → load & validate JSONL
  → interpolate to common grid (400–4000 cm⁻¹)
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

Each spectral point is embedded as a combination of its intensity and its
physical wavenumber coordinate:

```
h_i = W_I · I_i + p(ν_i)
```

where `I_i` is the normalized intensity at point i, `p(ν_i)` is a Fourier
encoding (sin/cos at multiple frequencies) of the wavenumber ν_i, and `W_I`
is a learned linear projection. This respects the physical meaning of the
x-axis rather than treating the spectrum as a flat sequence.

### 5.2 Spectral-Distance-Bias Attention

Self-attention augmented with a relative-distance bias term that encodes the
physical intuition that nearby wavenumbers interact more strongly:

```
A_ij = softmax(Q_i K_j^T / √d + b(|ν_i − ν_j|))
b(Δν) = exp(−(Δν / σ)²)
```

The RBF kernel `b(Δν)` decays with wavenumber separation, so attention
weights between distant spectral regions are suppressed unless their feature
content is highly correlated. This is the spectral analogue of
coordinate-aware attention in the full proposal.

### 5.3 Candidate Retrieval

After encoding, the query spectrum is represented as a fixed-dimensional
embedding vector via global average pooling of the attended token sequence.
Cosine similarity is computed against all 199 library embeddings. The
Top-M most similar candidates are selected for reranking.

### 5.4 Forward-Spectrum Reranking

Retrieved candidates are reranked by fine-grained spectral similarity against
the query spectrum using three complementary metrics:

- **Cosine similarity** of full spectrum vectors
- **L1 similarity** (inverse of L1 distance)
- **Peak matching score** — agreement in peak positions and intensities
  between query and candidate

The final score is a weighted combination of the retrieval score and the
forward-spectrum score, mirroring the forward-spectrum consistency principle
from the full proposal.

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

- **Baseline**: Raw spectral cosine similarity (no encoding, no attention)
- **Full**: Coordinate-aware encoding + distance-bias attention + reranking

The ablation report is saved to `05_outputs/ablation_results.csv`.

## 8. Limitations

- **200-molecule library** is small; retrieval accuracy is not benchmark-level.
- **Deterministic encoder** uses random projections, not learned parameters.
- **IR only** — Raman and other modalities are not included.
- **No graph diffusion** — retrieval is the Stage-I substitute for generation.
- **No 3D refinement** — TFN-Transformer is an interface stub only.
- **Experimental noise** and variable resolution affect cross-spectrum comparison.

## 9. Stage-II/III Roadmap

| Stage | Module | Description | Status |
|---|---|---|---|
| **Stage II** | Graph diffusion | Size-adaptive de-novo candidate generation | 🔜 Stub |
| | Distance head | 2D graph → pairwise distance matrix D | 🔜 Stub |
| **Stage III** | TFN-Transformer | SE(3)-equivariant refinement of 3D geometry | 🔜 Stub |
| | Parity channels | 0e, 0o, 1e, 1o, 2e, 2o tensor features | 🔜 Design only |

## 10. Submission Checklist

- [x] Runnable demo CLI (`python 03_code/run_demo.py --query_id 0 ...`)
- [x] Experimental NIST IR dataset (200 spectra, `04_data/`)
- [x] All Stage-I source modules (`03_code/src/`)
- [x] Test suite (pytest, `03_code/tests/`)
- [x] CI configuration (GitHub Actions, `.github/workflows/tests.yml`)
- [x] Notebook walkthrough (`06_notebook/demo_pipeline.ipynb`)
- [x] Generated outputs (`05_outputs/`)
- [x] Demo report (this document)
- [x] Project proposal (`01_project_proposal/proposal.md`)
- [x] Trace log with full execution record
