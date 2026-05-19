# MultiSpec-GeoDiff: Stage-I Demo for Spectra-driven Molecular Structure Inversion

## What this demo shows

This demo validates Stage-I feasibility on **200 experimental NIST IR spectra**.
Given a query IR spectrum, the pipeline retrieves and ranks candidate molecules
from a 199-molecule library using coordinate-aware spectral encoding,
spectral-distance-bias attention, and forward-spectrum reranking.

## Honest scope

| Implemented | Not implemented |
|---|---|
| NIST IR JSONL loading | Trained graph diffusion |
| Spectrum interpolation/normalization | Trained pairwise distance model |
| Coordinate-aware spectral encoding | Trained TFN-Transformer |
| Spectral-distance-bias attention | Full multimodal system |
| Retrieval + reranking | Production unknown-molecule identification |
| Top-K outputs + trace log | |

## Quick Start

```bash
# Install dependencies
pip install -r 03_code/requirements.txt

# Run the Stage-I demo
python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5

# Run tests
PYTHONPATH=03_code/src pytest 03_code/tests -q
```

## Expected outputs

All outputs are written to `05_outputs/`:

| File | Description |
|---|---|
| `05_outputs/topk_candidates.csv` | Ranked candidate table with scores |
| `05_outputs/spectrum_overlay.png` | Query vs Top-K spectra overlay |
| `05_outputs/molecule_grid.png` | Top-K molecular structures |
| `05_outputs/attention_heatmap.png` | Spectral distance bias attention pattern |
| `05_outputs/ablation_results.csv` | Simplified ablation comparison |
| `05_outputs/trace_log.json` | Full execution trace |

## Pipeline

```
query IR spectrum
  → JSONL loading
  → common-grid interpolation (400–4000 cm⁻¹, 1800 points)
  → intensity normalization
  → coordinate-aware encoding: h_i = W_I · I_i + p(ν_i)
  → spectral-distance-bias attention: A_ij = softmax(Q_i K_j^T / √d + b(|ν_i − ν_j|))
  → global embedding pooling
  → cosine similarity retrieval (Top-M)
  → forward-spectrum reranking (cosine + L1 + peak match)
  → Top-K outputs + trace log
```

## Repository structure

```
├── README.md
├── 01_project_proposal/
│   └── proposal.md              # Research proposal
├── 02_demo_document/
│   ├── demo_report.md           # Technical demo report
│   └── demo_report.pdf          # Generated PDF
├── 03_code/
│   ├── run_demo.py              # CLI entry point
│   ├── requirements.txt
│   ├── src/                     # Core modules (see below)
│   └── tests/                   # Pytest suite
├── 04_data/
│   ├── IR_nist_200.jsonl        # 200 NIST IR spectra
│   ├── sample_query.jsonl
│   └── data_card.md
├── 05_outputs/                  # Generated demo outputs
├── 06_notebook/
│   └── demo_pipeline.ipynb      # Interactive walkthrough
├── .github/workflows/
│   └── tests.yml                # CI configuration
└── legacy/
    └── toy_ir_raman_demo/       # Archived earlier prototype
```

## Dataset

`04_data/IR_nist_200.jsonl` contains 200 experimental NIST IR spectra.
Each line is a JSON object:

```json
{
  "smiles": "NNc1ccc([N+](=O)[O-])cc1",
  "value": {"x": [wavenumbers], "y": [intensities]},
  "source": "NIST"
}
```

One molecule is selected by `--query_id` as the query; the remaining 199 form
the candidate library. See `04_data/data_card.md` for details.

## Implemented modules

All Stage-I modules live in `03_code/src/`:

| Module | File | Description |
|---|---|---|
| `data_loader` | `data_loader.py` | JSONL loading, validation, query/library split |
| `spectra_preprocess` | `spectra_preprocess.py` | Interpolation, normalization, peak extraction |
| `spectra_encoder` | `spectra_encoder.py` | Fourier coordinate encoding + intensity projection |
| `spectral_attention` | `spectral_attention.py` | Self-attention with RBF wavenumber-distance bias |
| `candidate_generator` | `candidate_generator.py` | Cosine similarity retrieval |
| `reranker` | `reranker.py` | Forward-spectrum reranking (cos + L1 + peak) |
| `metrics` | `metrics.py` | Similarity and evaluation metrics |
| `visualization` | `visualization.py` | Spectrum overlay, attention heatmap, molecule grid |
| `distance_head` | `distance_head.py` | Stage-II stub: 2D graph → pairwise distance |
| `tfn_transformer_stub` | `tfn_transformer_stub.py` | Stage-III stub: SE(3)-equivariant refinement |

## Future roadmap

| Stage | Module | Status |
|---|---|---|
| **Stage II** | Size-adaptive graph diffusion | Stub |
| | Pairwise distance matrix D (2D→3D bridge) | Stub |
| **Stage III** | Parity-aware TFN-Transformer (0e, 0o, 1e, 1o, 2e, 2o) | Stub |

## Limitations

- **200 molecules only** — library is too small for benchmark-level retrieval.
- **IR only** — Raman and other modalities not yet included.
- **Retrieval substitutes for graph diffusion** — no de-novo candidate generation.
- **Random/deterministic encoder** — not a fully trained neural network.
- **No 3D refinement** — TFN-Transformer is an interface stub.
- **Not production** — not suitable for unknown-compound identification.
