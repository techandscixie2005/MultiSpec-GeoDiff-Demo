# MultiSpec-GeoDiff: Multimodal Spectra-driven Molecular Structure Inversion

**Stage-I Demo**: Spectral encoding → candidate retrieval → forward-spectrum reranking on 200 experimental NIST IR spectra.

> **Honest scope**: This demo implements retrieval/reranking only. Graph diffusion (Stage-II) and TFN-Transformer refinement (Stage-III) are interface stubs, not trained models.

## Quick Start

```bash
# Install dependencies
pip install -r 03_code/requirements.txt

# Run the Stage-I demo
python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5

# Run tests
PYTHONPATH=03_code/src pytest 03_code/tests -q
```

## What This Demo Proves

| Proves ✅ | Does not prove ❌ |
|---|---|
| Runnable IR → Top-K candidate ranking workflow | Real experimental benchmark performance |
| Coordinate-aware spectral encoding with $\Delta\nu$ distance bias | Trained graph diffusion for de-novo molecule generation |
| Forward-spectrum consistency reranking | Trained pairwise distance prediction (2D→3D) |
| Traceable, inspectable outputs (CSV, JSON, figures, trace log) | Trained TFN-Transformer parity-aware refinement |

## Pipeline

```
query IR spectrum
  → load & validate JSONL
  → interpolate to common grid (400–4000 cm^-1)
  → normalize intensities
  → build coordinate-aware spectral embeddings (h_i = W_I I_i + p(ν_i))
  → spectral-distance-bias attention (A_ij = softmax(QK^T/√d + b(|ν_i-ν_j|)))
  → global embedding pooling
  → cosine similarity retrieval (Top-M)
  → forward-spectrum similarity reranking (Top-K)
  → visual outputs & trace log
```

## Repository Structure

```
├── README.md
├── 02_demo_document/
│   └── demo_report.md          # Technical demo report
├── 03_code/
│   ├── run_demo.py             # CLI entry point
│   ├── requirements.txt
│   ├── src/                    # Core modules
│   │   ├── data_loader.py      # JSONL loading & splitting
│   │   ├── spectra_preprocess.py  # Interpolation, normalization, peaks
│   │   ├── spectra_encoder.py  # Coordinate-aware encoder
│   │   ├── spectral_attention.py  # Δν-distance-bias attention
│   │   ├── candidate_generator.py  # Cosine similarity retrieval
│   │   ├── reranker.py         # Forward-spectrum reranking
│   │   ├── metrics.py          # Similarity metrics
│   │   ├── visualization.py    # Plotting utilities
│   │   ├── distance_head.py    # Stage-II stub
│   │   └── tfn_transformer_stub.py  # Stage-III stub
│   └── tests/                  # pytest suite (22 tests)
├── 04_data/
│   ├── IR_nist_200.jsonl       # 200 NIST IR spectra
│   └── data_card.md            # Dataset documentation
├── 05_outputs/                 # Generated outputs
│   ├── topk_candidates.csv
│   ├── spectrum_overlay.png
│   ├── attention_heatmap.png
│   ├── molecule_grid.png
│   ├── ablation_results.csv
│   └── trace_log.json
├── 06_notebook/
│   └── demo_pipeline.ipynb     # Interactive walkthrough
└── proposal.md                 # Full research proposal (Chinese)
```

## Dataset Format

`04_data/IR_nist_200.jsonl` contains 200 experimental NIST IR spectra.
Each line is a JSON object:
```json
{
  "smiles": "NNc1ccc([N+](=O)[O-])cc1",
  "value": {"x": [wavenumbers], "y": [intensities]},
  "source": "NIST"
}
```

## Implemented Modules (Stage-I)

| Module | Description |
|---|---|
| `data_loader` | JSONL loading, validation, query/library splitting |
| `spectra_preprocess` | Interpolation to 400-4000 cm⁻¹ grid, min-max normalization, peak extraction |
| `spectra_encoder` | Fourier coordinate encoding + intensity projection → token embeddings |
| `spectral_attention` | Self-attention with RBF wavenumber-distance bias |
| `candidate_generator` | Cosine similarity retrieval from 199-molecule library |
| `reranker` | Forward-spectrum reranking (cosine + L1 + peak match) |
| `metrics` | Cosine similarity, L1 distance, peak matching, top-k hit rate |
| `visualization` | Spectrum overlay, attention heatmap, molecule grid (RDKit optional) |

## Future Modules

| Stage | Module | Status |
|---|---|---|
| **Stage II** | Size-adaptive graph diffusion | 🔜 Stub |
| | Pairwise distance prediction (2D→3D bridge) | 🔜 Stub |
| **Stage III** | Parity-aware TFN-Transformer (0e, 0o, 1e, 1o, 2e, 2o) | 🔜 Stub |

## Known Limitations

- 200-molecule library is small; retrieval accuracy is not benchmark-level
- Encoder uses random projections, not learned parameters
- Only IR modality; Raman extension is straightforward
- No de-novo generation (graph diffusion not implemented)
- No 3D geometry refinement (TFN-Transformer is a stub)

## Technical Background

- **Coordinate-aware encoding**: Inspired by NeRF's Fourier features and coordinate-based MLPs
- **Spectral distance bias**: Attention with physical wavenumber-distance bias
- **Forward-spectrum reranking**: Multi-metric comparison (cosine, L1, peak match)
- **Full proposal**: See `proposal.md` for the complete research vision
