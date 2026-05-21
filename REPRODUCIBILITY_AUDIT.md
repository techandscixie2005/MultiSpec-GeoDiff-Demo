# Reproducibility Audit: MultiSpec-GeoDiff-Demo (Stage-I)

**Audit date:** 2026-05-21  
**Audit branch:** `audit/reproducibility-check`  
**Audited commit:** `55c43dcf25b9772dba563bf8af659bdb5e2f2171`  
**Auditor:** Claude Code (independent reproducibility check)  
**Artifacts:** `audit_artifacts/reproducibility_check/`

---

## 1. Executive Verdict

| Criterion | Verdict |
|---|---|
| **Runnable** | **YES** — demo runs successfully from a clean virtual environment |
| **Tests pass** | **YES** — 22/22 tests pass in 32.8s |
| **Results reproduced** | **YES** — CSV outputs bit-identical; JSON differs only in timestamp/env |
| **Repeated-run reproducibility** | **FULLY DETERMINISTIC** — 3 consecutive runs produce bit-identical CSV and PNG outputs |
| **Committed outputs consistent with code** | **YES** — committed CSVs match newly generated outputs exactly |
| **Submission risk level** | **LOW** — the demo is honest about scope, reproducible, and well-tested |

---

## 2. Evidence

### 2.1 Environment

| Property | Value |
|---|---|
| **Commit** | `55c43dcf25b9772dba563bf8af659bdb5e2f2171` |
| **Python** | 3.10.18 |
| **OS** | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64 |
| **Shell** | bash (WSL2) |
| **Clean venv** | `.venv_audit` (Python 3.10.18, pip 26.1.1) |

### 2.2 Installed Packages (audit venv)

```
numpy==2.2.6, pandas==2.3.3, scipy==1.15.3, scikit-learn==1.7.2
matplotlib==3.10.9, tqdm==4.67.3, pytest==9.0.3
rdkit==2026.3.2, cairosvg==2.9.0
```

Note: `torch` is listed as optional in `requirements.txt` and is not needed to run the demo. The committed trace_log shows torch was installed in the original environment, but the demo does not import it.

### 2.3 Commands Run

```bash
# Test command
PYTHONPATH=03_code/src pytest 03_code/tests -q   # 22 passed

# Documented demo command
python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5

# Committed trace_log parameters
python 03_code/run_demo.py --query_id 5 --data 04_data/IR_nist_200.jsonl --top_k 3
```

**Important note:** The documented README command uses `--query_id 0 --top_k 5`, but the committed `trace_log.json` was generated with `--query_id 5 --top_k 3`. Both parameter sets produce valid, deterministic outputs.

### 2.4 Test Summary

```
22 passed in 32.83s (verbose: 22 passed in 32.83s)

Breakdown:
  test_attention.py         5 tests (attention shapes, softmax, nonneg, bias, heatmap)
  test_data_loader.py       6 tests (load, validate, split, range, unique)
  test_demo_runs.py          4 tests (creates CSV, trace_log, query_id_1, all outputs)
  test_preprocess.py         7 tests (interpolate, irregular, errors, normalize, peaks)
```

0 failures, 0 skipped, 0 warnings.

### 2.5 Generated Output Files (query_id=5, top_k=3)

| File | Size | SHA256 (run 1) | Reproducible? |
|---|---|---|---|
| `topk_candidates.csv` | 330 B | `36a3ff1e...` | Bit-identical across 3 runs AND committed version |
| `ablation_results.csv` | 550 B | `c70df64f...` | Bit-identical across 3 runs AND committed version |
| `trace_log.json` | ~2.7 KB | varies | Timestamp/env differ; all result fields identical |
| `spectrum_overlay.png` | ~285 KB | `56abc2a5...` | Bit-identical across 3 runs |
| `molecule_grid.png` | ~104 KB | `7d9db3ea...` | Bit-identical across 3 runs |
| `attention_heatmap.png` | ~189 KB | `ebef8cb4...` | Bit-identical across 3 runs |

### 2.6 Comparison with Committed Outputs

| Output | Comparison Result |
|---|---|
| `topk_candidates.csv` | **IDENTICAL** — SHA256 matches exactly |
| `ablation_results.csv` | **IDENTICAL** — SHA256 matches exactly |
| `trace_log.json` | **Result fields identical**; timestamp and package versions differ (expected) |
| PNG files | Hash differs from committed (different matplotlib version → different rendering metadata), but all 3 re-runs produce identical PNGs within the new environment |

Scientific result fields in trace_log.json (query_id, query_smiles, parameters, top_candidates with all scores, ground_truth_in_top_k) are all **identical** to the committed version.

### 2.7 Repeated Runs — Determinism Confirmed

3 consecutive runs with `--query_id 5 --top_k 3` produce:
- **Identical** console output (same SMILES, same scores to 4 decimal places)
- **Identical** CSV files (SHA256 match)
- **Identical** PNG files (SHA256 match)
- trace_log.json differs only in `timestamp` field

**Source of determinism:** All random operations use fixed seeds (42, 7) hardcoded in `spectra_encoder.py` and `spectral_attention.py`. No dependency on hash ordering or non-deterministic iteration.

---

## 3. Claim Audit Table

| Claim | Source | Evidence in Code/Output | Status | Notes |
|---|---|---|---|---|
| "200 experimental NIST IR spectra" | README, demo_report | `04_data/IR_nist_200.jsonl`: 200 lines, each with SMILES + x/y + source="NIST" | **SUPPORTED** | |
| "Coordinate-aware spectral encoding" | README, proposal | `spectra_encoder.py`: h_i = W_I * I_i + p(ν_i) with Fourier position encoding | **SUPPORTED** | |
| "Spectral-distance-bias attention" | README, proposal | `spectral_attention.py`: A_ij = softmax(QK^T/√d + b(\|ν_i-ν_j\|)) with RBF bias | **SUPPORTED** | Verified by test suite + attention heatmap output |
| "Candidate retrieval (cosine similarity)" | README, demo_report | `candidate_generator.py`: cosine similarity retrieval, Top-M | **SUPPORTED** | |
| "Forward-spectrum reranking" | README, demo_report | `reranker.py`: cos + L1 + peak match weighted rerank | **SUPPORTED** | |
| "Top-K visualization" | README, demo_report | `visualization.py`: spectrum overlay, molecule grid, attention heatmap | **SUPPORTED** | All PNG files generated and non-empty |
| "Traceable outputs" | README, demo_report | `trace_log.json` generated with parameters, results, environment | **SUPPORTED** | See traceability gaps below |
| "CI tests" | README badge | `.github/workflows/tests.yml` exists; GH Actions passing on main | **SUPPORTED** | Latest 8 CI runs on main all pass |
| "Stage-I only; Stage-II/III are roadmap" | README honest scope table, demo_report §6 | `distance_head.py` and `tfn_transformer_stub.py` clearly labeled as stubs | **SUPPORTED** | Honest scope table in README is explicit |
| "Graph diffusion" | proposal §2, README roadmap | Not implemented; README marks as "Stub" | **CORRECTLY SCOPED** | Proposal presents as future work, README and demo_report mark as roadmap |
| "Pairwise distance prediction" | proposal §3, README roadmap | `distance_head.py`: stub returning heuristic matrix with status="stub" | **CORRECTLY SCOPED** | |
| "TFN/SE(3)-Transformer" | proposal §4, README roadmap | `tfn_transformer_stub.py`: stub returning status="stub", no real computation | **CORRECTLY SCOPED** | |
| "Multimodal spectra (Raman, NMR, MS, UV)" | proposal §1, README roadmap | Not implemented; IR only | **CORRECTLY SCOPED** | Marked as future in README |
| "Chirality-aware tensor channels (0e/0o/1e/1o/2e/2o)" | proposal §5, README roadmap | `tfn_transformer_stub.py`: channel names listed in stub docstring only | **CORRECTLY SCOPED** | |

**No overclaims detected.** All Stage-II/III capabilities are consistently marked as "Stub", "Roadmap-only", or "Future" across README, demo_report, proposal, and source code. The proposal presents the full vision but explicitly states Stage-I is implemented and Stages II/III are future work.

---

## 4. Limitations

- **Stage-I only** — retrieval and reranking, not de-novo generation
- **No trained graph diffusion** — candidate generation is retrieval-based, not generative
- **No de-novo generation** — candidates come from the existing 199-molecule library
- **No trained pairwise-distance model** — `distance_head.py` is a stub
- **No trained TFN/SE(3) refinement** — `tfn_transformer_stub.py` is a stub
- **IR only** — single modality; Raman, NMR, MS, UV not implemented
- **200-molecule library** — not benchmark-scale; retrieval accuracy claims would be invalid at this scale
- **Deterministic random encoder** — uses fixed random projections, not a learned neural network

---

## 5. Recommended Minimal Fixes Before Submission

### 5.1 CRITICAL — None

No critical issues found. The demo runs, tests pass, and outputs reproduce.

### 5.2 RECOMMENDED — Traceability improvements

The current `trace_log.json` is missing several fields that would strengthen reproducibility claims:

1. **Git commit hash** — add `git rev-parse HEAD` output to trace_log
2. **Exact CLI command** — record `sys.argv` in trace_log
3. **SHA256 of input data file** — add hash of `IR_nist_200.jsonl`
4. **SHA256 of output files** — add hashes of generated outputs
5. **Repository dirty/clean status** — record `git status --porcelain` output
6. **Random seed** — record the hardcoded seed values (currently implicit)

These are one-line additions to `_build_trace_log()` in `run_demo.py`.

### 5.3 SUGGESTED — Align committed outputs with documented command

The committed `trace_log.json` uses `query_id=5, top_k=3` but the documented command uses `query_id=0, top_k=5`. Consider either:
- Re-running and committing outputs with the documented parameters, OR
- Adding a note that the committed trace_log uses different parameters as an additional example

### 5.4 SUGGESTED — Add torch to requirements or remove from trace_log

`torch` is listed in `get_environment_summary()` but is not required to run the demo. The current audit venv has `torch: null` in trace_log. Either add torch as an optional dependency or explicitly mark it as optional in the environment summary.

---

## 6. Final Suggested Submission Statement

> This demo verifies a reproducible Stage-I IR retrieval/reranking feasibility loop on 200 experimental NIST IR spectra. The pipeline loads spectra, interpolates to a common wavenumber grid, encodes them with coordinate-aware Fourier features and spectral-distance-bias attention, retrieves candidates by cosine similarity, and reranks using three forward-spectrum consistency metrics (cosine, L1, peak match). All 22 tests pass, the demo runs from a clean environment with a single command, and repeated runs produce bit-identical CSV and PNG outputs. Stage-II (graph diffusion, pairwise distance) and Stage-III (TFN-Transformer, chirality-aware tensor channels) are documented as roadmap stubs with importable interfaces. The current implementation validates the "candidate proposal + forward consistency check" closed-loop concept on real experimental data, establishing a traceable baseline for future generative and 3D-equivariant extensions.

---

## 7. Audit Artifacts

All supporting evidence is saved under `audit_artifacts/reproducibility_check/`:

```
audit_artifacts/reproducibility_check/
├── 00_environment.txt              # Environment snapshot
├── 02_pip_freeze.txt               # Installed packages
├── original_05_outputs/            # Committed outputs (backed up before re-run)
├── test_generated_outputs/         # Outputs from test suite execution
├── run_documented/                 # Outputs from documented command (query_id=0, top_k=5)
├── run_committed_params/           # Outputs matching committed trace_log (query_id=5, top_k=3)
├── run_1/                          # Repeated run 1 (query_id=5, top_k=3)
├── run_2/                          # Repeated run 2 (query_id=5, top_k=3)
└── run_3/                          # Repeated run 3 (query_id=5, top_k=3)
```
