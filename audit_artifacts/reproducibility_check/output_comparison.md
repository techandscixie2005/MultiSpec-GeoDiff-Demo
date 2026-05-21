# Output Comparison: Committed vs. Newly Generated

## Comparison Method

- **Committed outputs:** Restored from git at commit `55c43dc`
- **New outputs:** Generated in clean `.venv_audit` environment
- **Identical command:** `python 03_code/run_demo.py --query_id 5 --data 04_data/IR_nist_200.jsonl --top_k 3`

## topk_candidates.csv

| Field | Committed | Generated |
|---|---|---|
| SHA256 | `36a3ff1e809d3f984da4377743a8d2feeac09d60316f86bfa5b472750bf94660` | `36a3ff1e...` (MATCH) |
| Rank 1 SMILES | COC(=O)C(Cl)(CCl)CCl | COC(=O)C(Cl)(CCl)CCl |
| Rank 1 final_score | 0.9158 | 0.9158 |
| Rank 2 SMILES | COc1ccc(N2CCN(C)c3ccccc3C2)cc1 | COc1ccc(N2CCN(C)c3ccccc3C2)cc1 |
| Rank 2 final_score | 0.9106 | 0.9106 |
| Rank 3 SMILES | COC(O)=NC(C)(C)c1ccccc1 | COC(O)=NC(C)(C)c1ccccc1 |
| Rank 3 final_score | 0.9067 | 0.9067 |

**Verdict: IDENTICAL (bit-level match)**

## ablation_results.csv

| Field | Committed | Generated |
|---|---|---|
| SHA256 | `c70df64f26f0b671762a5875d31f11631d0d67cd07afb34a63b098a648a78367` | `c70df64f...` (MATCH) |

**Verdict: IDENTICAL (bit-level match)**

## trace_log.json

| Field | Committed | Generated | Match? |
|---|---|---|---|
| demo_name | MultiSpec-GeoDiff Stage-I Demo | same | YES |
| query_id | 5 | 5 | YES |
| query_smiles | C/C(=C/CCC1...)COC(O)=Nc1ccccc1[N+](=O)[O-] | same | YES |
| parameters.top_k | 3 | 3 | YES |
| parameters.top_m | 20 | 20 | YES |
| results.ground_truth_in_top_k | false | false | YES |
| results.top_candidates | 3 candidates, all scores match to 4 d.p. | same | YES |
| implemented_modules | 8 modules listed | same | YES |
| outputs file paths | 6 output paths | same | YES |
| timestamp | 2026-05-19T15:44:31Z | 2026-05-21T09:26:01Z | DIFF (expected) |
| environment.packages | numpy 2.2.5, torch 2.8.0, rdkit 2025.03.6 | numpy 2.2.6, torch null, rdkit 2026.03.2 | DIFF (expected) |

**Verdict: Scientific result fields IDENTICAL; timestamp and environment differ as expected**

## PNG Files

| File | Committed SHA256 | Generated SHA256 | Match? |
|---|---|---|---|
| spectrum_overlay.png | `dd2d54ab...` | `56abc2a5...` | DIFF |
| molecule_grid.png | `9131dd48...` | `7d9db3ea...` | DIFF |
| attention_heatmap.png | `7a401521...` | `ebef8cb4...` | DIFF |

PNG files differ due to different matplotlib versions (3.10.6 vs 3.10.9) producing different rendering metadata. However:
- All 3 repeated runs using the same matplotlib version produce **identical** PNGs
- Image dimensions are the same
- Visual content is equivalent (spectra, molecules, heatmaps appear correctly)

**Verdict: PNGs differ due to rendering engine version; within-version reproducibility is perfect**
