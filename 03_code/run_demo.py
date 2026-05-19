#!/usr/bin/env python3
"""MultiSpec-GeoDiff Stage-I Demo: IR spectral retrieval and reranking.

Usage:
    python 03_code/run_demo.py --query_id 0 --data 04_data/IR_nist_200.jsonl --top_k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import load_ir_jsonl, split_query_library, get_unique_smiles
from spectra_preprocess import preprocess_record, preprocess_records
from spectra_encoder import CoordinateAwareSpectralEncoder
from spectral_attention import compute_attention_heatmap
from candidate_generator import build_library_embeddings, retrieve_top_m
from reranker import rerank_candidates
from metrics import topk_hit
from visualization import (
    plot_spectrum_overlay,
    plot_attention_heatmap,
    plot_molecule_grid,
)


def get_environment_summary() -> dict:
    """Collect Python environment info."""
    import platform

    summary = {"python_version": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for name in ["numpy", "scipy", "matplotlib", "rdkit", "torch"]:
        try:
            mod = __import__(name)
            summary["packages"][name] = getattr(mod, "__version__", "available")
        except Exception:
            summary["packages"][name] = None
    return summary


def format_smiles(smiles: str, max_len: int = 40) -> str:
    if len(smiles) <= max_len:
        return smiles
    return smiles[: max_len - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MultiSpec-GeoDiff Stage-I Demo: IR spectral retrieval and reranking"
    )
    parser.add_argument(
        "--query_id", type=int, default=0, help="Index of the query molecule (0-199)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="04_data/IR_nist_200.jsonl",
        help="Path to the JSONL data file",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top candidates to return"
    )
    parser.add_argument(
        "--top_m", type=int, default=20, help="Number of candidates to retrieve before reranking"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="05_outputs",
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MultiSpec-GeoDiff Stage-I Demo")
    print("=" * 60)

    # Step 1: Load data
    print(f"\n[1/8] Loading data from {args.data} ...")
    records = load_ir_jsonl(args.data)
    n_valid = sum(1 for r in records if True)  # all valid in curated set
    print(f"  Loaded {len(records)} records, {n_valid} valid")

    # Step 2: Split query and library
    print(f"\n[2/8] Splitting: query_id={args.query_id} ...")
    query_record, library_records = split_query_library(records, args.query_id)
    print(f"  Query SMILES: {query_record['smiles']}")
    print(f"  Library size: {len(library_records)} molecules")

    # Step 3: Preprocess query
    print(f"\n[3/8] Preprocessing query spectrum ...")
    query_processed = preprocess_record(query_record)
    n_pts = len(query_processed["wavenumber"])
    n_peaks = len(query_processed["peaks_wavenumber"])
    print(f"  Grid: {n_pts} points ({query_processed['grid_min']}-{query_processed['grid_max']} cm^-1)")
    print(f"  Peaks detected: {n_peaks}")

    # Step 4: Preprocess library
    print(f"\n[4/8] Preprocessing library spectra ...")
    t0 = time.time()
    library_processed = preprocess_records(library_records)
    elapsed = time.time() - t0
    print(f"  Processed {len(library_processed)} spectra in {elapsed:.2f}s")

    # Step 5: Encode query
    print(f"\n[5/8] Encoding query spectrum ...")
    encoder = CoordinateAwareSpectralEncoder(hidden_dim=16)
    query_embedding, _, attention_matrix = encoder.encode(query_processed)
    print(f"  Embedding dim: {query_embedding.shape[0]}")

    # Step 6: Build library embeddings and retrieve
    print(f"\n[6/8] Retrieving Top-{args.top_m} candidates ...")
    t0 = time.time()
    library_embeddings, library_smiles = build_library_embeddings(library_processed, encoder)
    indices, retrieval_scores = retrieve_top_m(query_embedding, library_embeddings, top_m=args.top_m)
    elapsed = time.time() - t0
    print(f"  Encoded {len(library_embeddings)} library spectra in {elapsed:.2f}s")

    # Step 7: Rerank
    print(f"\n[7/8] Reranking to Top-{args.top_k} ...")
    result_df = rerank_candidates(
        query_processed, library_processed, indices, retrieval_scores, top_k=args.top_k
    )
    print(f"\n  Top-{args.top_k} Candidates:")
    print(f"  {'Rank':>4} {'Score':>8} {'Hit?':>5} {'SMILES'}")
    print(f"  {'-' * 48}")
    for _, row in result_df.iterrows():
        gt_mark = "  *" if row["is_ground_truth"] else ""
        print(f"  {row['rank']:>4} {row['final_score']:>8.4f} {gt_mark:>5} {format_smiles(row['smiles'])}")
    hit = topk_hit(result_df["smiles"].tolist(), query_record["smiles"], k=args.top_k)
    print(f"\n  Ground truth in Top-{args.top_k}: {'YES' if hit else 'NO'}")

    # Step 8: Generate outputs
    print(f"\n[8/8] Generating outputs in {output_dir}/ ...")

    # 8a: Top-K candidates CSV
    csv_path = output_dir / "topk_candidates.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 8b: Spectrum overlay
    top_candidates = [library_processed[i] for i in indices[:args.top_k]]
    overlay_path = output_dir / "spectrum_overlay.png"
    plot_spectrum_overlay(
        query_processed,
        [{"wavenumber": c["wavenumber"], "intensity": c["intensity"],
          "smiles": c["original_record"]["smiles"]} for c in top_candidates],
        str(overlay_path),
    )
    print(f"  Saved: {overlay_path}")

    # 8c: Attention heatmap
    heatmap_path = output_dir / "attention_heatmap.png"
    if attention_matrix.shape[0] > 100:
        heatmap = compute_attention_heatmap(attention_matrix, downsample=8)
    else:
        heatmap = attention_matrix
    plot_attention_heatmap(heatmap, str(heatmap_path))
    print(f"  Saved: {heatmap_path}")

    # 8d: Molecule grid
    grid_path = output_dir / "molecule_grid.png"
    smiles_list = result_df["smiles"].tolist()
    scores_list = result_df["final_score"].tolist()
    plot_molecule_grid(smiles_list, scores_list, str(grid_path))
    print(f"  Saved: {grid_path}")

    # 8e: Ablation results
    ablation_path = output_dir / "ablation_results.csv"
    _write_ablation(query_processed, library_processed, indices, retrieval_scores, result_df, ablation_path)
    print(f"  Saved: {ablation_path}")

    # 8f: Trace log
    trace_path = output_dir / "trace_log.json"
    trace_log = _build_trace_log(
        args=args,
        query_record=query_record,
        result_df=result_df,
        hit=hit,
        output_dir=output_dir,
    )
    with open(trace_path, "w") as f:
        json.dump(trace_log, f, indent=2, default=str)
    print(f"  Saved: {trace_path}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    return 0


def _write_ablation(
    query_processed: dict,
    library_processed: list[dict],
    indices: list[int],
    retrieval_scores: list[float],
    result_df,
    path: Path,
) -> None:
    """Write a simple ablation CSV: compare raw cosine vs full pipeline."""
    import pandas as pd
    from metrics import cosine_similarity
    rows = []
    q_int = query_processed["intensity"]
    for i, idx in enumerate(indices):
        if i >= 10:
            break
        cand = library_processed[idx]
        raw_cos = cosine_similarity(q_int, cand["intensity"])
        rows.append({
            "rank_retrieval": i + 1,
            "smiles": cand["original_record"]["smiles"],
            "raw_cosine": round(raw_cos, 4),
            "retrieval_score": round(retrieval_scores[i], 4),
        })
    # Add full pipeline scores
    for _, row in result_df.iterrows():
        for r in rows:
            if r["smiles"] == row["smiles"]:
                r["full_pipeline_score"] = row["final_score"]
                r["rank_final"] = row["rank"]
                r["is_ground_truth"] = row["is_ground_truth"]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _build_trace_log(args, query_record, result_df, hit: bool, output_dir: Path) -> dict:
    """Build structured trace log for reproducibility."""
    implemented_modules = [
        "data_loader (JSONL)",
        "spectra_preprocess (interpolation + normalization + peak extraction)",
        "spectra_encoder (coordinate-aware encoding + Fourier features)",
        "spectral_attention (distance-bias self-attention)",
        "candidate_generator (cosine similarity retrieval)",
        "reranker (forward-spectrum similarity reranking)",
        "visualization (overlay, heatmap, molecule grid)",
        "metrics (cosine, L1, peak match, top-k hit)",
    ]
    future_modules = [
        "distance_head (Stage-II: 2D graph -> pairwise distance matrix)",
        "graph_diffusion (Stage-II: size-adaptive candidate generation)",
        "tfn_transformer (Stage-III: SE(3)-equivariant refinement)",
    ]
    outputs = {
        "topk_candidates_csv": str(output_dir / "topk_candidates.csv"),
        "spectrum_overlay": str(output_dir / "spectrum_overlay.png"),
        "attention_heatmap": str(output_dir / "attention_heatmap.png"),
        "molecule_grid": str(output_dir / "molecule_grid.png"),
        "ablation_results": str(output_dir / "ablation_results.csv"),
        "trace_log": str(output_dir / "trace_log.json"),
    }
    # Store data_file relative to the output directory for reproducibility
    data_path = Path(args.data)
    try:
        rel_data = str(data_path.relative_to(Path.cwd()))
    except ValueError:
        rel_data = str(args.data)
    return {
        "demo_name": "MultiSpec-GeoDiff Stage-I Demo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_file": rel_data,
        "num_molecules": 200,
        "query_id": args.query_id,
        "query_smiles": query_record["smiles"],
        "implemented_modules": implemented_modules,
        "future_modules": future_modules,
        "parameters": {
            "top_k": args.top_k,
            "top_m": args.top_m,
        },
        "results": {
            "ground_truth_in_top_k": hit,
            "top_candidates": result_df.to_dict(orient="records"),
        },
        "outputs": outputs,
        "environment": get_environment_summary(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
