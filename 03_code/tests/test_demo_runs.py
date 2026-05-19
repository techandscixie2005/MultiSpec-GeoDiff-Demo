"""End-to-end tests for the demo pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = DEMO_DIR / "src"
DEMO_SCRIPT = DEMO_DIR / "run_demo.py"
DATA_FILE = DEMO_DIR.parent / "04_data" / "IR_nist_200.jsonl"
OUTPUT_DIR = DEMO_DIR.parent / "05_outputs"


@pytest.fixture(scope="module")
def demo_env():
    env = dict(PYTHONPATH=str(SRC_DIR))
    return env


def test_demo_creates_csv(demo_env):
    """Run the demo and check that topk_candidates.csv is created."""
    cmd = [
        sys.executable, str(DEMO_SCRIPT),
        "--query_id", "0",
        "--data", str(DATA_FILE),
        "--top_k", "5",
    ]
    result = subprocess.run(cmd, env={**dict(PYTHONPATH=str(SRC_DIR))}, capture_output=True, text=True, timeout=120)
    csv_path = OUTPUT_DIR / "topk_candidates.csv"
    assert csv_path.exists(), f"CSV not found at {csv_path}"
    assert csv_path.stat().st_size > 0


def test_demo_creates_trace_log(demo_env):
    """Check that trace_log.json exists and has required fields."""
    trace_path = OUTPUT_DIR / "trace_log.json"
    assert trace_path.exists()
    with open(trace_path) as f:
        log = json.load(f)
    assert log["demo_name"] == "MultiSpec-GeoDiff Stage-I Demo"
    assert log["num_molecules"] == 200
    assert "implemented_modules" in log
    assert "future_modules" in log
    assert "outputs" in log


def test_demo_query_id_1(demo_env):
    """Run demo with a different query_id."""
    cmd = [
        sys.executable, str(DEMO_SCRIPT),
        "--query_id", "5",
        "--data", str(DATA_FILE),
        "--top_k", "3",
    ]
    result = subprocess.run(cmd, env={**dict(PYTHONPATH=str(SRC_DIR))}, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Demo failed: {result.stderr}"


def test_demo_creates_all_outputs(demo_env):
    """Verify all expected output files exist."""
    expected = [
        "topk_candidates.csv",
        "spectrum_overlay.png",
        "attention_heatmap.png",
        "molecule_grid.png",
        "ablation_results.csv",
        "trace_log.json",
    ]
    for fname in expected:
        assert (OUTPUT_DIR / fname).exists(), f"Missing output: {fname}"
