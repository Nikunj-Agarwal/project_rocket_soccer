"""
data_layout.py — Canonical paths under data/ for datasets, tests, runs, and reports.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# --- Top-level buckets ---
DATASET_DIR = DATA_ROOT / "dataset"
TRAINING_DIR = DATA_ROOT / "training"
REPORTS_DIR = DATA_ROOT / "reports"
TESTS_DIR = DATA_ROOT / "tests"
RUNS_DIR = DATA_ROOT / "runs"

INTEGRATION_TESTS_DIR = TESTS_DIR / "integration"
STATIC_TESTS_DIR = TESTS_DIR / "static"
MANUAL_RUNS_DIR = RUNS_DIR / "manual"

# --- Named artifacts ---
STRIKE_DATASET = DATASET_DIR / "strike_dataset.npy"
DATASET_STATS = DATASET_DIR / "dataset_stats.json"
TRAINING_LOG = TRAINING_DIR / "training_log.csv"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_GLOBAL_DIR = PLOTS_DIR / "global"
PLOTS_INTEGRATION_DIR = PLOTS_DIR / "integration"
BENCHMARKS_DIR = REPORTS_DIR / "benchmarks"
SCALABILITY_CSV = BENCHMARKS_DIR / "scalability.csv"

TRAJECTORY_CSV = "trajectory.csv"
SIMULATION_MP4 = "simulation.mp4"
SIMULATION_GIF = "simulation.gif"
RUN_METADATA = "metadata.json"
BATCH_LOG = "batch.log"


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_integration_batch(batch_id: str | None = None) -> Path:
    """data/tests/integration/{batch_id}/"""
    bid = batch_id or timestamp_id()
    return ensure_dir(INTEGRATION_TESTS_DIR / bid)


def integration_seed_run(batch_dir: Path, seed: int) -> Path:
    """data/tests/integration/{batch}/seed_{seed}/"""
    return ensure_dir(batch_dir / f"seed_{seed}")


def new_static_test_batch(batch_id: str | None = None) -> Path:
    """data/tests/static/{batch_id}/"""
    bid = batch_id or timestamp_id()
    return ensure_dir(STATIC_TESTS_DIR / bid)


def new_manual_run(seed: int | None = None, run_id: str | None = None) -> Path:
    """data/runs/manual/manual_seed{N}_{timestamp}/ or manual_{timestamp}/"""
    ts = timestamp_id()
    if seed is not None:
        name = run_id or f"manual_seed{seed}_{ts}"
    else:
        name = run_id or f"manual_{ts}"
    return ensure_dir(MANUAL_RUNS_DIR / name)


def list_integration_batches() -> list[Path]:
    if not INTEGRATION_TESTS_DIR.is_dir():
        return []
    batches = [p for p in INTEGRATION_TESTS_DIR.iterdir() if p.is_dir()]
    return sorted(batches, key=lambda p: p.name, reverse=True)


def latest_integration_batch() -> Path | None:
    batches = list_integration_batches()
    return batches[0] if batches else None


def plots_global_dir() -> Path:
    """Report figures not tied to one integration batch."""
    return ensure_dir(PLOTS_GLOBAL_DIR)


def plots_batch_dir(batch_id: str) -> Path:
    """Report figures for one integration batch: plots/integration/{batch_id}/"""
    return ensure_dir(PLOTS_INTEGRATION_DIR / batch_id)


def plots_seed_dir(batch_plot_dir: Path, seed: int | str) -> Path:
    """Per-seed plot folder: .../seed_{N}/"""
    return ensure_dir(batch_plot_dir / f"seed_{seed}")


def iter_integration_seed_runs(batch_dir: Path):
    """Yield (seed_str, run_dir) for each seed_* folder with trajectory.csv."""
    if not batch_dir.is_dir():
        return
    for run_dir in sorted(batch_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("seed_"):
            continue
        if (run_dir / TRAJECTORY_CSV).is_file():
            yield run_dir.name.replace("seed_", ""), run_dir
