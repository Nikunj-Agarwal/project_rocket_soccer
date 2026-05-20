"""
generate_plots.py — Phase 4

Builds report figures under data/reports/plots/ with clear batch/seed structure.

Layout:
  plots/global/                          — training curve, StrikeNet sanity (not batch-specific)
  plots/integration/{batch_id}/          — batch summary + README
  plots/integration/{batch_id}/seed_{N}/ — trajectory.png, errors.png (from that batch's run)

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --batch 20260521_022824
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ball_physics import DEFAULT_FIELD_H, DEFAULT_FIELD_W
from src.data_layout import (
    BATCH_LOG,
    INTEGRATION_TESTS_DIR,
    RUN_METADATA,
    STRIKE_DATASET,
    TRAINING_LOG,
    TRAJECTORY_CSV,
    iter_integration_seed_runs,
    latest_integration_batch,
    list_integration_batches,
    plots_batch_dir,
    plots_global_dir,
    plots_seed_dir,
)

GOAL_POS = (9.5, 3.0)
FIELD_W = DEFAULT_FIELD_W
FIELD_H = DEFAULT_FIELD_H


def _draw_field(ax, title: str = "") -> None:
    ax.set_xlim(-0.3, FIELD_W + 0.3)
    ax.set_ylim(-0.3, FIELD_H + 0.3)
    ax.set_aspect("equal")
    ax.add_patch(
        plt.Rectangle((0, 0), FIELD_W, FIELD_H, fill=True, facecolor="#2e7d32", edgecolor="white", lw=2)
    )
    gx, gy = GOAL_POS
    ax.plot(gx, gy, marker="x", color="gold", markersize=12, markeredgewidth=2, zorder=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)


def plot_training_curves(training_log: Path, out_dir: Path) -> str:
    df = pd.read_csv(training_log)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train MSE", lw=1.5)
    ax.plot(df["epoch"], df["test_loss"], label="Test MSE", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("StrikeNet Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "training_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_trajectory(df: pd.DataFrame, seed: str, batch_id: str, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_field(ax, title=f"Batch {batch_id} — Seed {seed}")

    ax.plot(df["car_x"], df["car_y"], color="#1565c0", lw=2, label="Car path")
    ax.plot(df["ball_x"], df["ball_y"], color="red", lw=2, ls="--", label="Ball path")
    ax.scatter(df["car_x"].iloc[0], df["car_y"].iloc[0], c="#1565c0", s=60, zorder=6)
    ax.scatter(df["ball_x"].iloc[0], df["ball_y"].iloc[0], c="red", s=60, zorder=6)
    ax.scatter(df["car_x"].iloc[-1], df["car_y"].iloc[-1], c="#1565c0", marker="*", s=120, zorder=6)
    ax.scatter(df["ball_x"].iloc[-1], df["ball_y"].iloc[-1], c="red", marker="*", s=120, zorder=6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = out_dir / "trajectory.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_errors(df: pd.DataFrame, seed: str, batch_id: str, out_dir: Path) -> str:
    steps = df["step"].values
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(steps, df["pos_err"], color="#1565c0", lw=1.5)
    axes[0].axhline(0.2, color="gray", ls="--", lw=1, label="Threshold (0.2 m)")
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title(f"Batch {batch_id} — Tracking Errors — Seed {seed}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, df["heading_err"], color="#e65100", lw=1.5)
    axes[1].axhline(0.15, color="gray", ls="--", lw=1, label="Threshold (0.15 rad)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Heading error (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path = out_dir / "errors.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_integration_summary(
    seed_runs: list[tuple[str, Path]], batch_id: str, out_dir: Path
) -> str | None:
    if not seed_runs:
        return None

    seeds, final_pos, final_head, successes = [], [], [], []
    for seed, run_dir in seed_runs:
        csv_path = run_dir / TRAJECTORY_CSV
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        seeds.append(seed)
        final_pos.append(float(df["pos_err"].iloc[-1]))
        final_head.append(float(df["heading_err"].iloc[-1]))
        meta_path = run_dir / RUN_METADATA
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            successes.append(bool(meta.get("success", False)))
        else:
            successes.append(final_pos[-1] <= 0.2 and final_head[-1] <= 0.15)

    if not seeds:
        return None

    x = np.arange(len(seeds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, final_pos, width, label="Final pos err (m)", color="#1565c0")
    ax.bar(x + width / 2, final_head, width, label="Final heading err (rad)", color="#e65100")
    ax.axhline(0.2, color="#1565c0", ls="--", lw=1, alpha=0.6)
    ax.axhline(0.15, color="#e65100", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({'OK' if ok else 'X'})" for s, ok in zip(seeds, successes)], fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_title(f"Integration Batch {batch_id} — Final Errors per Seed")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "integration_summary.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_strikenet_samples(dataset_path: Path, model_path: Path, out_dir: Path, n_samples: int = 8) -> str | None:
    if not dataset_path.is_file() or not model_path.is_file():
        return None

    import torch
    from src.network import StrikeNet

    data = np.load(dataset_path)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data), size=min(n_samples, len(data)), replace=False)

    device = torch.device("cpu")
    model = StrikeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for ax, i in zip(axes, idx):
        row = data[i]
        inputs = row[:7]
        gt = row[7:11]
        pred = model.predict(inputs)
        ax.bar(
            ["T", "x", "y", "θ"],
            np.abs(pred - gt),
            color=["#5c6bc0", "#26a69a", "#26a69a", "#ff7043"],
        )
        ax.set_title(f"Dataset idx {i}", fontsize=9)
        ax.set_ylabel("|error|")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("StrikeNet vs Ground Truth (random dataset samples)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "strikenet_sample_errors.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def write_batch_readme(
    batch_id: str,
    batch_dir: Path,
    plot_batch_dir: Path,
    seed_runs: list[tuple[str, Path]],
) -> Path:
    lines = [
        f"# Integration batch `{batch_id}` — report plots",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Source data (raw runs)",
        "",
        f"Each seed's simulation artifacts live under:",
        "",
        f"`data/tests/integration/{batch_id}/seed_<N>/`",
        "",
        "| Seed | Source run | Video | Trajectory | Metadata |",
        "|------|------------|-------|------------|----------|",
    ]
    for seed, run_dir in seed_runs:
        rel = run_dir.relative_to(PROJECT_ROOT).as_posix()
        vid = "simulation.mp4" if (run_dir / "simulation.mp4").is_file() else "simulation.gif"
        lines.append(
            f"| {seed} | `{rel}/` | `{vid}` | `trajectory.csv` | `metadata.json` |"
        )

    lines.extend(
        [
            "",
            "## Plots in this folder",
            "",
            "- `integration_summary.png` — final errors for all seeds in this batch",
            "- `seed_<N>/trajectory.png` — car/ball paths on the field",
            "- `seed_<N>/errors.png` — position & heading error vs step",
            "",
            f"Batch log: `data/tests/integration/{batch_id}/batch.log`",
            "",
        ]
    )
    readme = plot_batch_dir / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def resolve_batch(batch_id: str | None) -> tuple[str | None, Path | None, list[tuple[str, Path]]]:
    if batch_id:
        batch_dir = INTEGRATION_TESTS_DIR / batch_id
        if batch_dir.is_dir():
            return batch_id, batch_dir, list(iter_integration_seed_runs(batch_dir))

    latest = latest_integration_batch()
    if latest is not None:
        return latest.name, latest, list(iter_integration_seed_runs(latest))
    return None, None, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 4 report plots (batch-organized)")
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Integration batch id (default: latest under data/tests/integration/)",
    )
    parser.add_argument("--training-log", type=str, default=str(TRAINING_LOG))
    parser.add_argument("--dataset", type=str, default=str(STRIKE_DATASET))
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "strategy_net.pth"),
    )
    args = parser.parse_args()

    saved: list[str] = []
    global_dir = plots_global_dir()

    training_log = Path(args.training_log)
    if training_log.is_file():
        saved.append(plot_training_curves(training_log, global_dir))

    sample_plot = plot_strikenet_samples(Path(args.dataset), Path(args.model), global_dir)
    if sample_plot:
        saved.append(sample_plot)

    batch_id, batch_dir, seed_runs = resolve_batch(args.batch)
    if batch_id and batch_dir is not None:
        plot_batch_root = plots_batch_dir(batch_id)
        summary = plot_integration_summary(seed_runs, batch_id, plot_batch_root)
        if summary:
            saved.append(summary)

        for seed, run_dir in seed_runs:
            csv_path = run_dir / TRAJECTORY_CSV
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            seed_plot_dir = plots_seed_dir(plot_batch_root, seed)
            saved.append(plot_trajectory(df, seed, batch_id, seed_plot_dir))
            saved.append(plot_errors(df, seed, batch_id, seed_plot_dir))

        saved.append(str(write_batch_readme(batch_id, batch_dir, plot_batch_root, seed_runs)))

    # Top-level plots README
    plots_root = global_dir.parent
    batches = list_integration_batches()
    index_lines = [
        "# Report plots index",
        "",
        "> **Note:** `training_curve.png` and `strikenet_sample_errors.png` are **not**",
        "> in this folder root. They live under **`global/`** (batch-independent).",
        "",
        "## `global/`",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `global/training_curve.png` | StrikeNet train/test MSE from `data/training/training_log.csv` |",
        "| `global/strikenet_sample_errors.png` | Model vs dataset labels (8 random samples) |",
        "",
        "## `integration/<batch_id>/`",
        "",
        "One folder per `python scripts/test_main.py` run (batch id = timestamp).",
        "Per seed: `seed_<N>/trajectory.png`, `seed_<N>/errors.png`.",
        "",
    ]
    if batches:
        index_lines.append("Available batches (newest first):")
        for b in batches:
            index_lines.append(f"- `{b.name}` → `integration/{b.name}/`")
    if batch_id:
        index_lines.append(f"\n**Latest plotted batch:** `{batch_id}`")
    (plots_root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print("=" * 60)
    print("  PHASE 4 — Plot Generation")
    print("=" * 60)
    print(f"  Global plots     : {global_dir}")
    if batch_id:
        print(f"  Batch plots      : {plots_batch_dir(batch_id)}")
        print(f"  Seeds plotted    : {len(seed_runs)}")
    print(f"  Figures saved    : {len(saved)}")
    for p in saved:
        print(f"    - {Path(p).relative_to(PROJECT_ROOT).as_posix()}")
    print("=" * 60)
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
