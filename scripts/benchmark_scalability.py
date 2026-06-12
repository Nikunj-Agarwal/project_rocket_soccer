"""
benchmark_scalability.py — Sweep angular resolution and compare analytic search
vs StrikeNet inference time to visualise the amortisation argument.

Usage
-----
  python -m scripts.benchmark_scalability [--n-scenes 200] [--repeats 30]
                                          [--model-path models/strategy_net.pth]

Outputs
-------
  data/reports/benchmarks/scalability.csv
  data/reports/plots/global/scalability_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
)
from src.goal import Goal
from src.planner import analytic_strike_plan
from src.network import StrikeNet
from src.data_layout import (
    BENCHMARKS_DIR,
    SCALABILITY_CSV,
    ensure_dir,
    plots_global_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scenes(n: int, seed: int = 0) -> list[dict]:
    """Generate n random scenes from the same distribution as data_generator."""
    rng = np.random.RandomState(seed)
    scenes = []
    for _ in range(n):
        b_x = rng.uniform(2.0, 8.0)
        b_y = rng.uniform(0.0, 6.0)
        phi = rng.uniform(0.0, 2 * np.pi)
        v_b = rng.uniform(0.5, 2.0)
        scenes.append({
            "ball_pos": np.array([b_x, b_y]),
            "ball_vel": np.array([v_b * np.cos(phi), v_b * np.sin(phi)]),
            "car_state": np.array([
                rng.uniform(0.0, 4.0),
                rng.uniform(0.0, 6.0),
                rng.uniform(-np.pi, np.pi),
            ]),
        })
    return scenes


def _median_ms(times_s: list[float]) -> float:
    return float(np.median(times_s)) * 1e3


def _iqr_ms(times_s: list[float]) -> float:
    arr = np.array(times_s) * 1e3
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    n_scenes: int = 200,
    repeats: int = 30,
    warmup: int = 3,
    model_path: str | None = None,
    n_angles_sweep: list[int] | None = None,
) -> None:
    if n_angles_sweep is None:
        n_angles_sweep = [18, 36, 72, 144, 288]

    goal = Goal()
    scenes = _build_scenes(n_scenes)

    # --- Load (or initialise) StrikeNet ---
    if model_path is None:
        model_path = str(PROJECT_ROOT / "models" / "strategy_net.pth")

    model = StrikeNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model from {model_path}")
    else:
        print(f"[WARNING] Model not found at {model_path}. Using random weights. "
              "Inference latency is still valid (architecture-only).")
    model = model.to("cpu")
    model.eval()

    # Pre-build input tensors for all scenes
    scene_inputs = []
    for sc in scenes:
        bp, bv, cs = sc["ball_pos"], sc["ball_vel"], sc["car_state"]
        scene_inputs.append(np.array([bp[0], bp[1], bv[0], bv[1], cs[0], cs[1], cs[2]], dtype=np.float32))

    # --- Warm-up StrikeNet ---
    for _ in range(warmup):
        model.predict(scene_inputs[0])

    # --- Time StrikeNet (independent of n_angles) ---
    print(f"\nTiming StrikeNet inference ({repeats} reps × {n_scenes} scenes) ...")
    net_raw: list[float] = []
    for inp in scene_inputs:
        per_scene = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict(inp)
            per_scene.append(time.perf_counter() - t0)
        net_raw.append(float(np.median(per_scene)))
    network_ms_mean = float(np.mean(net_raw)) * 1e3
    network_ms_std  = float(np.std(net_raw))  * 1e3

    print(f"  Network: {network_ms_mean:.3f} ± {network_ms_std:.3f} ms  (median per scene)")

    # --- Sweep n_angles for analytic search ---
    rows = []
    for n_ang in n_angles_sweep:
        print(f"\nTiming analytic search  n_angles={n_ang} ({repeats} reps × {n_scenes} scenes) ...")

        # Warm-up for this resolution
        for _ in range(warmup):
            analytic_strike_plan(
                scenes[0]["ball_pos"].copy(), scenes[0]["ball_vel"].copy(),
                scenes[0]["car_state"], goal,
                field_w=DEFAULT_FIELD_W, field_h=DEFAULT_FIELD_H,
                ball_dt=DEFAULT_BALL_DT, ball_restitution=DEFAULT_BALL_RESTITUTION,
                n_angles=n_ang,
            )

        analytic_raw: list[float] = []
        for sc in scenes:
            per_scene = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                analytic_strike_plan(
                    sc["ball_pos"].copy(), sc["ball_vel"].copy(),
                    sc["car_state"], goal,
                    field_w=DEFAULT_FIELD_W, field_h=DEFAULT_FIELD_H,
                    ball_dt=DEFAULT_BALL_DT, ball_restitution=DEFAULT_BALL_RESTITUTION,
                    n_angles=n_ang,
                )
                per_scene.append(time.perf_counter() - t0)
            analytic_raw.append(float(np.median(per_scene)))

        analytic_ms_mean = float(np.mean(analytic_raw)) * 1e3
        analytic_ms_std  = float(np.std(analytic_raw))  * 1e3
        speedup = analytic_ms_mean / max(network_ms_mean, 1e-9)

        print(f"  Analytic ({n_ang:>3d} angles): {analytic_ms_mean:.1f} ± {analytic_ms_std:.1f} ms  "
              f"| speedup = {speedup:.1f}x")

        rows.append({
            "n_angles": n_ang,
            "analytic_ms_mean": analytic_ms_mean,
            "analytic_ms_std":  analytic_ms_std,
            "network_ms_mean":  network_ms_mean,
            "network_ms_std":   network_ms_std,
            "speedup":          speedup,
        })

    # --- Save CSV ---
    ensure_dir(BENCHMARKS_DIR)
    import csv
    with open(SCALABILITY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {SCALABILITY_CSV}")

    # --- Save plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_vals  = [r["n_angles"]         for r in rows]
    a_mean  = [r["analytic_ms_mean"] for r in rows]
    a_std   = [r["analytic_ms_std"]  for r in rows]
    n_mean  = [r["network_ms_mean"]  for r in rows]
    n_std   = [r["network_ms_std"]   for r in rows]
    speedup = [r["speedup"]          for r in rows]

    ax.errorbar(x_vals, a_mean, yerr=a_std, marker="o", label="Analytic search", linewidth=2)
    ax.errorbar(x_vals, n_mean, yerr=n_std, marker="s", linestyle="--", label="StrikeNet (CPU)", linewidth=2)

    # Annotate speedup at the default resolution (n_angles=36)
    default_row = next((r for r in rows if r["n_angles"] == 36), rows[0])
    ax.annotate(
        f"Default n_angles=36\n{default_row['speedup']:.0f}× faster",
        xy=(36, default_row["analytic_ms_mean"]),
        xytext=(50, default_row["analytic_ms_mean"] * 1.4),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Angular resolution n_angles", fontsize=11)
    ax.set_ylabel("Decision latency (ms, log scale)", fontsize=11)
    ax.set_title("Amortisation: Analytic Search vs. StrikeNet Inference", fontsize=12)
    ax.set_xticks(x_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()

    plot_path = plots_global_dir() / "scalability_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # --- Brief summary ---
    print("\n" + "=" * 60)
    print("SCALABILITY BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Scenes  : {n_scenes}  |  Repeats/scene: {repeats}")
    print(f"  Network : {network_ms_mean:.3f} ms  (flat across all n_angles)")
    for r in rows:
        print(f"  n={r['n_angles']:>3d}  analytic={r['analytic_ms_mean']:.1f} ms  "
              f"speedup={r['speedup']:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalability benchmark: analytic search vs StrikeNet")
    parser.add_argument("--n-scenes",   type=int,   default=200,  help="Number of random scenes")
    parser.add_argument("--repeats",    type=int,   default=30,   help="Timing repeats per scene")
    parser.add_argument("--model-path", type=str,   default=None, help="Path to strategy_net.pth")
    args = parser.parse_args()

    run_benchmark(
        n_scenes=args.n_scenes,
        repeats=args.repeats,
        model_path=args.model_path,
    )
