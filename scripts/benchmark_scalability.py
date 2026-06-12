"""
benchmark_scalability.py — Sweep angular resolution and compare analytic search
vs StrikeNet inference time to visualise the amortisation argument.

Usage
-----
  python -m scripts.benchmark_scalability [--n-scenes 200] [--repeats 30]
                                          [--model-path models/strategy_net.pth]
                                          [--workers 0]

  --workers 0 (default) uses ~75% of CPU cores, matching data_generator.py.
  Analytic timing is parallelised across scenes; StrikeNet timing stays serial
  (fast enough, and avoids per-worker model copies).

Outputs
-------
  data/reports/benchmarks/scalability.csv
  data/reports/plots/global/scalability_curve.png
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
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

# Worker-local goal (same pattern as data_generator._goal)
_worker_goal = Goal()


def _default_num_workers() -> int:
    total = multiprocessing.cpu_count()
    return max(1, int(total * 0.75))


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
            "ball_pos": np.array([b_x, b_y], dtype=float),
            "ball_vel": np.array([v_b * np.cos(phi), v_b * np.sin(phi)], dtype=float),
            "car_state": np.array([
                rng.uniform(0.0, 4.0),
                rng.uniform(0.0, 6.0),
                rng.uniform(-np.pi, np.pi),
            ], dtype=float),
        })
    return scenes


def _time_analytic_scene_worker(args: tuple) -> float:
    """Worker: median perf_counter (seconds) over repeats for one scene.

    Same measurement as the serial loop: warm-up is done once per n_angles in the
    parent process before the pool starts; each worker times `repeats` calls
    and returns the median.
    """
    ball_pos, ball_vel, car_state, n_angles, repeats = args
    bp = np.array(ball_pos, dtype=float)
    bv = np.array(ball_vel, dtype=float)
    cs = np.array(car_state, dtype=float)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        analytic_strike_plan(
            bp.copy(), bv.copy(), cs, _worker_goal,
            field_w=DEFAULT_FIELD_W,
            field_h=DEFAULT_FIELD_H,
            ball_dt=DEFAULT_BALL_DT,
            ball_restitution=DEFAULT_BALL_RESTITUTION,
            n_angles=n_angles,
        )
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _time_analytic_sweep(
    scenes: list[dict],
    n_angles: int,
    repeats: int,
    warmup: int,
    num_workers: int,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) across scene medians for one n_angles level."""
    # Serial warm-up (discarded) — same as before parallel dispatch
    sc0 = scenes[0]
    for _ in range(warmup):
        analytic_strike_plan(
            sc0["ball_pos"].copy(), sc0["ball_vel"].copy(),
            sc0["car_state"], Goal(),
            field_w=DEFAULT_FIELD_W, field_h=DEFAULT_FIELD_H,
            ball_dt=DEFAULT_BALL_DT, ball_restitution=DEFAULT_BALL_RESTITUTION,
            n_angles=n_angles,
        )

    tasks = [
        (sc["ball_pos"], sc["ball_vel"], sc["car_state"], n_angles, repeats)
        for sc in scenes
    ]

    if num_workers <= 1 or len(tasks) == 1:
        medians = [_time_analytic_scene_worker(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            medians = list(pool.map(_time_analytic_scene_worker, tasks))

    medians_arr = np.array(medians, dtype=float)
    return float(np.mean(medians_arr) * 1e3), float(np.std(medians_arr) * 1e3)


def _time_network_inference(
    model: StrikeNet,
    scene_inputs: list[np.ndarray],
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) across scene medians for StrikeNet predict."""
    for _ in range(warmup):
        model.predict(scene_inputs[0])

    net_raw: list[float] = []
    for inp in scene_inputs:
        per_scene: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict(inp)
            per_scene.append(time.perf_counter() - t0)
        net_raw.append(float(np.median(per_scene)))

    return float(np.mean(net_raw) * 1e3), float(np.std(net_raw) * 1e3)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    n_scenes: int = 200,
    repeats: int = 30,
    warmup: int = 3,
    model_path: str | None = None,
    n_angles_sweep: list[int] | None = None,
    num_workers: int = 0,
) -> None:
    if n_angles_sweep is None:
        n_angles_sweep = [18, 36, 72, 144, 288]

    if num_workers <= 0:
        num_workers = _default_num_workers()

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

    scene_inputs = []
    for sc in scenes:
        bp, bv, cs = sc["ball_pos"], sc["ball_vel"], sc["car_state"]
        scene_inputs.append(
            np.array([bp[0], bp[1], bv[0], bv[1], cs[0], cs[1], cs[2]], dtype=np.float32)
        )

    total_cores = multiprocessing.cpu_count()
    print(f"  Parallel workers: {num_workers}/{total_cores} (analytic sweep only)")

    # --- Time StrikeNet (serial — sub-second total, avoids model duplication) ---
    print(f"\nTiming StrikeNet inference ({repeats} reps × {n_scenes} scenes) ...")
    network_ms_mean, network_ms_std = _time_network_inference(
        model, scene_inputs, repeats, warmup,
    )
    print(f"  Network: {network_ms_mean:.3f} ± {network_ms_std:.3f} ms  (median per scene)")

    # --- Sweep n_angles for analytic search (parallel over scenes) ---
    rows = []
    for n_ang in n_angles_sweep:
        print(f"\nTiming analytic search  n_angles={n_ang} "
              f"({repeats} reps × {n_scenes} scenes, {num_workers} workers) ...")
        t_sweep = time.perf_counter()

        analytic_ms_mean, analytic_ms_std = _time_analytic_sweep(
            scenes, n_ang, repeats, warmup, num_workers,
        )
        sweep_s = time.perf_counter() - t_sweep
        speedup = analytic_ms_mean / max(network_ms_mean, 1e-9)

        print(f"  Analytic ({n_ang:>3d} angles): {analytic_ms_mean:.1f} ± {analytic_ms_std:.1f} ms  "
              f"| speedup = {speedup:.1f}x  | wall {sweep_s:.0f}s")

        rows.append({
            "n_angles": n_ang,
            "analytic_ms_mean": analytic_ms_mean,
            "analytic_ms_std": analytic_ms_std,
            "network_ms_mean": network_ms_mean,
            "network_ms_std": network_ms_std,
            "speedup": speedup,
        })

    # --- Save CSV ---
    ensure_dir(BENCHMARKS_DIR)
    with open(SCALABILITY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {SCALABILITY_CSV}")

    # --- Save plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_vals = [r["n_angles"] for r in rows]
    a_mean = [r["analytic_ms_mean"] for r in rows]
    a_std = [r["analytic_ms_std"] for r in rows]
    n_mean = [r["network_ms_mean"] for r in rows]
    n_std = [r["network_ms_std"] for r in rows]

    ax.errorbar(x_vals, a_mean, yerr=a_std, marker="o", label="Analytic search", linewidth=2)
    ax.errorbar(x_vals, n_mean, yerr=n_std, marker="s", linestyle="--", label="StrikeNet (CPU)", linewidth=2)

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

    print("\n" + "=" * 60)
    print("SCALABILITY BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Scenes  : {n_scenes}  |  Repeats/scene: {repeats}  |  Workers: {num_workers}")
    print(f"  Network : {network_ms_mean:.3f} ms  (flat across all n_angles)")
    for r in rows:
        print(f"  n={r['n_angles']:>3d}  analytic={r['analytic_ms_mean']:.1f} ms  "
              f"speedup={r['speedup']:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Scalability benchmark: analytic search vs StrikeNet")
    parser.add_argument("--n-scenes", type=int, default=200, help="Number of random scenes")
    parser.add_argument("--repeats", type=int, default=30, help="Timing repeats per scene")
    parser.add_argument("--model-path", type=str, default=None, help="Path to strategy_net.pth")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for analytic sweep (0 = ~75%% of CPU cores, like data_generator)",
    )
    args = parser.parse_args()

    run_benchmark(
        n_scenes=args.n_scenes,
        repeats=args.repeats,
        model_path=args.model_path,
        num_workers=args.workers,
    )
