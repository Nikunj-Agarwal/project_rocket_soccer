"""
benchmark_scalability.py — Sweep angular resolution and compare analytic search
vs StrikeNet inference time to visualise the amortisation argument.

Usage
-----
  python -m scripts.benchmark_scalability [--n-scenes 200] [--repeats 30]
                                          [--model-variant legacy|structured|both]
                                          [--model-path models/strategy_net_legacy.pth]
                                          [--workers 0]

  For the structured variant, decision latency = StrikeNet inference + one ball
  rollout (the strike position is derived from physics, not predicted directly).

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
    propagate_ball_for_time,
    compute_strike_velocity,
)
from src.goal import Goal
from src.planner import analytic_strike_plan
from src.network import StrikeNet
from src.data_layout import (
    BENCHMARKS_DIR,
    SCALABILITY_CSV,
    ensure_dir,
    plots_global_dir,
    model_path_for_variant
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


def _time_ball_rollout(
    scenes: list[dict],
    repeats: int,
    warmup: int,
    rollout_T: float = 2.0,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for one ball-physics rollout per scene.

    The structured StrikeNet predicts only (T, theta); the strike position is
    derived by propagating the ball to the predicted horizon. That rollout is
    part of the structured decision latency, so it must be timed and added to
    the raw inference time for an honest comparison.
    """
    for _ in range(warmup):
        sc = scenes[0]
        propagate_ball_for_time(
            sc["ball_pos"], sc["ball_vel"], rollout_T, dt=DEFAULT_BALL_DT,
            field_w=DEFAULT_FIELD_W, field_h=DEFAULT_FIELD_H,
            restitution=DEFAULT_BALL_RESTITUTION,
        )

    roll_raw: list[float] = []
    for sc in scenes:
        per_scene: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            propagate_ball_for_time(
                sc["ball_pos"], sc["ball_vel"], rollout_T, dt=DEFAULT_BALL_DT,
                field_w=DEFAULT_FIELD_W, field_h=DEFAULT_FIELD_H,
                restitution=DEFAULT_BALL_RESTITUTION,
            )
            per_scene.append(time.perf_counter() - t0)
        roll_raw.append(float(np.median(per_scene)))

    return float(np.mean(roll_raw) * 1e3), float(np.std(roll_raw) * 1e3)


def _hybrid_fallback_sweep_once(sp, sv, goal, dt, field_w, field_h, restitution, v_impact=1.0):
    """Run the 36-heading scoring sweep used when hybrid rejects the network plan."""

    def scores(pos_xy, theta, svel):
        v_post = compute_strike_velocity(svel, v_car=v_impact, theta_car=theta, e_strike=0.8)
        fp, _ = propagate_ball_for_time(
            pos_xy, v_post, total_time=5.0, dt=dt,
            field_w=field_w, field_h=field_h,
            restitution=restitution, goal=goal,
        )
        return fp[0] >= goal.x - 1e-9 and goal.y_min <= fp[1] <= goal.y_max

    tg = np.arctan2(goal.center[1] - sp[1], goal.center[0] - sp[0])
    thetas = [t for t in np.linspace(-np.pi, np.pi, 36, endpoint=False) if scores(sp, t, sv)]
    if thetas:
        return min(thetas, key=lambda th: abs(np.arctan2(np.sin(th - tg), np.cos(th - tg))))
    return tg


def _time_hybrid_fallback_sweep(
    scenes: list[dict],
    repeats: int,
    warmup: int,
    rollout_T: float = 2.0,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for the hybrid fallback 36-heading sweep per scene."""
    goal = Goal()
    dt = DEFAULT_BALL_DT
    field_w, field_h = DEFAULT_FIELD_W, DEFAULT_FIELD_H
    rest = DEFAULT_BALL_RESTITUTION

    for _ in range(warmup):
        sc = scenes[0]
        sp, sv = propagate_ball_for_time(
            sc["ball_pos"], sc["ball_vel"], rollout_T, dt=dt,
            field_w=field_w, field_h=field_h, restitution=rest,
        )
        _hybrid_fallback_sweep_once(sp, sv, goal, dt, field_w, field_h, rest)

    sweep_raw: list[float] = []
    for sc in scenes:
        sp, sv = propagate_ball_for_time(
            sc["ball_pos"], sc["ball_vel"], rollout_T, dt=dt,
            field_w=field_w, field_h=field_h, restitution=rest,
        )
        per_scene: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _hybrid_fallback_sweep_once(sp, sv, goal, dt, field_w, field_h, rest)
            per_scene.append(time.perf_counter() - t0)
        sweep_raw.append(float(np.median(per_scene)))

    return float(np.mean(sweep_raw) * 1e3), float(np.std(sweep_raw) * 1e3)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _time_network_decision(
    variant: str,
    scene_inputs: list[np.ndarray],
    scenes: list[dict],
    repeats: int,
    warmup: int,
    model_path: str | None = None,
) -> tuple[float, float]:
    """Total per-decision latency for a variant (mean_ms, std_ms).

    legacy     -> StrikeNet inference only (predicts T, x, y, theta directly).
    structured -> StrikeNet inference + one ball rollout (position is derived by
                  propagating the ball to the predicted horizon T).
    """
    if model_path is None:
        model_path = str(model_path_for_variant(variant))

    if os.path.exists(model_path):
        model = StrikeNet.load(model_path)
        print(f"  Loaded {variant} model from {model_path}")
    else:
        model = StrikeNet(variant=variant)
        print(f"  [WARNING] {variant} model not found at {model_path}. Using random "
              "weights (inference latency is still architecture-valid).")
    model = model.to("cpu")
    model.eval()

    infer_mean, infer_std = _time_network_inference(model, scene_inputs, repeats, warmup)

    if variant == "structured":
        roll_mean, roll_std = _time_ball_rollout(scenes, repeats, warmup)
        total_mean = infer_mean + roll_mean
        # Independent timings: combine spreads in quadrature.
        total_std = float(np.hypot(infer_std, roll_std))
        print(f"  {variant}: infer {infer_mean:.3f} + rollout {roll_mean:.3f} "
              f"= {total_mean:.3f} ± {total_std:.3f} ms")
        return total_mean, total_std

    print(f"  {variant}: infer {infer_mean:.3f} ± {infer_std:.3f} ms")
    return infer_mean, infer_std


def run_benchmark(
    n_scenes: int = 200,
    repeats: int = 30,
    warmup: int = 3,
    model_path: str | None = None,
    variants: list[str] | None = None,
    n_angles_sweep: list[int] | None = None,
    num_workers: int = 0,
) -> None:
    if variants is None:
        variants = ["legacy"]
    if n_angles_sweep is None:
        n_angles_sweep = [18, 36, 72, 144, 288]

    if num_workers <= 0:
        num_workers = _default_num_workers()

    scenes = _build_scenes(n_scenes)

    scene_inputs = []
    for sc in scenes:
        bp, bv, cs = sc["ball_pos"], sc["ball_vel"], sc["car_state"]
        scene_inputs.append(
            np.array([bp[0], bp[1], bv[0], bv[1], cs[0], cs[1], cs[2]], dtype=np.float32)
        )

    total_cores = multiprocessing.cpu_count()
    print(f"  Parallel workers: {num_workers}/{total_cores} (analytic sweep only)")

    # --- Time StrikeNet decision latency per variant (serial; sub-second total) ---
    print(f"\nTiming StrikeNet decision latency ({repeats} reps × {n_scenes} scenes) ...")
    # model_path override only applies when a single variant is requested.
    mp_override = model_path if len(variants) == 1 else None
    variant_latency = {}
    for v in variants:
        variant_latency[v] = _time_network_decision(
            v, scene_inputs, scenes, repeats, warmup, model_path=mp_override,
        )

    # --- Sweep n_angles for analytic search once (variant-independent) ---
    analytic_sweep = {}
    for n_ang in n_angles_sweep:
        print(f"\nTiming analytic search  n_angles={n_ang} "
              f"({repeats} reps × {n_scenes} scenes, {num_workers} workers) ...")
        t_sweep = time.perf_counter()
        analytic_ms_mean, analytic_ms_std = _time_analytic_sweep(
            scenes, n_ang, repeats, warmup, num_workers,
        )
        sweep_s = time.perf_counter() - t_sweep
        analytic_sweep[n_ang] = (analytic_ms_mean, analytic_ms_std)
        print(f"  Analytic ({n_ang:>3d} angles): {analytic_ms_mean:.1f} ± {analytic_ms_std:.1f} ms "
              f"| wall {sweep_s:.0f}s")

    # --- Hybrid fallback sweep (36-heading scoring check; variant-independent) ---
    print(f"\nTiming hybrid fallback sweep ({repeats} reps × {n_scenes} scenes) ...")
    fb_sweep_mean, fb_sweep_std = _time_hybrid_fallback_sweep(scenes, repeats, warmup)
    print(f"  Hybrid fallback sweep: {fb_sweep_mean:.1f} ± {fb_sweep_std:.1f} ms  "
          f"(add to network latency when fallback fires)")

    # --- Build rows: one (variant × n_angles) record each ---
    rows = []
    for v in variants:
        net_mean, net_std = variant_latency[v]
        for n_ang in n_angles_sweep:
            a_mean, a_std = analytic_sweep[n_ang]
            rows.append({
                "variant": v,
                "n_angles": n_ang,
                "analytic_ms_mean": a_mean,
                "analytic_ms_std": a_std,
                "network_ms_mean": net_mean,
                "network_ms_std": net_std,
                "fallback_sweep_ms_mean": fb_sweep_mean if n_ang == 36 else "",
                "fallback_sweep_ms_std": fb_sweep_std if n_ang == 36 else "",
                "deployed_hybrid_worst_ms": (net_mean + fb_sweep_mean) if n_ang == 36 else "",
                "speedup": a_mean / max(net_mean, 1e-9),
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

    # Analytic curve is shared across variants.
    x_vals = list(n_angles_sweep)
    a_mean = [analytic_sweep[n][0] for n in x_vals]
    a_std = [analytic_sweep[n][1] for n in x_vals]
    ax.errorbar(x_vals, a_mean, yerr=a_std, marker="o", label="Analytic search", linewidth=2)

    markers = {"legacy": "s", "structured": "^"}
    for v in variants:
        net_mean, net_std = variant_latency[v]
        label = (f"StrikeNet {v} (CPU)" if v == "legacy"
                 else f"StrikeNet {v} = infer+rollout (CPU)")
        ax.errorbar(x_vals, [net_mean] * len(x_vals), yerr=[net_std] * len(x_vals),
                    marker=markers.get(v, "x"), linestyle="--", label=label, linewidth=2)

    # Annotate speedup at the default n_angles=36 for the first variant.
    ref_v = variants[0]
    ref_net = variant_latency[ref_v][0]
    a36 = analytic_sweep.get(36, analytic_sweep[x_vals[0]])[0]
    ax.annotate(
        f"Default n_angles=36\n{a36 / max(ref_net, 1e-9):.0f}× faster ({ref_v})",
        xy=(36 if 36 in x_vals else x_vals[0], a36),
        xytext=(50 if 36 in x_vals else x_vals[0], a36 * 1.4),
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
    ax.legend(fontsize=9)
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
    for v in variants:
        net_mean = variant_latency[v][0]
        print(f"  Network ({v:<10s}): {net_mean:.3f} ms  (flat across all n_angles)")
    print(f"  Hybrid fallback sweep : {fb_sweep_mean:.1f} ms  (when network plan rejected)")
    for v in variants:
        net_mean = variant_latency[v][0]
        print(f"  Hybrid worst-case ({v}): {net_mean + fb_sweep_mean:.1f} ms  (infer + fallback sweep)")
    for n_ang in n_angles_sweep:
        print(f"  n={n_ang:>3d}  analytic={analytic_sweep[n_ang][0]:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Scalability benchmark: analytic search vs StrikeNet")
    parser.add_argument("--n-scenes", type=int, default=200, help="Number of random scenes")
    parser.add_argument("--repeats", type=int, default=30, help="Timing repeats per scene")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Explicit model path (only honoured with a single --model-variant)")
    parser.add_argument("--model-variant", type=str,
                        choices=["legacy", "structured", "both"], default="legacy",
                        help="Which StrikeNet variant(s) to benchmark. 'both' overlays both.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for analytic sweep (0 = ~75%% of CPU cores, like data_generator)",
    )
    args = parser.parse_args()

    variants = ["legacy", "structured"] if args.model_variant == "both" else [args.model_variant]

    run_benchmark(
        n_scenes=args.n_scenes,
        repeats=args.repeats,
        model_path=args.model_path,
        variants=variants,
        num_workers=args.workers,
    )
