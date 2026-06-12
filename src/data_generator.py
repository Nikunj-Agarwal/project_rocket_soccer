"""
data_generator.py — Phase 2 / 3.6

Generates geometric ground-truth training examples using brute-force reachability checks.
Ball future positions use the same inelastic wall-bounce model as the simulator.

Per-sample search time is now measured and a dataset_stats.json artifact is
written alongside the dataset so the offline generation cost is observable.
"""

import os
import json
import time
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm

from src.goal import Goal
from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
)
from src.planner import analytic_strike_plan

# Global goal object for workers
_goal = Goal()

def _generate_single_sample(args):
    """Worker function to generate exactly one valid sample.

    Returns
    -------
    (sample, attempts, elapsed_s)
      sample    : list of 11 floats [inputs..., outputs...]
      attempts  : number of random scenes tried before accepting one
      elapsed_s : wall-clock seconds spent searching (perf_counter)
    """
    field_w, field_h, ball_dt, ball_restitution, seed = args

    np.random.seed(seed)

    attempts = 0
    t_start = time.perf_counter()

    while True:
        attempts += 1

        # Randomize ball state
        b_x = np.random.uniform(2.0, 8.0)
        b_y = np.random.uniform(0.0, 6.0)
        phi = np.random.uniform(0.0, 2 * np.pi)
        v_b = np.random.uniform(0.5, 2.0)
        b_vx = v_b * np.cos(phi)
        b_vy = v_b * np.sin(phi)

        # Randomize car state
        c_x = np.random.uniform(0.0, 4.0)
        c_y = np.random.uniform(0.0, 6.0)
        c_theta = np.random.uniform(-np.pi, np.pi)

        ball_pos = np.array([b_x, b_y], dtype=float)
        ball_vel = np.array([b_vx, b_vy], dtype=float)
        car_state = np.array([c_x, c_y, c_theta], dtype=float)

        result = analytic_strike_plan(
            ball_pos, ball_vel, car_state, _goal,
            field_w=field_w, field_h=field_h,
            ball_dt=ball_dt, ball_restitution=ball_restitution,
        )

        if result is not None:
            T_feasible, x_strike, y_strike, theta_strike = result
            elapsed_s = time.perf_counter() - t_start
            sample = [
                b_x, b_y, b_vx, b_vy, c_x, c_y, c_theta,   # Inputs
                T_feasible, x_strike, y_strike, theta_strike, # Outputs
            ]
            return sample, attempts, elapsed_s


def generate_data(
    num_samples: int,
    output_path: str,
    field_w: float = DEFAULT_FIELD_W,
    field_h: float = DEFAULT_FIELD_H,
    ball_dt: float = DEFAULT_BALL_DT,
    ball_restitution: float = DEFAULT_BALL_RESTITUTION,
):
    print(f"Generating {num_samples} samples using brute-force reachability (bounce & score-aware)...")
    print(f"  Field: {field_w}x{field_h} m | restitution={ball_restitution} | ball_dt={ball_dt}")

    total_cores = multiprocessing.cpu_count()
    num_workers = max(1, int(total_cores * 0.75))
    print(f"  Using {num_workers}/{total_cores} parallel CPU workers (leaving ~25% free)...")

    valid_samples: list = []
    total_attempts = 0
    total_cpu_search_s = 0.0
    per_sample_search_s: list[float] = []

    seeds = np.random.randint(0, 2**31 - 1, size=num_samples)
    tasks = [
        (field_w, field_h, ball_dt, ball_restitution, int(seeds[i]))
        for i in range(num_samples)
    ]

    pbar = tqdm(total=num_samples, desc="Generating samples", unit="sample")
    wall_start = time.perf_counter()

    with multiprocessing.Pool(processes=num_workers) as pool:
        for sample, attempts, elapsed_s in pool.imap_unordered(_generate_single_sample, tasks):
            valid_samples.append(sample)
            total_attempts += attempts
            total_cpu_search_s += elapsed_s
            per_sample_search_s.append(elapsed_s)

            pbar.update(1)
            if len(valid_samples) % max(1, num_samples // 1000) == 0:
                pbar.set_postfix(acc=f"{len(valid_samples)/total_attempts*100:.1f}%")

    pbar.close()
    wall_clock_s = time.perf_counter() - wall_start

    dataset = np.array(valid_samples, dtype=np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, dataset)

    print(f"Data generation complete.")
    print(f"Total attempts: {total_attempts} for {num_samples} valid samples "
          f"(acceptance rate: {num_samples/total_attempts*100:.1f}%)")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Saved to {output_path}")
    print(f"Wall-clock time: {wall_clock_s:.1f} s | "
          f"Total CPU search: {total_cpu_search_s:.1f} s | "
          f"Workers: {num_workers}")

    # Write companion stats artifact
    stats = {
        "num_samples": num_samples,
        "total_attempts": total_attempts,
        "acceptance_rate": num_samples / max(1, total_attempts),
        "wall_clock_s": wall_clock_s,
        "total_cpu_search_s": total_cpu_search_s,
        "num_workers": num_workers,
        "mean_search_s_per_valid_sample": float(np.mean(per_sample_search_s)),
        "median_search_s_per_valid_sample": float(np.median(per_sample_search_s)),
        "mean_search_s_per_attempt": total_cpu_search_s / max(1, total_attempts),
        "generation_params": {
            "field_w": field_w,
            "field_h": field_h,
            "ball_dt": ball_dt,
            "ball_restitution": ball_restitution,
            "n_angles": 36,
            "t_min": 0.5,
            "t_max": 5.0,
            "t_step": 0.05,
        },
    }
    stats_path = os.path.join(os.path.dirname(output_path), "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Generation stats saved to {stats_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    from src.data_layout import STRIKE_DATASET

    generate_data(args.num_samples, str(STRIKE_DATASET))
