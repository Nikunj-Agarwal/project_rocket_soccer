"""
test_main.py — Phase 5 Integration Test

Runs the real-time simulation loop for distinct random seeds (including wall-bounce cases).
Evaluates goal scoring success, on-field final states, and strike precision metrics.
"""

import argparse
import os
import sys
import numpy as np
import torch
import logging
import json
import subprocess
import pandas as pd

# Force non-interactive matplotlib backend BEFORE importing anything else
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
    propagate_ball_for_time,
)
from src.data_layout import (
    BATCH_LOG,
    new_integration_batch,
    integration_seed_run,
)
from src.main import run_simulation

# Default batch: 50 seeds for a comprehensive evaluation
DEFAULT_INTEGRATION_SEEDS = list(range(100, 150))


def check_bounce_target_consistency(
    ball_start: np.ndarray,
    ball_vel: np.ndarray,
    T_final: float,
    expected_pos: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """Strike target must match shared bounce integrator."""
    pos, _ = propagate_ball_for_time(
        ball_start,
        ball_vel,
        T_final,
        dt=DEFAULT_BALL_DT,
        field_w=DEFAULT_FIELD_W,
        field_h=DEFAULT_FIELD_H,
        restitution=DEFAULT_BALL_RESTITUTION,
    )
    return np.linalg.norm(pos - expected_pos) <= tol

def setup_logging(batch_dir: str) -> logging.Logger:
    """Create a logger that writes to both batch.log and stdout."""
    os.makedirs(batch_dir, exist_ok=True)
    log_path = os.path.join(batch_dir, BATCH_LOG)

    logger = logging.getLogger("integration_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s"))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger

def main():
    parser = argparse.ArgumentParser(description="Integration test with per-seed videos")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_INTEGRATION_SEEDS,
        help="Random seeds to run (default: 50 seeds, 100-149)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable saving simulation videos for faster runs",
    )
    args = parser.parse_args()
    seeds = args.seeds

    batch_dir = new_integration_batch()
    log = setup_logging(str(batch_dir))
    log.info(f"Batch directory: {batch_dir}")
    log.info(f"Seeds: {seeds}")
    successes = 0
    failures = 0
    
    pos_errors = []
    heading_errors = []

    log.info("=" * 65)
    log.info("  PHASE 5 INTEGRATION TEST — Strike & Score with Pursuit Warm-Start")
    log.info(f"  Bounce: restitution={DEFAULT_BALL_RESTITUTION}, dt={DEFAULT_BALL_DT}, field={DEFAULT_FIELD_W}x{DEFAULT_FIELD_H}")
    log.info("=" * 65)

    for i, seed in enumerate(seeds):
        log.info(f"\n--- Running Seed {seed} ({i+1}/{len(seeds)}) ---")
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate states exactly as in main.py to be consistent
        b_x = np.random.uniform(2.0, 8.0)
        b_y = np.random.uniform(0.0, 6.0)
        phi = np.random.uniform(0.0, 2 * np.pi)
        v_b = np.random.uniform(0.5, 2.0)
        b_vx = v_b * np.cos(phi)
        b_vy = v_b * np.sin(phi)
        
        c_x = np.random.uniform(0.0, 4.0)
        c_y = np.random.uniform(0.0, 6.0)
        c_theta = np.random.uniform(-np.pi, np.pi)

        ball_start = np.array([b_x, b_y])
        ball_vel = np.array([b_vx, b_vy])
        car_start = np.array([c_x, c_y, c_theta, 0.0])

        log.info(f"  Ball start: {ball_start} | Vel: {ball_vel}")
        log.info(f"  Car start : {car_start}")

        # Run simulation in a subprocess to completely avoid CasADi/IPOPT memory leaks and segfaults
        try:
            seed_run_dir = integration_seed_run(batch_dir, seed)
            
            # Setup command to run main.py as a subprocess
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "src", "main.py"),
                "--seed", str(seed),
                "--run-dir", str(seed_run_dir)
            ]
            if not args.no_video:
                cmd.append("--save-video")
                
            # Run the command and capture output
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Check for crash
            if result.returncode != 0:
                raise RuntimeError(
                    f"Subprocess failed with exit code {result.returncode}.\n"
                    f"Stdout:\n{result.stdout}\n"
                    f"Stderr:\n{result.stderr}"
                )
            
            # Read metadata.json and trajectory.csv
            meta_path = os.path.join(seed_run_dir, "metadata.json")
            traj_path = os.path.join(seed_run_dir, "trajectory.csv")
            
            if not os.path.exists(meta_path) or not os.path.exists(traj_path):
                raise FileNotFoundError(f"Subprocess completed but missing metadata or trajectory file in {seed_run_dir}")
                
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            df = pd.read_csv(traj_path)
            history = df.to_dict(orient="records")
            
            success = meta["success"]
            
            # Extract interception errors at the actual moment of strike (closest approach / contact)
            if history:
                closest_step = min(history, key=lambda h: h["pos_err"])
                strike_pos_err = closest_step["pos_err"]
                strike_heading_err = closest_step["heading_err"]
            else:
                strike_pos_err = meta["final_pos_err_m"]
                strike_heading_err = meta["final_heading_err_rad"]

            pos_errors.append(strike_pos_err)
            heading_errors.append(strike_heading_err)

            final_ball = np.array([history[-1]["ball_x"], history[-1]["ball_y"]])
            final_car = np.array([history[-1]["car_x"], history[-1]["car_y"]])
            
            # Ball is allowed to cross W if it was a goal
            ball_x_max = DEFAULT_FIELD_W + 2.0 if success else DEFAULT_FIELD_W
            on_field = (
                0.0 <= final_ball[0] <= ball_x_max
                and 0.0 <= final_ball[1] <= DEFAULT_FIELD_H
                and 0.0 <= final_car[0] <= DEFAULT_FIELD_W
                and 0.0 <= final_car[1] <= DEFAULT_FIELD_H
            )
            if not on_field:
                log.warning(f"  [WARN] Final state left the field: ball={final_ball}, car={final_car}")

            if success and on_field:
                successes += 1
                log.info(f"  [SUCCESS] Goal scored!")
            else:
                failures += 1
                log.info(f"  [FAILED]: Missed goal or left field.")
            
            log.info(f"  Strike errors: pos={strike_pos_err:.4f} m | heading={strike_heading_err:.4f} rad")

        except Exception as e:
            log.error(f"  [CRASH] Run crashed with exception: {e}")
            failures += 1
            pos_errors.append(999.0)
            heading_errors.append(999.0)

    # Summary
    log.info("")
    log.info("=" * 65)
    log.info("  INTEGRATION TEST SUMMARY")
    log.info("=" * 65)
    log.info(f"  Total runs       : {len(seeds)}")
    log.info(f"  Goals (Success)  : {successes} / {len(seeds)}")
    log.info(f"  Misses (Fail)    : {failures} / {len(seeds)}")
    log.info(f"  Avg Strike Pos Error    : {np.mean(pos_errors):.4f} m  (target: <= 0.35)")
    log.info(f"  Avg Strike Heading Error: {np.mean(heading_errors):.4f} rad (target: <= 0.25)")
    log.info("-" * 65)

    # Pass criteria: at least 60% success, average errors below thresholds
    min_successes = max(1, int(np.ceil(0.6 * len(seeds))))
    passed = (
        successes >= min_successes
        and np.mean(pos_errors) <= 0.35
        and np.mean(heading_errors) <= 0.25
    )
    log.info(f"  Pass threshold     : {min_successes} / {len(seeds)} successes (60%)")

    if passed:
        log.info("  [PASSED] INTEGRATION TEST PASSED!")
        return 0
    else:
        log.info("  [FAILED] INTEGRATION TEST FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
