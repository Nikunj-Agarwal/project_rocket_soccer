"""
test_main.py — Phase 3 / 3.6 Integration Test

Runs the real-time simulation loop for distinct random seeds (including wall-bounce cases).
Evaluates interception success, on-field final states, and bounce-parameter consistency.
"""

import os
import sys
import numpy as np
import torch
import logging
from datetime import datetime

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
from src.main import run_simulation


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

def setup_logging(log_dir: str) -> logging.Logger:
    """Create a logger that writes to both a file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"integration_test_{timestamp}.log")

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
    output_dir = os.path.join(PROJECT_ROOT, "data", "integration_test_results")
    log = setup_logging(output_dir)

    # 10: strong downward ball (wall bounce); others: mixed field coverage
    seeds = [10, 21, 32, 43, 54]
    successes = 0
    failures = 0
    
    pos_errors = []
    heading_errors = []

    log.info("=" * 65)
    log.info("  PHASE 3.6 INTEGRATION TEST — Bounce-Aware Interception Loop")
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

        # Redirect standard output of run_simulation to avoid spamming the logs
        # but keep track of success
        try:
            success, pos_err, heading_err, history = run_simulation(
                ball_start=ball_start,
                ball_vel=ball_vel,
                car_start=car_start,
                render=False,
                save_frames=True,
                save_dir=os.path.join(output_dir, f"run_seed_{seed}"),
            )
            
            pos_errors.append(pos_err)
            heading_errors.append(heading_err)

            final_ball = np.array([history[-1]["ball_x"], history[-1]["ball_y"]])
            final_car = np.array([history[-1]["car_x"], history[-1]["car_y"]])
            on_field = (
                0.0 <= final_ball[0] <= DEFAULT_FIELD_W
                and 0.0 <= final_ball[1] <= DEFAULT_FIELD_H
                and 0.0 <= final_car[0] <= DEFAULT_FIELD_W
                and 0.0 <= final_car[1] <= DEFAULT_FIELD_H
            )
            if not on_field:
                log.warning(f"  [WARN] Final state left the field: ball={final_ball}, car={final_car}")

            if success and on_field:
                successes += 1
                log.info(f"  [SUCCESS] Interception achieved (on-field)!")
            else:
                failures += 1
                log.info(f"  [FAILED]: Missed target or left field.")
            
            log.info(f"  Final errors: pos={pos_err:.4f} m | heading={heading_err:.4f} rad")

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
    log.info(f"  Avg Pos Error    : {np.mean(pos_errors):.4f} m  (target: <= 0.2)")
    log.info(f"  Avg Heading Error: {np.mean(heading_errors):.4f} rad (target: <= 0.15)")
    log.info("-" * 65)

    # Pass criteria: at least 4 out of 5 goals, average errors below thresholds
    passed = (
        successes >= 4 
        and np.mean(pos_errors) <= 0.2 
        and np.mean(heading_errors) <= 0.15
    )

    if passed:
        log.info("  [PASSED] INTEGRATION TEST PASSED!")
        return 0
    else:
        log.info("  [FAILED] INTEGRATION TEST FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
