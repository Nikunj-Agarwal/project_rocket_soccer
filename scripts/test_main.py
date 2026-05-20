"""
test_main.py — Phase 3 Integration Test

Runs the real-time simulation loop for 5 distinct random seeds.
Evaluates:
  - Final position error (threshold: <= 0.2 m)
  - Final heading error (threshold: <= 0.15 rad)
  - Overall success rate (threshold: >= 4/5 goals)
  - Solver status (failures must be 0)
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

from src.main import run_simulation

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

    seeds = [10, 21, 32, 43, 54] # 5 distinct random seeds
    successes = 0
    failures = 0
    
    pos_errors = []
    heading_errors = []

    log.info("=" * 65)
    log.info("  PHASE 3 INTEGRATION TEST — Real-Time Interception Loop")
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

            if success:
                successes += 1
                log.info(f"  [SUCCESS] Interception achieved!")
            else:
                failures += 1
                log.info(f"  [FAILED]: Missed target.")
            
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
