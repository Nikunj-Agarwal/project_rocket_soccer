"""
test_static_target.py — Phase 1 smoke test.

Drives the car from (1,3) to a STATIC target at (5,3) using the
shrinking-horizon NMPC.  No ball movement, no neural network.

Uses the non-interactive Agg backend (no popup windows).
All output goes to data/tests/static/{batch_id}/ (trajectory, video, batch.log).

Pass criteria:
  - Position error < 0.2 m
  - Heading error  < 0.1 rad
  - No solver failures
"""

import sys
import os
import time
import logging
from datetime import datetime

# Force non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

# Add project root to path so we can import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_layout import BATCH_LOG, TRAJECTORY_CSV, new_static_test_batch
from src.recording import SimulationRecorder, render_and_capture
from src.simulator import World
from src.nmpc_solver import InterceptionMPC


def setup_logging(batch_dir: str) -> logging.Logger:
    """Create a logger that writes to both batch.log and stdout."""
    os.makedirs(batch_dir, exist_ok=True)
    log_path = os.path.join(batch_dir, BATCH_LOG)

    logger = logging.getLogger("smoke_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s"))

    # Console handler — same detail
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


def main():
    batch_dir = new_static_test_batch()
    log = setup_logging(str(batch_dir))
    recorder = SimulationRecorder()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    car_start = np.array([1.0, 3.0, 0.0, 0.0])   # at rest, facing right
    target = np.array([5.0, 3.0, 0.0, 1.0])       # 4 m ahead, v=1 m/s
    goal_pos = np.array([9.5, 3.0])
    ball_pos = target[:2].copy()                    # ball sitting at target
    ball_vel = np.array([0.0, 0.0])                 # no movement
    N_total = 30                                    # horizon steps

    log.info("=" * 65)
    log.info("  PHASE 1 SMOKE TEST — Static Target Interception")
    log.info("=" * 65)
    log.info(f"  Car start   : {car_start}")
    log.info(f"  Target state: {target}")
    log.info(f"  Goal pos    : {goal_pos}")
    log.info(f"  Horizon     : {N_total} steps  (dt=0.1 → T={N_total*0.1:.1f}s)")
    log.info("-" * 65)

    world = World(
        car_state=car_start.copy(),
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        goal_pos=goal_pos,
    )
    mpc = InterceptionMPC()

    # ------------------------------------------------------------------
    # Shrinking-horizon loop
    # ------------------------------------------------------------------
    solver_failures = 0
    history = []      # for the log summary
    total_solve_ms = 0.0

    for step in range(N_total):
        N_remaining = N_total - step

        t0 = time.perf_counter()
        u0 = mpc.solve(world.car_state, target, N_remaining)
        solve_ms = (time.perf_counter() - t0) * 1000
        total_solve_ms += solve_ms

        if np.allclose(u0, 0.0) and N_remaining > 1:
            solver_failures += 1
            log.warning(f"Step {step}: SOLVER RETURNED ZERO (possible failure)")

        world.step(u0)

        # Compute errors
        pos_err = np.linalg.norm(world.car_state[:2] - target[:2])
        heading_err = abs(np.arctan2(
            np.sin(world.car_state[2] - target[2]),
            np.cos(world.car_state[2] - target[2]),
        ))
        speed = world.car_state[3]

        row = {
            "step": step,
            "N_rem": N_remaining,
            "x": world.car_state[0],
            "y": world.car_state[1],
            "theta": world.car_state[2],
            "v": speed,
            "a": u0[0],
            "delta": u0[1],
            "pos_err": pos_err,
            "heading_err": heading_err,
            "solve_ms": solve_ms,
        }
        history.append(row)

        log.info(
            f"Step {step:3d} | N={N_remaining:3d} | "
            f"pos=({row['x']:6.3f},{row['y']:6.3f}) | θ={row['theta']:+6.3f} | "
            f"v={speed:5.3f} | u=[{u0[0]:+6.3f},{u0[1]:+6.3f}] | "
            f"err_p={pos_err:.3f} err_θ={heading_err:.3f} | "
            f"{solve_ms:6.1f}ms"
        )

        render_and_capture(world, f"Step {step}  |  pos_err={pos_err:.2f} m", recorder)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    final_pos_err = np.linalg.norm(world.car_state[:2] - target[:2])
    final_heading_err = abs(np.arctan2(
        np.sin(world.car_state[2] - target[2]),
        np.cos(world.car_state[2] - target[2]),
    ))

    log.info("")
    log.info("=" * 65)
    log.info("  RESULTS")
    log.info("=" * 65)
    log.info(f"  Final car state      : {world.car_state}")
    log.info(f"  Target state         : {target}")
    log.info(f"  Final position error : {final_pos_err:.4f} m   (threshold: 0.2)")
    log.info(f"  Final heading error  : {final_heading_err:.4f} rad (threshold: 0.1)")
    log.info(f"  Solver failures      : {solver_failures}")
    log.info(f"  Avg solve time       : {total_solve_ms / N_total:.1f} ms")
    log.info(f"  Total solve time     : {total_solve_ms:.0f} ms")
    video_path = recorder.save(batch_dir, fps=10.0)
    if video_path:
        log.info(f"  Simulation video     : {video_path}")
    log.info("-" * 65)

    passed = (
        final_pos_err < 0.2
        and final_heading_err < 0.1
        and solver_failures == 0
    )

    if passed:
        log.info("  ✅  TEST PASSED")
    else:
        log.info("  ❌  TEST FAILED")
        if final_pos_err >= 0.2:
            log.info(f"       Position error too large: {final_pos_err:.4f} >= 0.2")
        if final_heading_err >= 0.1:
            log.info(f"       Heading error too large:  {final_heading_err:.4f} >= 0.1")
        if solver_failures > 0:
            log.info(f"       Solver failures:          {solver_failures}")

    log.info("=" * 65)

    # ------------------------------------------------------------------
    # Write CSV summary for easy post-analysis
    # ------------------------------------------------------------------
    csv_path = os.path.join(batch_dir, TRAJECTORY_CSV)
    with open(csv_path, "w") as f:
        cols = list(history[0].keys())
        f.write(",".join(cols) + "\n")
        for row in history:
            f.write(",".join(f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c])
                            for c in cols) + "\n")
    log.info(f"  Trajectory CSV: {csv_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
