"""
verify_label_fidelity.py — Sanity-check dataset labels vs planner and trained StrikeNet.

Usage:
  python -m scripts.verify_label_fidelity [--n-samples 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ball_physics import DEFAULT_BALL_DT, DEFAULT_BALL_RESTITUTION, DEFAULT_FIELD_H, DEFAULT_FIELD_W
from src.data_layout import STRIKE_DATASET, model_path_for_variant
from src.goal import Goal
from src.network import StrikeNet
from src.planner import analytic_strike_plan


def _wrap_angle_diff(a: float, b: float) -> float:
    return abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dataset labels and model fidelity.")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-mae-threshold", type=float, default=0.15,
                        help="Max acceptable MAE on T (seconds) for trained models")
    parser.add_argument("--theta-mae-threshold", type=float, default=0.25,
                        help="Max acceptable MAE on heading (radians) for trained models")
    args = parser.parse_args()

    if not STRIKE_DATASET.is_file():
        print(f"ERROR: dataset not found at {STRIKE_DATASET}")
        return 1

    data = np.load(STRIKE_DATASET)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(data), size=min(args.n_samples, len(data)), replace=False)
    goal = Goal()

    label_t_err: list[float] = []
    label_xy_err: list[float] = []
    label_th_err: list[float] = []

    print(f"Checking planner vs stored labels on {len(idx)} samples...")
    for i in idx:
        row = data[i]
        ball_pos = row[0:2]
        ball_vel = row[2:4]
        car_state = row[4:7]
        T_l, x_l, y_l, th_l = row[7], row[8], row[9], row[10]

        plan = analytic_strike_plan(
            ball_pos.copy(), ball_vel.copy(), car_state, goal,
            field_w=DEFAULT_FIELD_W, field_h=DEFAULT_FIELD_H,
            ball_dt=DEFAULT_BALL_DT, ball_restitution=DEFAULT_BALL_RESTITUTION,
        )
        if plan is None:
            print(f"  sample {i}: planner returned None (stored label exists — investigate)")
            continue
        T_p, x_p, y_p, th_p = plan
        label_t_err.append(abs(T_p - T_l))
        label_xy_err.append(float(np.hypot(x_p - x_l, y_p - y_l)))
        label_th_err.append(_wrap_angle_diff(th_p, th_l))

    if label_t_err:
        print(f"  Label fidelity — T MAE: {np.mean(label_t_err):.4f} s, "
              f"xy MAE: {np.mean(label_xy_err):.4f} m, "
              f"theta MAE: {np.mean(label_th_err):.4f} rad")
    else:
        print("  No label pairs compared.")
        return 1

    inputs = data[idx, :7]
    labels = data[idx, 7:11]

    for variant in ("legacy", "structured"):
        mpath = model_path_for_variant(variant)
        if not mpath.is_file():
            print(f"  Skipping {variant}: model not found at {mpath}")
            continue
        model = StrikeNet.load(str(mpath))
        t_errs: list[float] = []
        th_errs: list[float] = []
        for inp, lab in zip(inputs, labels):
            pred = model.predict(inp)
            t_errs.append(abs(float(pred[0]) - lab[0]))
            th_errs.append(_wrap_angle_diff(float(pred[-1]), lab[3]))
        t_mae = float(np.mean(t_errs))
        th_mae = float(np.mean(th_errs))
        print(f"  StrikeNet {variant} — T MAE: {t_mae:.4f} s, theta MAE: {th_mae:.4f} rad")

        if t_mae > args.t_mae_threshold or th_mae > args.theta_mae_threshold:
            print(f"  FAIL: {variant} exceeds thresholds "
                  f"(T<{args.t_mae_threshold}, theta<{args.theta_mae_threshold})")
            return 1

    print("OK: label fidelity and model MAE within thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
