"""
planner.py — Shared analytic strike planner.

Provides the min-T + canonical-theta search that was originally embedded
inline in data_generator._generate_single_sample.  Having it here as a
standalone function lets data_generator, main.py (fallback timing), and
benchmark_scalability all call identical code, guaranteeing apples-to-apples
latency comparisons.

The default arguments reproduce the exact dataset label logic:
  n_angles=36, t_min=0.5, t_max=5.0, t_step=0.05, R_turn=0.35
"""

from __future__ import annotations

import numpy as np

from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
    propagate_ball_step,
    propagate_ball_for_time,
    compute_strike_velocity,
)
from src.goal import Goal


def analytic_strike_plan(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    car_state: np.ndarray,
    goal: Goal,
    *,
    field_w: float = DEFAULT_FIELD_W,
    field_h: float = DEFAULT_FIELD_H,
    ball_dt: float = DEFAULT_BALL_DT,
    ball_restitution: float = DEFAULT_BALL_RESTITUTION,
    v_impact: float = 1.0,
    e_strike: float = 0.8,
    n_angles: int = 36,
    t_min: float = 0.5,
    t_max: float = 5.0,
    t_step: float = 0.05,
    R_turn: float = 0.35,
) -> tuple[float, float, float, float] | None:
    """Find the minimum feasible interception plan for the given scene.

    Parameters
    ----------
    ball_pos : (2,) current ball position [m]
    ball_vel : (2,) current ball velocity [m/s]
    car_state : (>=3,) car state; only car_state[0:3] = (x, y, theta) used
    goal      : Goal geometry object
    n_angles  : number of candidate strike headings swept in [-pi, pi)
    (all other kwargs mirror ball_physics defaults)

    Returns
    -------
    (T, x_strike, y_strike, theta_strike) on success, or None if no feasible
    plan exists within [t_min, t_max].
    """
    c_x, c_y, c_theta = float(car_state[0]), float(car_state[1]), float(car_state[2])

    bp = np.array(ball_pos, dtype=float)
    bv = np.array(ball_vel, dtype=float)

    t_current = 0.0

    for T in np.arange(t_min, t_max + 1e-9, t_step):
        # Incremental integration: advance from where we left off rather than
        # restarting from ball_pos each iteration (same trick as data_generator).
        while t_current < T - 1e-12:
            step_dt = min(ball_dt, T - t_current)
            bp, bv = propagate_ball_step(
                bp, bv, step_dt,
                field_w=field_w, field_h=field_h, restitution=ball_restitution,
            )
            t_current += step_dt

        b_T_x, b_T_y = float(bp[0]), float(bp[1])

        d = np.hypot(b_T_x - c_x, b_T_y - c_y)

        theta_candidates = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
        theta_los_goal = np.arctan2(goal.center[1] - b_T_y, goal.center[0] - b_T_x)

        feasible_thetas: list[float] = []
        for theta_cand in theta_candidates:
            # 1. Does this heading redirect the ball into the goal?
            v_post = compute_strike_velocity(bv, v_car=v_impact, theta_car=theta_cand, e_strike=e_strike)
            final_pos, _ = propagate_ball_for_time(
                bp, v_post, total_time=5.0,
                dt=ball_dt, field_w=field_w, field_h=field_h,
                restitution=ball_restitution, goal=goal,
            )
            if not (final_pos[0] >= goal.x - 1e-9 and goal.y_min <= final_pos[1] <= goal.y_max):
                continue

            # 2. Is the strike point geometrically reachable in time T?
            theta_los = np.arctan2(b_T_y - c_y, b_T_x - c_x)
            dtheta_start = abs(np.arctan2(np.sin(theta_los - c_theta), np.cos(theta_los - c_theta)))
            dtheta_end   = abs(np.arctan2(np.sin(theta_cand - theta_los), np.cos(theta_cand - theta_los)))
            d_effective  = d + R_turn * (dtheta_start + dtheta_end)

            if T <= 1.0:
                d_max = T ** 2
            else:
                d_max = 2.0 * T - 1.0

            if d_effective <= d_max:
                feasible_thetas.append(float(theta_cand))

        if feasible_thetas:
            # Canonical label: scoring+reachable heading closest to goal LOS
            theta_strike = min(
                feasible_thetas,
                key=lambda th: abs(np.arctan2(np.sin(th - theta_los_goal), np.cos(th - theta_los_goal))),
            )
            return float(T), b_T_x, b_T_y, theta_strike

    return None
