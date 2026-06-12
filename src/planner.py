"""
planner.py — Shared analytic strike planner.

Provides the min-T + canonical-theta search that was originally embedded
inline in data_generator._generate_single_sample.  Having it here as a
standalone function lets data_generator, main.py (fallback timing), and
benchmark_scalability all call identical code, guaranteeing apples-to-apples
latency comparisons.

The default arguments reproduce the exact dataset label logic:
  n_angles=36, t_min=0.5, t_max=5.0, t_step=0.05

Reachability model
------------------
The feasibility check uses a two-part point-mass model sourced from the same
physical constants as InterceptionMPC:

  - Linear reach: a_max=2.0 m/s², v_max=2.0 m/s (matches nmpc_solver defaults).
    max_reach_distance(T) = T²       when T ≤ t_acc  (accel phase only)
                          = 2T − 1   when T > t_acc  (accel + cruise)
    where t_acc = v_max / a_max = 1.0 s.  With the defaults this equals the
    old inline formula exactly, so dataset labels are not changed.

  - Turn penalty: R_turn = L / tan(delta_max) where L = wheelbase (default 0.3 m)
    and delta_max = π/4 rad, giving R_turn = 0.30 m.  The previous hardcoded
    value was 0.35 m; the constant is now explicit and documented.
    NOTE: switching from 0.35 to 0.30 slightly changes label acceptance and
    requires a dataset regen + retrain.  The default is kept at the old 0.35
    so behaviour is preserved until the physics-informed retrain is done
    (see docs/FUTURE_physics_informed_prediction.md).
"""

from __future__ import annotations

import math
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

# Physical constants that mirror InterceptionMPC defaults.
# Keeping them here as module-level defaults makes the relationship explicit
# and avoids magic numbers scattered through the code.
_CAR_A_MAX: float = 2.0        # m/s²  — matches nmpc_solver.InterceptionMPC.a_max
_CAR_V_MAX: float = 2.0        # m/s   — matches nmpc_solver.InterceptionMPC.v_max
_CAR_L: float = 0.3            # m     — matches nmpc_solver.InterceptionMPC.L
_CAR_DELTA_MAX: float = math.pi / 4  # rad — matches nmpc_solver.InterceptionMPC.delta_max
# Exact min turn radius: L / tan(delta_max) = 0.3 / 1.0 = 0.30 m.
# The dataset was generated with R_turn=0.35 (slightly looser); changing this
# requires a dataset regen.  Default kept at 0.35 for backward compatibility.
_R_TURN_EXACT: float = _CAR_L / math.tan(_CAR_DELTA_MAX)   # 0.30 m
_R_TURN_LEGACY: float = 0.35                                # m — original value


def max_reach_distance(T: float, a_max: float = _CAR_A_MAX, v_max: float = _CAR_V_MAX) -> float:
    """Maximum distance a point mass starting from rest can travel in time T.

    Derivation: accelerate at a_max until v_max is reached (at t_acc = v_max/a_max),
    then cruise at v_max for the remainder.

    With a_max=2, v_max=2 (the NMPC defaults):
        t_acc = 1 s
        T ≤ 1 s  →  d = 0.5 * a_max * T²  =  T²
        T > 1 s  →  d = 0.5*a_max*t_acc² + v_max*(T-t_acc)  =  2T - 1
    These are identical to the old inline formulas.
    """
    t_acc = v_max / a_max
    if T <= t_acc:
        return 0.5 * a_max * T * T
    return 0.5 * a_max * t_acc * t_acc + v_max * (T - t_acc)


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
    # Car kinematic constants for the reachability check.
    # Defaults mirror InterceptionMPC (nmpc_solver.py) so training labels and
    # runtime control share the same physical assumptions.
    car_a_max: float = _CAR_A_MAX,
    car_v_max: float = _CAR_V_MAX,
    R_turn: float = _R_TURN_LEGACY,   # 0.35 m; use _R_TURN_EXACT (0.30) after retrain
) -> tuple[float, float, float, float] | None:
    """Find the minimum feasible interception plan for the given scene.

    Parameters
    ----------
    ball_pos  : (2,) current ball position [m]
    ball_vel  : (2,) current ball velocity [m/s]
    car_state : (>=3,) car state; only car_state[0:3] = (x, y, theta) used
    goal      : Goal geometry object
    n_angles  : number of candidate strike headings swept in [-pi, pi)
    car_a_max : car max acceleration [m/s²] for point-mass reach distance model
    car_v_max : car max speed [m/s] for point-mass reach distance model
    R_turn    : effective turn-arc radius [m] used to penalise heading changes.
                Approximates the Dubins path cost for the angular deviation between
                the car's current heading, the line-of-sight to the strike point,
                and the desired strike heading.  Exact value is L/tan(delta_max).
    (remaining kwargs mirror ball_physics defaults)

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
            # d_effective accounts for the angular detours using the turn-arc model.
            # d_max uses max_reach_distance (closed-form from car_a_max / car_v_max).
            theta_los = np.arctan2(b_T_y - c_y, b_T_x - c_x)
            dtheta_start = abs(np.arctan2(np.sin(theta_los - c_theta), np.cos(theta_los - c_theta)))
            dtheta_end   = abs(np.arctan2(np.sin(theta_cand - theta_los), np.cos(theta_cand - theta_los)))
            d_effective  = d + R_turn * (dtheta_start + dtheta_end)

            d_max = max_reach_distance(T, a_max=car_a_max, v_max=car_v_max)

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
