"""
ball_physics.py — Shared inelastic wall-bounce model for the ball.

Used by World.step, data_generator labels, and main.py strike-target computation
so training, inference, and simulation stay consistent.
"""

from __future__ import annotations

import numpy as np

DEFAULT_FIELD_W = 10.0
DEFAULT_FIELD_H = 6.0
DEFAULT_BALL_RESTITUTION = 0.85
DEFAULT_BALL_DT = 0.1


def propagate_ball_step(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    dt: float,
    field_w: float = DEFAULT_FIELD_W,
    field_h: float = DEFAULT_FIELD_H,
    restitution: float = DEFAULT_BALL_RESTITUTION,
    max_bounces: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance ball position by *dt* with axis-aligned inelastic wall reflections.

    Walls at x in [0, field_w] and y in [0, field_h]. On impact,
    velocity component normal to the wall is multiplied by -restitution.
    """
    pos = np.asarray(ball_pos, dtype=float).copy()
    vel = np.asarray(ball_vel, dtype=float).copy()
    W, H = float(field_w), float(field_h)
    e = float(restitution)

    remaining = float(dt)
    for _ in range(max_bounces):
        if remaining <= 1e-12:
            break

        t_candidates = [remaining]
        if vel[0] < -1e-12 and pos[0] > 0.0:
            t_candidates.append(pos[0] / (-vel[0]))
        if vel[0] > 1e-12 and pos[0] < W:
            t_candidates.append((W - pos[0]) / vel[0])
        if vel[1] < -1e-12 and pos[1] > 0.0:
            t_candidates.append(pos[1] / (-vel[1]))
        if vel[1] > 1e-12 and pos[1] < H:
            t_candidates.append((H - pos[1]) / vel[1])

        t_move = min(t for t in t_candidates if t > 1e-12)
        pos = pos + vel * t_move
        remaining -= t_move

        bounced = False
        if pos[0] <= 1e-9:
            pos[0] = 0.0
            if vel[0] < 0.0:
                vel[0] = -e * vel[0]
                bounced = True
        elif pos[0] >= W - 1e-9:
            pos[0] = W
            if vel[0] > 0.0:
                vel[0] = -e * vel[0]
                bounced = True

        if pos[1] <= 1e-9:
            pos[1] = 0.0
            if vel[1] < 0.0:
                vel[1] = -e * vel[1]
                bounced = True
        elif pos[1] >= H - 1e-9:
            pos[1] = H
            if vel[1] > 0.0:
                vel[1] = -e * vel[1]
                bounced = True

        if not bounced and remaining > 1e-12:
            break

    return pos, vel


def propagate_ball_for_time(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    total_time: float,
    dt: float = DEFAULT_BALL_DT,
    field_w: float = DEFAULT_FIELD_W,
    field_h: float = DEFAULT_FIELD_H,
    restitution: float = DEFAULT_BALL_RESTITUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate bounce physics from t=0 to *total_time* using steps of at most *dt*."""
    pos = np.asarray(ball_pos, dtype=float).copy()
    vel = np.asarray(ball_vel, dtype=float).copy()
    t = 0.0
    while t < total_time - 1e-12:
        step_dt = min(dt, total_time - t)
        pos, vel = propagate_ball_step(
            pos, vel, step_dt, field_w, field_h, restitution
        )
        t += step_dt
    return pos, vel
