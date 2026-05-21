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
    goal = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance ball position by *dt* with axis-aligned inelastic wall reflections.

    Walls at x in [0, field_w] and y in [0, field_h]. On impact,
    velocity component normal to the wall is multiplied by -restitution.
    If a goal is provided, the ball can cross the right wall without bouncing
    if it is inside the goal mouth.
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
                # Check if ball enters the goal mouth (on right wall W)
                if goal is not None and goal.y_min <= pos[1] <= goal.y_max:
                    # No bounce, let it pass through
                    pass
                else:
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
    goal = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate bounce physics from t=0 to *total_time* using steps of at most *dt*."""
    pos = np.asarray(ball_pos, dtype=float).copy()
    vel = np.asarray(ball_vel, dtype=float).copy()
    t = 0.0
    while t < total_time - 1e-12:
        step_dt = min(dt, total_time - t)
        pos, vel = propagate_ball_step(
            pos, vel, step_dt, field_w, field_h, restitution, goal=goal
        )
        # If scored (went beyond x = field_w), stop propagating
        if goal is not None and pos[0] >= field_w - 1e-9 and goal.y_min <= pos[1] <= goal.y_max:
            break
        t += step_dt
    return pos, vel


def compute_strike_velocity(
    ball_vel: np.ndarray,
    v_car: float,
    theta_car: float,
    e_strike: float = 0.8,
) -> np.ndarray:
    """
    Computes the post-collision velocity of the ball after being struck by the car.
    Treats the car front bumper as a flat plane with normal n = [cos(theta_car), sin(theta_car)].
    """
    n = np.array([np.cos(theta_car), np.sin(theta_car)])
    v_car_vec = v_car * n
    
    # Relative velocity of ball with respect to car
    v_rel = ball_vel - v_car_vec
    
    # Component of relative velocity along the normal of collision
    v_rel_n = np.dot(v_rel, n)
    
    # If they are moving towards each other (v_rel_n < 0), perform impact reflection.
    # If they are already moving apart or car is pushing the ball from behind (v_rel_n >= 0),
    # the car bumper still gives a forward velocity if the car is moving faster.
    # To handle both cases robustly:
    if v_rel_n < 0:
        # Reflect relative normal velocity component
        v_rel_n_post = -e_strike * v_rel_n
        v_rel_post = v_rel - (1.0 + e_strike) * v_rel_n * n
        return v_rel_post + v_car_vec
    else:
        # If the car is moving faster in the normal direction than the ball,
        # it pushes the ball forward.
        v_ball_n = np.dot(ball_vel, n)
        if v_car > v_ball_n:
            # Transfer momentum along normal
            return ball_vel + (v_car - v_ball_n) * (1.0 + e_strike) * n
        return ball_vel

