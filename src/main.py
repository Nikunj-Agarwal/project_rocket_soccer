"""
main.py — Real-time simulation loop.

Loads the trained StrikeNet model, initializes the simulator (World),
queries StrikeNet for a target interception point, and runs the
shrinking-horizon NMPC loop using InterceptionMPC.
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_layout import (
    RUN_METADATA,
    TRAJECTORY_CSV,
    new_manual_run,
)
from src.recording import SimulationRecorder, render_and_capture
from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
    propagate_ball_for_time,
    compute_strike_velocity,
)
from src.simulator import World
from src.nmpc_solver import InterceptionMPC
from src.network import StrikeNet
from src.goal import Goal

def run_simulation(
    ball_start: np.ndarray,
    ball_vel: np.ndarray,
    car_start: np.ndarray,
    goal: Goal = None,
    model_path: str = None,
    render: bool = False,
    save_video: bool = False,
    run_dir: str | Path | None = None,
    video_fps: float = 10.0,
    run_metadata: dict | None = None,
    v_impact: float = 1.0,
    field_size: tuple = (DEFAULT_FIELD_W, DEFAULT_FIELD_H),
    ball_restitution: float = DEFAULT_BALL_RESTITUTION,
    dt: float = DEFAULT_BALL_DT,
):
    """
    Runs the closed-loop shrinking-horizon NMPC simulation.
    """
    if goal is None:
        goal = Goal()

    # 1. Load the trained StrikeNet model
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "strategy_net.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the network first.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StrikeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare inputs for StrikeNet
    # Input schema: [ball_x, ball_y, ball_vx, ball_vy, car_x, car_y, car_theta]
    inputs = np.array([
        ball_start[0], ball_start[1], ball_vel[0], ball_vel[1],
        car_start[0], car_start[1], car_start[2]
    ], dtype=np.float32)

    # 3. Query StrikeNet
    preds = model.predict(inputs) # Returns [T_strike, x_strike, y_strike, theta_strike]
    T_strike = float(preds[0])
    x_strike = float(preds[1])
    y_strike = float(preds[2])
    theta_strike = float(preds[3])

    # Convert T_strike to discrete steps
    N_steps = int(round(T_strike / dt))
    N_steps = max(1, min(50, N_steps)) # Clip between 0.1s and 5.0s

    print("=" * 65)
    print("  STRIKENET PREDICTION")
    print("=" * 65)
    print(f"  Predicted T_strike    : {T_strike:.3f} s  ({N_steps} steps)")
    print(f"  Predicted x_strike    : {x_strike:.3f} m")
    print(f"  Predicted y_strike    : {y_strike:.3f} m")
    print(f"  Predicted theta_strike : {theta_strike:+.3f} rad ({np.degrees(theta_strike):.1f} deg)")
    print("-" * 65)

    # Define strike state for NMPC
    # Bounce-aware ball position at T_final (matches World.step physics)
    T_final = N_steps * dt
    field_w, field_h = field_size
    strike_pos, strike_vel = propagate_ball_for_time(
        ball_start,
        ball_vel,
        T_final,
        dt=dt,
        field_w=field_w,
        field_h=field_h,
        restitution=ball_restitution,
    )
    x_strike_exact = float(strike_pos[0])
    y_strike_exact = float(strike_pos[1])
    
    # Sweep candidates at runtime to find exact scoring heading
    theta_candidates = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    best_theta = None
    for theta_cand in theta_candidates:
        v_post = compute_strike_velocity(strike_vel, v_car=v_impact, theta_car=theta_cand, e_strike=0.8)
        final_pos, _ = propagate_ball_for_time(
            strike_pos,
            v_post,
            total_time=5.0,
            dt=dt,
            field_w=field_w,
            field_h=field_h,
            restitution=ball_restitution,
            goal=goal,
        )
        if final_pos[0] >= goal.x - 1e-9 and goal.y_min <= final_pos[1] <= goal.y_max:
            best_theta = theta_cand
            break
            
    if best_theta is None:
        # Fallback to line of sight to goal center
        best_theta = np.arctan2(goal.center[1] - y_strike_exact, goal.center[0] - x_strike_exact)
        
    theta_strike_exact = best_theta

    print(f"  Ball bounce           : restitution={ball_restitution}, field={field_w}x{field_h} m")
    print(f"  Exact strike target   : x={x_strike_exact:.3f} m, y={y_strike_exact:.3f} m, theta={theta_strike_exact:+.3f} rad")
    print("-" * 65)
    
    offset_dist = 0.32
    x_target = x_strike_exact - offset_dist * np.cos(theta_strike_exact)
    y_target = y_strike_exact - offset_dist * np.sin(theta_strike_exact)
    q_strike = np.array([x_target, y_target, theta_strike_exact, v_impact])
    
    # Save the original theta_strike for logging/error evaluation
    theta_strike_eval = theta_strike_exact

    # 4. Initialize World and InterceptionMPC
    world = World(
        car_state=car_start.copy(),
        ball_pos=ball_start.copy(),
        ball_vel=ball_vel.copy(),
        goal_pos=goal.center,
        dt=dt,
        field_size=field_size,
        ball_restitution=ball_restitution,
        goal=goal,
    )
    Q_term = np.diag([500.0, 500.0, 100.0, 1.0])
    R_weights = np.diag([0.01, 0.01])
    mpc = InterceptionMPC(dt=dt, Q_terminal=Q_term, R=R_weights)

    run_path = Path(run_dir) if run_dir else None
    if run_path is not None:
        run_path.mkdir(parents=True, exist_ok=True)

    recorder = SimulationRecorder() if (save_video and run_path) else None

    # Setup rendering backend
    if not render:
        matplotlib.use("Agg")

    # 5. Phase 1: NMPC Interception & Strike Loop
    history = []
    solver_failures = 0
    total_solve_ms = 0.0
    struck = False
    strike_step = None

    print("Running shrinking-horizon NMPC simulation...")
    for step in range(N_steps):
        N_remaining = N_steps - step

        # Get current states
        current_car = world.car_state.copy()
        
        # Solve NMPC
        t0 = time.perf_counter()
        u0 = mpc.solve(current_car, q_strike, N_remaining)
        solve_ms = (time.perf_counter() - t0) * 1000
        total_solve_ms += solve_ms

        if np.allclose(u0, 0.0) and N_remaining > 1:
            solver_failures += 1
            print(f"[NMPC] Warning at step {step}: Solver returned zero control.")

        # Advance world physics
        world.step(u0)

        # Compute errors relative to current ball position
        pos_err = np.linalg.norm(world.car_state[:2] - world.ball_pos)
        heading_err = abs(np.arctan2(
            np.sin(world.car_state[2] - theta_strike_eval),
            np.cos(world.car_state[2] - theta_strike_eval)
        ))

        step_data = {
            "step": step,
            "phase": "approach",
            "N_rem": N_remaining,
            "car_x": world.car_state[0],
            "car_y": world.car_state[1],
            "car_theta": world.car_state[2],
            "car_v": world.car_state[3],
            "ball_x": world.ball_pos[0],
            "ball_y": world.ball_pos[1],
            "u_acc": u0[0],
            "u_steer": u0[1],
            "pos_err": pos_err,
            "heading_err": heading_err,
            "solve_ms": solve_ms,
        }
        history.append(step_data)

        title = f"Step {step} | N_rem={N_remaining} | pos_err={pos_err:.2f}m"
        if recorder is not None:
            render_and_capture(world, title, recorder)
        elif render:
            import matplotlib.pyplot as plt
            world.render(title=title)

        if world.ball_struck:
            struck = True
            strike_step = step
            print(f"Collision/Strike detected at step {step}!")
            break

    # 6. Phase 2: Post-Strike Coasting & Ball Flight
    POST_STRIKE_STEPS = 50 # up to 5 seconds
    print("Running post-strike propagation...")
    for post_step in range(POST_STRIKE_STEPS):
        if world.scored:
            print("Ball entered the goal!")
            break

        # Active braking to decelerate the car and keep it on the field
        v_car = world.car_state[3]
        a_brake = np.clip(-v_car / dt, -2.0, 0.0) if v_car > 0 else 0.0
        world.step(np.array([a_brake, 0.0]))

        step_data = {
            "step": (strike_step if strike_step is not None else N_steps) + 1 + post_step,
            "phase": "post_strike",
            "N_rem": 0,
            "car_x": world.car_state[0],
            "car_y": world.car_state[1],
            "car_theta": world.car_state[2],
            "car_v": world.car_state[3],
            "ball_x": world.ball_pos[0],
            "ball_y": world.ball_pos[1],
            "u_acc": 0.0,
            "u_steer": 0.0,
            "pos_err": np.linalg.norm(world.car_state[:2] - world.ball_pos),
            "heading_err": 0.0,
            "solve_ms": 0.0,
        }
        history.append(step_data)

        title = f"Post-strike step {post_step} | Scored: {world.scored}"
        if recorder is not None:
            render_and_capture(world, title, recorder)
        elif render:
            import matplotlib.pyplot as plt
            world.render(title=title)

    # Final evaluation
    final_car = world.car_state
    final_ball = world.ball_pos
    final_pos_err = np.linalg.norm(final_car[:2] - final_ball)
    
    # Wrap heading error
    final_heading_err = abs(np.arctan2(
        np.sin(final_car[2] - theta_strike_eval),
        np.cos(final_car[2] - theta_strike_eval)
    ))

    print("\n" + "=" * 65)
    print("  SIMULATION RESULTS")
    print("=" * 65)
    print(f"  Final car state      : {final_car}")
    print(f"  Final ball state     : {final_ball}")
    print(f"  Final position error : {final_pos_err:.4f} m")
    print(f"  Final heading error  : {final_heading_err:.4f} rad")
    print(f"  Goal scored          : {world.scored}")
    print(f"  Solver failures      : {solver_failures}")
    print(f"  Avg solve time       : {total_solve_ms / max(1, step+1):.1f} ms")
    
    success = world.scored
    if success:
        print("  [SUCCESS] GOAL scored!")
    else:
        print("  [FAILED] MISSED GOAL")
    print("-" * 65)

    if run_path is not None:
        import pandas as pd

        csv_path = run_path / TRAJECTORY_CSV
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"Trajectory saved to {csv_path}")

        if recorder is not None:
            video_path = recorder.save(run_path, fps=video_fps)
            if video_path:
                print(f"Simulation video saved to {video_path}")

        meta = {
            "success": success,
            "final_pos_err_m": float(final_pos_err),
            "final_heading_err_rad": float(final_heading_err),
            "solver_failures": solver_failures,
            "N_steps": N_steps,
            "T_final_s": float(T_final),
            "ball_restitution": ball_restitution,
            "field_size_m": list(field_size),
            "strike_target": [x_strike_exact, y_strike_exact, theta_strike_exact],
            "scored": bool(world.scored),
            "ball_struck": bool(world.ball_struck),
            "strike_step": strike_step,
        }
        if run_metadata:
            meta.update(run_metadata)
        meta_path = run_path / RUN_METADATA
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Run metadata saved to {meta_path}")

    return success, final_pos_err, final_heading_err, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--render", action="store_true", help="Interactive rendering")
    parser.add_argument("--save-video", action="store_true", help="Save trajectory CSV + simulation.mp4")
    parser.add_argument("--run-dir", type=str, default=None, help="Output run directory (default: data/runs/manual/...)")
    parser.add_argument("--video-fps", type=float, default=10.0, help="FPS for saved simulation video")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Randomize ball and car state matching generator distributions
    # Ball
    b_x = np.random.uniform(2.0, 8.0)
    b_y = np.random.uniform(0.0, 6.0)
    phi = np.random.uniform(0.0, 2 * np.pi)
    v_b = np.random.uniform(0.5, 2.0)
    b_vx = v_b * np.cos(phi)
    b_vy = v_b * np.sin(phi)
    
    # Car
    c_x = np.random.uniform(0.0, 4.0)
    c_y = np.random.uniform(0.0, 6.0)
    c_theta = np.random.uniform(-np.pi, np.pi)

    ball_start = np.array([b_x, b_y])
    ball_vel = np.array([b_vx, b_vy])
    car_start = np.array([c_x, c_y, c_theta, 0.0]) # starting at rest

    print(f"Initialized simulation with random seed: {args.seed}")
    print(f"Ball start: {ball_start} | Vel: {ball_vel}")
    print(f"Car start : {car_start}")

    run_dir = args.run_dir
    if args.save_video and run_dir is None:
        run_dir = str(new_manual_run(seed=args.seed))

    run_simulation(
        ball_start=ball_start,
        ball_vel=ball_vel,
        car_start=car_start,
        render=args.render,
        save_video=args.save_video,
        run_dir=run_dir,
        video_fps=args.video_fps,
        run_metadata={
            "seed": args.seed,
            "ball_start": ball_start.tolist(),
            "ball_vel": ball_vel.tolist(),
            "car_start": car_start.tolist(),
        },
    )
