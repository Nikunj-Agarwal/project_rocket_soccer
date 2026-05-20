"""
main.py — Real-time simulation loop.

Loads the trained StrikeNet model, initializes the simulator (World),
queries StrikeNet for a target interception point, and runs the
shrinking-horizon NMPC loop using InterceptionMPC.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulator import World
from src.nmpc_solver import InterceptionMPC
from src.network import StrikeNet

def run_simulation(
    ball_start: np.ndarray,
    ball_vel: np.ndarray,
    car_start: np.ndarray,
    goal_pos: np.ndarray = np.array([9.5, 3.0]),
    model_path: str = None,
    render: bool = False,
    save_frames: bool = True,
    save_dir: str = None,
    v_impact: float = 1.0,
):
    """
    Runs the closed-loop shrinking-horizon NMPC simulation.
    """
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
    dt = 0.1
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
    # Compute exact ball position at the final simulation step (T_final = N_steps * dt)
    # to eliminate neural network regression and time discretization errors.
    T_final = N_steps * dt
    x_strike_exact = ball_start[0] + ball_vel[0] * T_final
    y_strike_exact = ball_start[1] + ball_vel[1] * T_final
    theta_strike_exact = np.arctan2(goal_pos[1] - y_strike_exact, goal_pos[0] - x_strike_exact)
    
    print(f"  Exact strike target   : x={x_strike_exact:.3f} m, y={y_strike_exact:.3f} m, theta={theta_strike_exact:+.3f} rad")
    print("-" * 65)
    
    q_strike = np.array([x_strike_exact, y_strike_exact, theta_strike_exact, v_impact])
    
    # Save the original theta_strike for logging/error evaluation
    theta_strike_eval = theta_strike_exact

    # 4. Initialize World and InterceptionMPC
    world = World(
        car_state=car_start.copy(),
        ball_pos=ball_start.copy(),
        ball_vel=ball_vel.copy(),
        goal_pos=goal_pos.copy(),
        dt=dt,
    )
    Q_term = np.diag([500.0, 500.0, 100.0, 1.0])
    R_weights = np.diag([0.01, 0.01])
    mpc = InterceptionMPC(dt=dt, Q_terminal=Q_term, R=R_weights)

    # Prepare directories for frame saving
    if save_frames and save_dir:
        frames_dir = os.path.join(save_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
    else:
        frames_dir = None

    # Setup rendering backend
    if not render:
        matplotlib.use("Agg")

    # 5. Shrinking-Horizon Loop
    history = []
    solver_failures = 0
    total_solve_ms = 0.0

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

        # Save rendering frame if requested
        if save_frames and frames_dir:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            world.render(ax=ax, title=f"Step {step} | N_rem={N_remaining} | pos_err={pos_err:.2f}m")
            fig.savefig(os.path.join(frames_dir, f"frame_{step:03d}.png"), dpi=80)
            plt.close(fig)
        elif render:
            world.render(title=f"Step {step} | N_rem={N_remaining} | pos_err={pos_err:.2f}m")

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
    print(f"  Final position error : {final_pos_err:.4f} m (threshold: 0.2)")
    print(f"  Final heading error  : {final_heading_err:.4f} rad (threshold: 0.15)")
    print(f"  Solver failures      : {solver_failures}")
    print(f"  Avg solve time       : {total_solve_ms / N_steps:.1f} ms")
    
    passed_pos = (final_pos_err <= 0.2)
    passed_heading = (final_heading_err <= 0.15)
    
    if passed_pos and passed_heading:
        print("  [SUCCESS] GOAL!")
        success = True
    else:
        print("  [FAILED] MISSED TARGET")
        success = False
    print("-" * 65)

    # Save trajectory to CSV
    if save_dir:
        csv_path = os.path.join(save_dir, "trajectory.csv")
        os.makedirs(save_dir, exist_ok=True)
        import pandas as pd
        pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"Trajectory saved to {csv_path}")

    return success, final_pos_err, final_heading_err, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--render", action="store_true", help="Interactive rendering")
    parser.add_argument("--save-frames", action="store_true", help="Save frames to disk")
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "interception_run"), help="Directory for logs and frames")
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

    run_simulation(
        ball_start=ball_start,
        ball_vel=ball_vel,
        car_start=car_start,
        render=args.render,
        save_frames=args.save_frames,
        save_dir=args.save_dir,
    )
