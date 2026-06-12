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
from src.planner import analytic_strike_plan

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
    offset_dist: float = 0.32,
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

    # 3. Query StrikeNet — with warm-up + repeated timing for latency logging.
    # We force CPU for the timing comparison so the measurement is comparable to
    # the analytic search (also single-threaded NumPy CPU).
    TIMING_REPEATS = 30
    TIMING_DEVICE = "cpu"
    timing_model = StrikeNet()
    timing_model.load_state_dict(model.state_dict())
    timing_model = timing_model.to(TIMING_DEVICE)
    timing_model.eval()
    timing_inputs = inputs.copy()

    # Warm-up (discarded): absorbs lazy init and JIT costs
    for _ in range(3):
        timing_model.predict(timing_inputs)

    _infer_times = []
    for _ in range(TIMING_REPEATS):
        _t0 = time.perf_counter()
        timing_model.predict(timing_inputs)
        _infer_times.append((time.perf_counter() - _t0) * 1e3)  # ms
    strikenet_infer_ms = float(np.median(_infer_times))

    # Run the actual (potentially GPU) predict for control
    preds = model.predict(inputs)  # Returns [T_strike, x_strike, y_strike, theta_strike]
    T_strike = float(preds[0])
    x_strike = float(preds[1])
    y_strike = float(preds[2])
    theta_strike = float(preds[3])

    # Timed analytic planner on same scene — pure measurement, does NOT drive control.
    # Warm-up
    _car_state_arr = np.array([car_start[0], car_start[1], car_start[2]])
    for _ in range(3):
        analytic_strike_plan(
            ball_start.copy(), ball_vel.copy(), _car_state_arr, goal,
            field_w=field_size[0], field_h=field_size[1],
            ball_dt=dt, ball_restitution=ball_restitution,
        )
    _analytic_times = []
    for _ in range(TIMING_REPEATS):
        _t0 = time.perf_counter()
        analytic_strike_plan(
            ball_start.copy(), ball_vel.copy(), _car_state_arr, goal,
            field_w=field_size[0], field_h=field_size[1],
            ball_dt=dt, ball_restitution=ball_restitution,
        )
        _analytic_times.append((time.perf_counter() - _t0) * 1e3)  # ms
    analytic_strategy_ms = float(np.median(_analytic_times))
    speedup_factor = analytic_strategy_ms / max(strikenet_infer_ms, 1e-6)

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
    print(f"  StrikeNet infer (CPU) : {strikenet_infer_ms:.3f} ms  (median/{TIMING_REPEATS} reps)")
    print(f"  Analytic search       : {analytic_strategy_ms:.1f} ms  (median/{TIMING_REPEATS} reps)")
    print(f"  Speedup factor        : {speedup_factor:.1f}x")
    print("-" * 65)

    # Define strike state for NMPC from the network's prediction.
    # The analytic ball rollout is kept only as a diagnostic and as a
    # sanity-check fallback if the predicted heading cannot score.
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

    # Network-predicted strike target, clipped to the playable field
    x_strike_net = float(np.clip(x_strike, 0.0, field_w))
    y_strike_net = float(np.clip(y_strike, 0.0, field_h))
    net_vs_analytic = np.hypot(x_strike_net - strike_pos[0], y_strike_net - strike_pos[1])

    def _scores(pos_xy: np.ndarray, theta: float) -> bool:
        """Does striking at pos_xy with heading theta send the ball into the goal?"""
        v_post = compute_strike_velocity(strike_vel, v_car=v_impact, theta_car=theta, e_strike=0.8)
        final_pos, _ = propagate_ball_for_time(
            pos_xy,
            v_post,
            total_time=5.0,
            dt=dt,
            field_w=field_w,
            field_h=field_h,
            restitution=ball_restitution,
            goal=goal,
        )
        return final_pos[0] >= goal.x - 1e-9 and goal.y_min <= final_pos[1] <= goal.y_max

    if _scores(np.array([x_strike_net, y_strike_net]), theta_strike):
        # Trust the network end-to-end
        target_source = "network"
        x_strike_tgt = x_strike_net
        y_strike_tgt = y_strike_net
        theta_strike_tgt = theta_strike
    else:
        # Fallback: analytic strike point + scoring-heading sweep
        target_source = "fallback"
        x_strike_tgt = float(strike_pos[0])
        y_strike_tgt = float(strike_pos[1])

        theta_los_goal = np.arctan2(goal.center[1] - y_strike_tgt, goal.center[0] - x_strike_tgt)
        scoring_thetas = [
            theta_cand
            for theta_cand in np.linspace(-np.pi, np.pi, 36, endpoint=False)
            if _scores(strike_pos, theta_cand)
        ]
        if scoring_thetas:
            # Pick the scoring heading closest to line-of-sight to the goal
            # (same canonical choice as the dataset labels)
            theta_strike_tgt = min(
                scoring_thetas,
                key=lambda th: abs(np.arctan2(np.sin(th - theta_los_goal), np.cos(th - theta_los_goal))),
            )
        else:
            theta_strike_tgt = theta_los_goal

    print(f"  Ball bounce           : restitution={ball_restitution}, field={field_w}x{field_h} m")
    print(f"  Net vs analytic ball  : {net_vs_analytic:.3f} m apart at T_final={T_final:.2f} s")
    print(f"  Target source         : {target_source}")
    print(f"  Strike target         : x={x_strike_tgt:.3f} m, y={y_strike_tgt:.3f} m, theta={theta_strike_tgt:+.3f} rad")
    print("-" * 65)
    
    x_target = x_strike_tgt - offset_dist * np.cos(theta_strike_tgt)
    y_target = y_strike_tgt - offset_dist * np.sin(theta_strike_tgt)
    q_strike = np.array([x_target, y_target, theta_strike_tgt, v_impact])
    
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
    Q_term = np.diag([3000.0, 3000.0, 300.0, 1.0])
    R_weights = np.diag([0.005, 0.005])
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
    strike_step = None
    ball_at_strike = None  # ball position at the exact moment of contact

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
            np.sin(world.car_state[2] - theta_strike_tgt),
            np.cos(world.car_state[2] - theta_strike_tgt)
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
            world.render(title=title)

        if world.ball_struck:
            strike_step = step
            ball_at_strike = world.ball_pos.copy()
            print(f"Collision/Strike detected at step {step}!")
            break

    # 6. Phase 2: Post-Strike Coasting & Ball Flight
    POST_STRIKE_STEPS = 80 # up to 8 seconds; loop breaks early once scored,
                           # so this only extends episodes where the ball is
                           # still travelling toward the goal (e.g. late/slow strikes)
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
            "u_acc": a_brake,
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
            world.render(title=title)

    # Final evaluation
    final_car = world.car_state
    final_ball = world.ball_pos
    final_pos_err = np.linalg.norm(final_car[:2] - final_ball)
    
    # Wrap heading error
    final_heading_err = abs(np.arctan2(
        np.sin(final_car[2] - theta_strike_tgt),
        np.cos(final_car[2] - theta_strike_tgt)
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
    
    # A goal only counts when the car actually struck the ball.
    # world.scored is kept as the raw physics flag; success is the research metric.
    success = bool(world.scored and world.ball_struck)
    if success:
        print("  [SUCCESS] GOAL scored (with strike)!")
    elif world.scored and not world.ball_struck:
        print("  [UNSTRUCK GOAL] Ball entered goal without car contact — not counted.")
    else:
        print("  [FAILED] MISSED GOAL")
    print("-" * 65)

    # Issue 2: interception prediction accuracy metrics
    # strike_point_pred_err_m: distance between the predicted strike target and
    # where the ball actually was when contact occurred.
    # strike_time_err_s: how many seconds early/late the strike happened vs the
    # predicted horizon N_steps.
    if ball_at_strike is not None and strike_step is not None:
        strike_point_pred_err_m = float(np.linalg.norm(
            np.array([x_strike_tgt, y_strike_tgt]) - ball_at_strike
        ))
        strike_time_err_s = float(abs(strike_step - N_steps) * dt)
    else:
        # No contact occurred; use large sentinel so summaries show the failure clearly
        strike_point_pred_err_m = float("nan")
        strike_time_err_s = float("nan")

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
            # success = scored AND ball was struck (ungated goal no longer counts)
            "success": success,
            "scored": bool(world.scored),
            "ball_struck": bool(world.ball_struck),
            # --- Interception prediction accuracy (Issue 2 metrics) ---
            # strike_point_pred_err_m: ||predicted_target - ball_at_contact||
            # Near-0 for the analytic fallback; nonzero for the network path.
            "strike_point_pred_err_m": strike_point_pred_err_m,
            # strike_time_err_s: |actual_strike_step - predicted_N_steps| * dt
            "strike_time_err_s": strike_time_err_s,
            # ball position at the actual moment of contact (null if no contact)
            "ball_at_strike": ball_at_strike.tolist() if ball_at_strike is not None else None,
            # --- Legacy closest-approach errors (kept as diagnostic only) ---
            # "final_pos_err_m" alias retained so old analysis scripts don't break.
            "contact_pos_err_m": float(final_pos_err),
            "final_pos_err_m": float(final_pos_err),
            "final_heading_err_rad": float(final_heading_err),
            # --- Run parameters ---
            "solver_failures": solver_failures,
            "N_steps": N_steps,
            "T_final_s": float(T_final),
            "ball_restitution": ball_restitution,
            "field_size_m": list(field_size),
            "strike_target": [x_strike_tgt, y_strike_tgt, theta_strike_tgt],
            "target_source": target_source,
            "net_vs_analytic_pos_m": float(net_vs_analytic),
            "strike_step": strike_step,
            # --- Scalability / latency fields ---
            "strikenet_infer_ms": strikenet_infer_ms,
            "analytic_strategy_ms": analytic_strategy_ms,
            "speedup_factor": speedup_factor,
            "timing_device": TIMING_DEVICE,
            "timing_repeats": TIMING_REPEATS,
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
