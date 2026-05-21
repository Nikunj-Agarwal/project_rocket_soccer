"""
data_generator.py — Phase 2 / 3.6

Generates geometric ground-truth training examples using brute-force reachability checks.
Ball future positions use the same inelastic wall-bounce model as the simulator.
"""

import os
import argparse
import numpy as np

from src.goal import Goal
from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
    propagate_ball_step,
    propagate_ball_for_time,
    compute_strike_velocity,
)


def generate_data(
    num_samples: int,
    output_path: str,
    field_w: float = DEFAULT_FIELD_W,
    field_h: float = DEFAULT_FIELD_H,
    ball_dt: float = DEFAULT_BALL_DT,
    ball_restitution: float = DEFAULT_BALL_RESTITUTION,
):
    print(f"Generating {num_samples} samples using brute-force reachability (bounce & score-aware)...")
    print(f"  Field: {field_w}x{field_h} m | restitution={ball_restitution} | ball_dt={ball_dt}")

    # Constants
    goal = Goal() # default: x=10.0, y in [2.0, 4.0]
    
    # We will accumulate valid samples here
    valid_samples = []
    
    attempts = 0
    while len(valid_samples) < num_samples:
        attempts += 1
        
        # Randomize ball state
        b_x = np.random.uniform(2.0, 8.0)
        b_y = np.random.uniform(0.0, 6.0)
        phi = np.random.uniform(0.0, 2 * np.pi)
        v_b = np.random.uniform(0.5, 2.0)
        b_vx = v_b * np.cos(phi)
        b_vy = v_b * np.sin(phi)
        
        # Randomize car state
        c_x = np.random.uniform(0.0, 4.0)
        c_y = np.random.uniform(0.0, 6.0)
        c_theta = np.random.uniform(-np.pi, np.pi)
        
        # Sweep future time T from 0.5 to 5.0 s
        T_feasible = None
        theta_strike_feasible = None
        x_strike_feasible = None
        y_strike_feasible = None
        
        ball_pos = np.array([b_x, b_y], dtype=float)
        ball_vel_arr = np.array([b_vx, b_vy], dtype=float)
        t_current = 0.0

        for T in np.arange(0.5, 5.05, 0.05):
            # Incremental bounce integration (same model as simulator, faster than restart each T)
            while t_current < T - 1e-12:
                step_dt = min(ball_dt, T - t_current)
                ball_pos, ball_vel_arr = propagate_ball_step(
                    ball_pos,
                    ball_vel_arr,
                    step_dt,
                    field_w=field_w,
                    field_h=field_h,
                    restitution=ball_restitution,
                )
                t_current += step_dt

            b_T_x, b_T_y = float(ball_pos[0]), float(ball_pos[1])

            # Strike must remain on the playable field
            if not (0.0 <= b_T_x <= field_w and 0.0 <= b_T_y <= field_h):
                continue

            # Straight-line distance
            d = np.sqrt((b_T_x - c_x)**2 + (b_T_y - c_y)**2)
            
            # Sweep theta_strike candidates to find one that scores AND is reachable
            # Sweep 36 angles: from -pi to pi with step 2*pi / 36 (10 degrees)
            theta_candidates = np.linspace(-np.pi, np.pi, 36, endpoint=False)
            
            found_feasible_for_T = False
            for theta_cand in theta_candidates:
                # 1. Check if this strike orientation can redirect the ball into the goal.
                # Assume impact speed is 1.0 m/s (matching v_impact in main.py)
                v_post = compute_strike_velocity(ball_vel_arr, v_car=1.0, theta_car=theta_cand, e_strike=0.8)
                
                # Propagate ball for up to 5.0 seconds
                final_pos, _ = propagate_ball_for_time(
                    ball_pos,
                    v_post,
                    total_time=5.0,
                    dt=ball_dt,
                    field_w=field_w,
                    field_h=field_h,
                    restitution=ball_restitution,
                    goal=goal
                )
                
                # If crossed the goal line within the goal mouth
                if final_pos[0] >= goal.x - 1e-9 and goal.y_min <= final_pos[1] <= goal.y_max:
                    # 2. Check if this specific orientation is reachable in time T.
                    # Line-of-sight angle from car to future ball position
                    theta_los = np.arctan2(b_T_y - c_y, b_T_x - c_x)
                    
                    # Heading change from initial car heading to face target
                    dtheta_start = np.abs(np.arctan2(np.sin(theta_los - c_theta), np.cos(theta_los - c_theta)))
                    
                    # Heading change from line-of-sight to the desired strike heading
                    dtheta_end = np.abs(np.arctan2(np.sin(theta_cand - theta_los), np.cos(theta_cand - theta_los)))
                    
                    # Minimum turning radius R = L/tan(delta_max) = 0.3 m. Use R = 0.35 m buffer
                    R_turn = 0.35
                    d_effective = d + R_turn * (dtheta_start + dtheta_end)
                    
                    # Reachability check (accounting for acceleration from rest: a_max=2.0, v_max=2.0)
                    if T <= 1.0:
                        d_max = T ** 2
                    else:
                        d_max = 2.0 * T - 1.0
                        
                    if d_effective <= d_max:
                        T_feasible = T
                        x_strike_feasible = b_T_x
                        y_strike_feasible = b_T_y
                        theta_strike_feasible = theta_cand
                        found_feasible_for_T = True
                        break
            
            if found_feasible_for_T:
                break # Found minimum feasible T with a scoring heading
                
        if T_feasible is not None:
            # Store [inputs..., outputs...]
            sample = [
                b_x, b_y, b_vx, b_vy, c_x, c_y, c_theta,  # Inputs
                T_feasible, x_strike_feasible, y_strike_feasible, theta_strike_feasible # Outputs
            ]
            valid_samples.append(sample)
            
            if len(valid_samples) % 5000 == 0:
                print(f"  Generated {len(valid_samples):5d} / {num_samples} samples... (attempts so far: {attempts})")

    # Convert to numpy array and save
    dataset = np.array(valid_samples, dtype=np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, dataset)
    print(f"Data generation complete.")
    print(f"Total attempts: {attempts} for {num_samples} valid samples (acceptance rate: {num_samples/attempts*100:.1f}%)")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100000)
    args = parser.parse_args()
    
    # Path relative to script location
    from src.data_layout import STRIKE_DATASET

    generate_data(args.num_samples, str(STRIKE_DATASET))
