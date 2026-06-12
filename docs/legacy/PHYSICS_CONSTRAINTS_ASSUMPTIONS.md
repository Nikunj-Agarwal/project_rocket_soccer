# Physics, constraints, and assumptions

## Field

| Parameter | Value | Source |
|-----------|-------|--------|
| Width `field_w` | 10.0 m | `DEFAULT_FIELD_W` in `ball_physics.py` |
| Height `field_h` | 6.0 m | `DEFAULT_FIELD_H` |
| Goal position | (9.5, 3.0) | `main.py`, `data_generator.py`, plots |
| Playable region | Closed rectangle `[0, W] × [0, H]` | Walls are inclusive boundaries |

**Assumption:** The field is flat; no obstacles except the four walls.

## Ball model

Implemented in `src/ball_physics.py` (`propagate_ball_step`, `propagate_ball_for_time`).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `dt` | 0.1 s | Simulation / integration step (matches `World.dt` and MPC) |
| `restitution` | **0.85** | Inelastic reflection: normal velocity component flips and scales by `e` |
| `max_bounces` | 8 per step | Safety cap inside one `dt` sub-step |

**Behavior:**

- Velocity is **constant** between wall contacts (no gravity, drag, or spin).
- Walls are **axis-aligned** at `x = 0`, `x = W`, `y = 0`, `y = H`.
- On impact, the component normal to the struck wall is multiplied by `-restitution`.
- Position is clamped to the wall before reflecting velocity.

**Assumptions:**

- Ball–wall collisions are instantaneous.
- Multiple bounces within one `dt` are resolved in sub-segments (piecewise linear motion).
- Training labels, `main.py` strike target, and `World.step` all use this **same** integrator — no train/sim mismatch for bounce geometry.

**Not modeled:** ball–car collision, ball–goal interaction, curvature of trajectories beyond bounces.

## Car model (NMPC)

`InterceptionMPC` in `src/nmpc_solver.py` uses a **kinematic bicycle** with RK4 discretization and IPOPT (`casadi.Opti`).

| Constraint | Value | Notes |
|------------|-------|-------|
| `dt` | 0.1 s | Must match `World.dt` |
| Wheelbase `L` | 0.3 m | |
| `v_min`, `v_max` | 0, 2.0 m/s | |
| `delta_max` | π/4 rad | Steering symmetric |
| `a_max` | 2.0 m/s² | Acceleration symmetric |
| Terminal state | `[x, y, θ, v_impact]` | Default `v_impact = 1.0` m/s at strike |
| Terminal weights `Q` | diag(500, 500, 100, 1) in `main.py` | Strong position/orientation tracking |
| Running cost `R` | diag(0.01, 0.01) on `[a, δ]` | |

**Assumptions:**

- No slip; unicycle/bicycle kinematics are exact.
- Obstacles are only implicit via planning to a strike point (no obstacle constraints in MPC).
- One NMPC solve per step; only `u_0` is applied (receding horizon).

## Data generation constraints

`src/data_generator.py` samples until `num_samples` valid rows are collected.

**Sampling ranges (must match `main.py` / `test_main.py`):**

| Variable | Range |
|----------|--------|
| `ball_x` | [2, 8] m |
| `ball_y` | [0, 6] m |
| Ball speed | [0.5, 2.0] m/s, random direction |
| `car_x` | [0, 4] m |
| `car_y` | [0, 6] m |
| `car_theta` | [−π, π] |
| Car initial speed | 0 |

**Feasibility filter (per sample):**

- Sweep `T` from 0.5 s to 5.0 s in 0.05 s steps.
- Ball future position at `T` from **bounce-aware** integration.
- **Reject** strike points outside the field (`0 ≤ x ≤ W`, `0 ≤ y ≤ H`).
- **Reachability:** effective path length ≤ `d_max(T)` with bi-arc turning margin (`R_turn = 0.35` m) and trapezoidal speed limit (`v_max = 2`, `a_max = 2`).
- Label = **first** (minimum) feasible `T` and corresponding `(x, y, θ_strike)` facing the goal.

**Assumptions:**

- Ground truth is geometric, not from NMPC rollouts.
- If no `T` in the sweep is feasible, the sample is discarded (acceptance rate ~90% for 100k target).
- Strike heading always points from strike point toward the goal.

## Integration test success criteria

`scripts/test_main.py`:

| Criterion | Threshold |
|-----------|-----------|
| Per-run success | Final position error ≤ **0.2 m** AND heading error ≤ **0.15 rad** |
| Batch pass | ≥ **80%** of seeds succeed **and** mean position error ≤ 0.2 m **and** mean heading error ≤ 0.15 rad |
| On-field | Final ball and car positions inside `[0, W] × [0, H]` (logged; failure if OOB) |

**Note:** A single bad seed (e.g. seed 7) can fail the batch on **mean** error even when 9/10 runs succeed.

## Rendering vs physics

`World.render` may clip the **view** for display; physics integration does **not** clamp ball/car position to the field except at walls for the ball. Final states should remain on-field when interception works; OOB finals indicate a bug or failed intercept.

## Known limitations

- StrikeNet `(x, y)` outputs are not directly used as the MPC target; only `T_strike` (via `N_steps`) and bounce physics define the terminal pose.
- NMPC does not model future ball motion after the current step (target is fixed at horizon start).
- Manual runs do not log StrikeNet raw predictions in `metadata.json` (only `strike_target` after bounce).
