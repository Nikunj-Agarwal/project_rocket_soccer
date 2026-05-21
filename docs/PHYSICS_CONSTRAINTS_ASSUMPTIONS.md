# Physics, Constraints, and Assumptions — Phase 5

## 🏟️ Field & Goal Geometry

| Parameter | Value | Source |
| :--- | :--- | :--- |
| **Field Width** (`field_w`) | 10.0 m | `DEFAULT_FIELD_W` in `ball_physics.py` |
| **Field Height** (`field_h`) | 6.0 m | `DEFAULT_FIELD_H` in `ball_physics.py` |
| **Goal Position** | Line segment at $x = 10.0$ for $y \in [2.0, 4.0]$ | `Goal` class in `src/goal.py` |
| **Playable Region** | Closed rectangle $[0, W] \times [0, H]$ | Inclusive boundaries |

* **Scoring Mouth**: Unlike other parts of the right wall ($x = 10.0$), the goal segment does not bounce the ball back; instead, the ball is allowed to pass through, triggering `scored = True` when its trajectory segment crosses the goal line.

---

## ⚽ Ball Physics & Collision Models

Implemented in `src/ball_physics.py` (`propagate_ball_step`, `propagate_ball_for_time`, `compute_strike_velocity`).

### 1. Ball-Wall Bounce Model
* Between bounces, velocity is constant (no drag, gravity, or spin).
* On wall impact, the normal velocity component flips and is scaled by the coefficient of restitution $e = 0.85$.
* Position is clamped to the wall boundary upon impact.

### 2. Car-Ball Bumper Collision Model
The car's front bumper is treated as a flat plane with normal vector $\mathbf{n} = [\cos(\theta_{car}), \sin(\theta_{car})]^T$. Let $\mathbf{v}_{car} = v_{car} \mathbf{n}$.
* **Relative Velocity**: $\mathbf{v}_{rel} = \mathbf{v}_{ball} - \mathbf{v}_{car}$
* **Normal Component**: $v_{rel,n} = \mathbf{v}_{rel} \cdot \mathbf{n}$
* **Impact Reflection**: If $v_{rel,n} < 0$ (approaching collision):
  $$\mathbf{v}_{ball}^{post} = \mathbf{v}_{ball} - (1 + e_{strike}) v_{rel,n} \mathbf{n}$$
  where $e_{strike} = 0.8$.
* **Momentum Push**: If $v_{rel,n} \ge 0$ but the car is moving faster in the normal direction ($v_{car} > \mathbf{v}_{ball} \cdot \mathbf{n}$), the car pushes the ball forward:
  $$\mathbf{v}_{ball}^{post} = \mathbf{v}_{ball} + (v_{car} - \mathbf{v}_{ball} \cdot \mathbf{n})(1 + e_{strike}) \mathbf{n}$$

---

## 🚗 Car Kinematic & NMPC Constraints

`InterceptionMPC` in `src/nmpc_solver.py` solves a multiple-shooting nonlinear programming problem using CasADi and IPOPT.

| Constraint | Value | Notes |
| :--- | :--- | :--- |
| **Step size** $\Delta t$ | 0.1 s | Must match simulator step size |
| **Wheelbase** $L$ | 0.3 m | |
| **Velocity bounds** | $[0.0, 2.0]$ m/s | |
| **Steering bounds** $\delta$ | $[-\pi/4, \pi/4]$ rad | Symmetric limits |
| **Acceleration bounds** $a$ | $[-2.0, 2.0]$ m/s² | Symmetric limits |
| **Impact Speed** $v_{impact}$ | 1.0 m/s | Set as target in `main.py` |
| **Terminal Weights** $\mathbf{Q}$ | diag(500, 500, 100, 1) | Penalizes position, heading, and speed |
| **Running Cost** $\mathbf{R}$ | diag(0.01, 0.01) | Penalizes acceleration and steering effort |

### Target Offset Position
To prevent early contact triggers from disrupting orientation alignment, the target for the NMPC solver is offset backward from the ball center:
$$x_{target} = x_{strike\_exact} - d_{offset} \cos(\theta_{strike\_exact})$$
$$y_{target} = y_{strike\_exact} - d_{offset} \sin(\theta_{strike\_exact})$$
where $d_{offset} = 0.32$ m. This ensures the car's heading error is minimal at the exact moment of collision (`CONTACT_RADIUS = 0.35` m).

### Post-Strike Active Braking
Post-strike, a braking controller decelerates the vehicle at the maximum rate ($-2.0$ m/s²) until it stops, ensuring it stays on the field:
$$a_{brake} = \text{clip}\left(-\frac{v_{car}}{\Delta t}, -2.0, 0.0\right)$$

---

## 📊 Dataset & Generation Constraints

The generator (`src/data_generator.py`) generates reachable scoring scenarios:
* **Ball Sampling**: Position $x_b \in [2.0, 8.0]$, $y_b \in [0.0, 6.0]$, speed $v_b \in [0.5, 2.0]$ m/s.
* **Car Sampling**: Position $x_c \in [0.0, 4.0]$, $y_c \in [0.0, 6.0]$, heading $\theta_c \in [-\pi, \pi]$, speed $v_c = 0.0$.
* **Scoring Heading Verification**: If a state is reachable at $T \in [0.5, 5.0]$ s, the generator sweeps 36 angular candidates $\theta_{strike} \in [-\pi, \pi]$ to find a heading that causes the ball to deflect into the goal. If no Working heading exists, the sample is rejected.

---

## 🏁 Integration Test Success Criteria

Evaluated in `scripts/test_main.py` over 10 distinct seeds:

| Metric | Threshold |
| :--- | :--- |
| **Scored Goal Rate** | $\ge 60\%$ (at least 6 successes) |
| **Avg Strike Position Error** | $\le 0.35$ m |
| **Avg Strike Heading Error** | $\le 0.25$ rad |
| **On-field Constraint** | Both car and ball must remain in $[0, W] \times [0, H]$ at termination |
