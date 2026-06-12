# Phase 5 System Upgrades — Strike & Score Overhaul

This document details the transition from **Phase 3.6 (Interception with wall bounces)** to **Phase 5 (Strike & Score)**. The system has evolved from a simple "reach-and-face" interceptor to a fully physical soccer striker that redirects a bouncing ball into a goal mouth and comes to a safe stop.

---

## 🛠️ Summary of Overhaul Features

| Feature | Phase 3.6 (Legacy) | Phase 5 (Current Upgraded System) | Rationale / Benefits |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Touch/intercept the ball center while facing the goal at $T$. | Redirect the ball into a **2m wide goal** ($x=10.0, y \in [2.0, 4.0]$). | Converts the project from simple interception to a scoring soccer striker. |
| **Physics Model** | Simple point overlap (no impact mechanics). | **2D Elastic Collision** (flat car bumper, $e_{strike} = 0.8$). | Simulates realistic momentum transfer between the car and the ball. |
| **Target Geometry** | Target ball center exactly: `q_strike = [x, y, θ, v]`. | Target shifted backward: `q_strike` offset by **0.32 m** behind ball. | Prevents early contact triggering before the car completes its terminal rotation. |
| **Post-Strike State** | Car continues coasting passively (`u = [0.0, 0.0]`). | Car **actively brakes** at maximum deceleration ($-2.0$ m/s²). | Prevents the car from exiting the field boundaries (keeping states on-pitch). |
| **Dataset Generation** | Simple reachability test at $T$. | 1D sweep over $\theta_{strike}$ to find a scoring deflection. | Guarantees that training labels represent actions that result in a goal. |
| **Solver Speed** | High IPOPT console log volume (~10 min/batch). | **Silenced solver output** (~2-3 sec/batch). | Removes console I/O bottleneck, speeding up development and validation. |

---

## 🧠 Key Design Decisions

### 1. NMPC Target Offset (`offset_dist = 0.32` m)
The simulator detects contact when the distance between the car center and the ball center is less than `CONTACT_RADIUS = 0.35` m. 
* **The Problem**: If NMPC targets the ball center directly, the car triggers collision *before* it has fully rotated to `theta_strike`. This resulted in a high strike heading error (~0.61 rad).
* **The Solution**: We shift the NMPC target position backward along the approach heading vector by `0.32` meters:
  $$\mathbf{p}_{target} = \mathbf{p}_{strike} - d_{offset} \cdot \begin{bmatrix} \cos(\theta_{strike}) \\ \sin(\theta_{strike}) \end{bmatrix}$$
  As the car smoothly approaches this offset target, it aligns its orientation perfectly. Contact is triggered at a distance of $0.35\text{ m}$ just before reaching the target, resulting in negligible heading error ($< 0.03\text{ rad}$).

### 2. Active Braking Control
Because the kinematic bicycle dynamics do not model ground friction, the car keeps moving at its impact speed forever under zero acceleration control. In post-strike propagation, this caused the car to drive off-field.
* **The Solution**: An active deceleration controller is executed in Phase 2:
  $$a_{brake} = \text{clip}\left(-\frac{v_{car}}{\Delta t}, -a_{max}, 0.0\right)$$
  This slows the car to a full stop in a few steps, keeping it safely within field limits.

### 3. Solver Silence Optimization
IPOPT print logging was bypassed by setting CasADi helper parameters (`print_time: False`, `verbose: False`) and IPOPT solver options (`print_level: 0`, `sb: "yes"`). This optimization resulted in a **120x speedup** in overall run times.

### 4. Pursuit-Based NMPC Warm-Start
* **The Problem**: In challenging initial configurations (e.g., the car pointing away from the target), starting with a physics-violating straight-line guess caused IPOPT to get stuck in local minima, resulting in acceleration chattering and eventual complete failure/stopping.
* **The Solution**: We generate a kinematically feasible initial guess by forward-simulating a proportional pursuit steering and acceleration controller starting from the current vehicle state using the symbolic RK4 dynamics. This satisfies the kinematics constraints perfectly and guides IPOPT directly to the global optimum, ensuring 100% convergence.

---

## 🔧 Post-Phase-5 Audit Overhaul (Network-Driven Targets)

A logical/mathematical audit found that the original Phase 5 online loop **discarded most of StrikeNet's output**: it kept only $T_{strike}$ (to set the horizon) and recomputed the strike position and heading analytically every run. The network was effectively imitating an oracle that was then thrown away. The following fixes make the learned policy actually drive the controller and address the modelling issues that made its outputs unreliable.

| Area | Before (Phase 5) | After (Audit Overhaul) | Rationale |
| :--- | :--- | :--- | :--- |
| **Target source** | Always analytic: ball rolled forward to $T$, heading swept at runtime. The net's $x, y, \theta$ were printed and discarded. | The network's $(x, y, \theta)$ **drives** the NMPC target. A scoring rollout validates the plan; the analytic point + heading sweep is used **only** as a fallback when the predicted plan cannot score. `target_source` is logged per episode. | Makes the ML meaningful — it is now the primary decision-maker, not a bypassed component. |
| **Label heading** | First scoring + reachable angle scanning from $-\pi$ (discontinuous, multimodal). | Among all scoring + reachable candidates at the minimum feasible $T$, the heading **closest to the goal line-of-sight** (deterministic, approximately continuous). | MSE regression averages multimodal targets into invalid in-between angles; a canonical label removes this. |
| **Output normalization** | Targets used raw; loss dominated by $T, x, y$ (scales 5–10× the $\sin/\cos$ heading terms). | Targets z-scored with train-split statistics (`output_mean`, `output_std`); loss computed in normalized space; `predict()` de-normalizes. | Lets the network actually learn heading instead of ignoring it. |
| **Collision model** | Unreachable "momentum push" branch in `compute_strike_velocity`. | Branch removed; returns ball velocity unchanged when not approaching. | Dead code; algebraically impossible to enter. |
| **Post-strike window** | 50 steps (5.0 s). | 80 steps (8.0 s), early-break on score. | Late/slow strikes were cut off mid-flight (e.g. seed 120) before the ball crossed the line. |
| **Plot/diagnostic fixes** | $\theta$ error unwrapped; stale goal marker at $(9.5, 3.0)$; wrong constants in `analyze_results.py`. | Wrapped $\theta$ error; goal drawn as true segment $x=10, y\in[2,4]$; corrected $\Delta t$, $a_{max}$, $\delta_{max}$, control-period constants. | Accurate reporting. |

> **Note:** After these changes the dataset and model schema changed, so `strike_dataset.npy` and `strategy_net.pth` must be regenerated (`python -m src.data_generator` then `python -m src.network`). The old checkpoint will not load (missing normalization buffers, by design).

---

## ✅ Evaluation Validity Fixes (Issues 1–3)

| Issue | Change | Where |
| :--- | :--- | :--- |
| **1 — Strike-gated success** | `success = scored AND ball_struck`. Raw `scored` and `ball_struck` still logged separately; unstruck goals print `[UNSTRUCK GOAL]` and do not count. | `src/main.py`, `scripts/test_main.py`, `scripts/analyze_fallback.py`, `scripts/analyze_results.py` |
| **2 — Headline metrics** | New metadata: `strike_point_pred_err_m`, `strike_time_err_s`, `ball_at_strike`. Closest-approach distance demoted to diagnostic (`contact_pos_err_m` / `final_pos_err_m` alias). Integration pass criterion is **60% success only** — the old $\le 0.35$ m contact threshold removed (tautological at `CONTACT_RADIUS`). | `src/main.py`, `scripts/test_main.py` |
| **3 — Shared planner module** | Analytic search refactored into `src/planner.py` (`analytic_strike_plan`, `max_reach_distance`). Car constants from NMPC; $R_{turn}$ default still **0.35 m** (legacy) — behaviour unchanged, no dataset regen needed. | `src/planner.py`, `src/data_generator.py`, `src/main.py` |

**Runtime architecture note:** Only one hybrid mode exists at runtime (network prediction with scoring-guard fallback). There is no selectable analytic-only or network-only mode; `target_source` in `metadata.json` logs which path drove each episode. The online fallback propagates the ball to the network's $T$ and sweeps headings inline — it does not call `analytic_strike_plan()` for control (that function is for offline labels and latency timing). Physics-informed prediction ($T+\theta$ only, derive $x,y$ from ball rollout) is **not implemented** — see [FUTURE_physics_informed_prediction.md](FUTURE_physics_informed_prediction.md).

---

## 📈 System Metrics & Pass Criteria

The integration test (`scripts/test_main.py`) runs **100** seeds by default (100–199). **Pass:** strike-gated success rate $\ge 60\%$.

**Headline accuracy** (reported in `batch.log` and `metadata.json`): mean/median `strike_point_pred_err_m`, `strike_time_err_s`.

**Diagnostic** (not pass gates): closest-approach `contact_pos_err_m`, `net_vs_analytic_pos_m`, network-vs-fallback breakdown via `scripts/analyze_fallback.py`.

*Example (50-seed batch `20260612_155705`):* 74% success (37/50); network 16/26 (61.5%), fallback 21/24 (87.5%). The earlier "88% / 44-50" figure reflected the old analytic-driven loop and is no longer representative.
