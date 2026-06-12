# Data Layout and Report Plots — Phase 5

Path helpers are defined in `src/data_layout.py`. This document describes the file layout, how to link evaluation plots back to their raw runs, and the database schema of outputs.

---

## 📁 Directory Structure

```text
project_root/
├── data/
│   ├── dataset/
│   │   ├── strike_dataset.npy            # Generated training dataset
│   │   └── dataset_stats.json            # Offline generation cost (wall-clock, per-sample search time)
│   ├── training/
│   │   └── training_log.csv              # MSE metrics per training epoch
│   ├── reports/
│   │   ├── benchmarks/
│   │   │   └── scalability.csv           # Analytic-vs-network latency sweep (benchmark_scalability.py)
│   │   └── plots/
│   │       ├── README.md                 # Index of all generated test batches
│   │       ├── global/                   # Batch-independent statistics
│   │       │   ├── training_curve.png
│   │       │   ├── strikenet_sample_errors.png
│   │       │   └── scalability_curve.png # Amortization: analytic search vs StrikeNet inference
│   │       └── integration/{batch_id}/   # Specific test batch directory
│   │           ├── README.md             # Summary links of seeds
│   │           ├── integration_summary.png
│   │           ├── decision_latency.png  # Per-seed StrikeNet vs analytic latency
│   │           ├── fallback_analysis.png # Network-vs-fallback breakdown (analyze_fallback.py)
│   │           ├── fallback_summary.md   # Network-vs-fallback report + per-seed CSV
│   │           ├── research_summary.md   # Full diagnostics report (analyze_results.py)
│   │           └── seed_{N}/
│   │               ├── trajectory.png    # Vehicle and ball paths
│   │               └── errors.png        # Tracking errors vs steps
│   └── tests/
│       └── integration/{batch_id}/       # Raw test run logs & data
│           ├── batch.log                 # Batch execution output
│           └── seed_{N}/
│               ├── trajectory.csv        # Numerical step series
│               ├── simulation.mp4        # Rendered trajectory video
│               └── metadata.json         # Episode configuration & metrics
├── docs/
│   ├── legacy/                           # Historic pre-Phase 5 documents
│   │   ├── DATA_AND_REPORTS.md
│   │   ├── PHYSICS_CONSTRAINTS_ASSUMPTIONS.md
│   │   ├── PIPELINE_LOGIC.md
│   │   ├── README.md
│   │   └── SYSTEM_OVERVIEW.md
│   ├── DATA_AND_REPORTS.md               # (This document)
│   ├── PHYSICS_CONSTRAINTS_ASSUMPTIONS.md
│   ├── PIPELINE_LOGIC.md
│   ├── SYSTEM_OVERVIEW.md
│   ├── UPDATE.md                         # Detailed update summary
│   └── README.md                         # Docs index
└── models/
    └── strategy_net.pth                  # Trained StrikeNet PyTorch weights
```

---

## 🏷️ Batch ID
Integration test batches are organized by timestamp:
```text
YYYYMMDD_HHMMSS
```
*Example*: `20260522_035708` matches the run initiated on May 22, 2026 at 03:57:08.

---

## 🔄 Linking Plots to Raw Runs

| To find the source of: | Look in this directory: |
| :--- | :--- |
| **Trajectory plot** `seed_10/trajectory.png` | `data/tests/integration/{batch_id}/seed_10/` |
| **Numeric series for errors** | `data/tests/integration/{batch_id}/seed_10/trajectory.csv` |
| **Evaluation video** | `data/tests/integration/{batch_id}/seed_10/simulation.mp4` |
| **Goal pass/fail metadata** | `data/tests/integration/{batch_id}/seed_10/metadata.json` |

---

## 📈 Generating Report Plots
To regenerate plots for the latest integration batch:
```powershell
python scripts/generate_plots.py
```
Or for a specific historical batch:
```powershell
python scripts/generate_plots.py --batch YYYYMMDD_HHMMSS
```

---

## 📝 Output Formats

### 1. `trajectory.csv` Column Fields

| Column | Data Type | Meaning |
| :--- | :--- | :--- |
| `step` | int | Simulation time step index. |
| `phase` | str | `"approach"` (NMPC phase) or `"post_strike"` (coasting/braking phase). |
| `N_rem` | int | Remaining NMPC horizon (0 in Phase 2). |
| `car_x`, `car_y`, `car_theta`, `car_v` | float | State variables of the kinematic bicycle model. |
| `ball_x`, `ball_y` | float | Coordinates of the ball center. |
| `u_acc` | float | Input acceleration command. |
| `u_steer` | float | Input steering angle command. |
| `pos_err` | float | Distance between the car center and the ball center. **Diagnostic only** — minimum during `approach` is closest-approach distance (tautologically $< 0.35$ m when contact occurs). |
| `heading_err` | float | Orientation error relative to `theta_strike` (wrapped to $[-\pi, \pi]$). **Diagnostic only** — value at minimum-`pos_err` step. |
| `solve_ms` | float | Execution time of NMPC solver in milliseconds. |

### 2. `metadata.json` Fields

* `success` (bool): **Strike-gated research metric** — `True` if `scored AND ball_struck`. Goals entered without car contact are excluded.
* `scored` (bool): Raw physics flag — ball segment crossed the goal mouth.
* `ball_struck` (bool): Raw physics flag — car–ball contact occurred (`dist < 0.35` m).
* `strike_point_pred_err_m` (float): Headline accuracy — $\|\text{strike\_target}_{xy} - \text{ball\_at\_contact}\|$. `NaN` if no contact.
* `strike_time_err_s` (float): Headline timing error — $|\text{strike\_step} - N_{steps}| \cdot \Delta t$. `NaN` if no contact.
* `ball_at_strike` (list or null): `[x, y]` ball position at contact; `null` if no strike.
* `contact_pos_err_m` (float): **Diagnostic** — closest-approach distance at end of approach (alias of legacy `final_pos_err_m`).
* `final_pos_err_m` (float): **Diagnostic** — same as `contact_pos_err_m` (kept for backward compatibility).
* `final_heading_err_rad` (float): **Diagnostic** — heading error at closest approach.
* `solver_failures` (int): Number of steps NMPC failed to converge.
* `N_steps` (int): Interception horizon steps.
* `T_final_s` (float): Predicted interception time.
* `ball_restitution` (float): Wall coefficient of restitution.
* `field_size_m` (list of float): `[W, H]`.
* `strike_target` (list of float): `[x_target, y_target, theta_target]` — the chosen strike point and heading (from StrikeNet when `target_source == "network"`, else from the analytic fallback).
* `target_source` (str): `"network"` if StrikeNet's predicted strike point/heading passed the scoring rollout and was used directly, or `"fallback"` if the analytic strike point + heading sweep was substituted.
* `net_vs_analytic_pos_m` (float): Distance between StrikeNet's predicted strike position and the analytically propagated ball position at `T_final`. Diagnostic for network spatial prediction quality.
* `strike_step` (int): The step index where collision occurred (`null`/absent if no strike).
* `strikenet_infer_ms` (float): Median StrikeNet inference latency on CPU over `timing_repeats` warm-up-discarded repetitions (online decision-layer cost).
* `analytic_strategy_ms` (float): Median latency of the equivalent analytic strike search on the same scene (timed for comparison only; does not drive control).
* `speedup_factor` (float): `analytic_strategy_ms / strikenet_infer_ms` — the per-decision amortization factor.
* `timing_device` (str): Device used for the latency comparison (e.g. `"cpu"`).
* `timing_repeats` (int): Number of timed repetitions used for the medians.

### 3. `dataset_stats.json` Fields (offline generation cost)

* `num_samples`, `total_attempts`, `acceptance_rate`: dataset size and search yield.
* `wall_clock_s`: real elapsed time of the parallel generation run.
* `total_cpu_search_s`: summed per-worker search time across all accepted samples.
* `num_workers`: parallel worker processes used.
* `mean_search_s_per_valid_sample`, `median_search_s_per_valid_sample`, `mean_search_s_per_attempt`: per-sample offline search cost (the expense amortized by StrikeNet).
* `generation_params`: field/physics and search-grid parameters (`n_angles`, `t_min`, `t_max`, `t_step`) for provenance.
