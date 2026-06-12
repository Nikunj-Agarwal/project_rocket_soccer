# Data Layout and Report Plots — Phase 5

Path helpers are defined in `src/data_layout.py`. This document describes the file layout, how to link evaluation plots back to their raw runs, and the database schema of outputs.

---

## 📁 Directory Structure

```text
project_root/
├── data/
│   ├── dataset/
│   │   └── strike_dataset.npy            # Generated training dataset
│   ├── training/
│   │   └── training_log.csv              # MSE metrics per training epoch
│   ├── reports/plots/
│   │   ├── README.md                     # Index of all generated test batches
│   │   ├── global/                       # Batch-independent training statistics
│   │   │   ├── training_curve.png
│   │   │   └── strikenet_sample_errors.png
│   │   └── integration/{batch_id}/       # Specific test batch directory
│   │       ├── README.md                 # Summary links of seeds
│   │       ├── integration_summary.png   # Performance chart for this batch
│   │       └── seed_{N}/
│   │           ├── trajectory.png        # Vehicle and ball paths
│   │           └── errors.png            # Tracking errors vs steps
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
| `pos_err` | float | Distance between the car center and the ball center. The minimum value during the `approach` phase is used as the strike position error. |
| `heading_err` | float | Orientation error relative to `theta_strike` (wrapped to $[-\pi, \pi]$). The value at the step of minimum `pos_err` is used as the strike heading error. |
| `solve_ms` | float | Execution time of NMPC solver in milliseconds. |

### 2. `metadata.json` Fields

* `success` (bool): `True` if `scored` is `True` and `on_field` is `True`.
* `final_pos_err_m` (float): Final position error at simulation termination.
* `final_heading_err_rad` (float): Final heading error at simulation termination.
* `solver_failures` (int): Number of steps NMPC failed to converge.
* `N_steps` (int): Interception horizon steps.
* `T_final_s` (float): Predicted interception time.
* `ball_restitution` (float): Wall coefficient of restitution.
* `field_size_m` (list of float): `[W, H]`.
* `strike_target` (list of float): `[x_target, y_target, theta_target]` — the chosen strike point and heading (from StrikeNet when `target_source == "network"`, else from the analytic fallback).
* `target_source` (str): `"network"` if StrikeNet's predicted strike point/heading passed the scoring rollout and was used directly, or `"fallback"` if the analytic strike point + heading sweep was substituted.
* `net_vs_analytic_pos_m` (float): Distance between StrikeNet's predicted strike position and the analytically propagated ball position at `T_final`. A diagnostic for how accurate the network's spatial prediction was (larger values correlate with network-driven misses).
* `scored` (bool): `True` if ball crossed the goal line.
* `ball_struck` (bool): `True` if collision occurred.
* `strike_step` (int): The step index where collision occurred.
