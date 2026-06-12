# Data layout and report plots

Path helpers live in `src/data_layout.py`. This document explains how to find **which run** produced **which plot**.

## Directory tree

```text
data/
├── dataset/strike_dataset.npy
├── training/training_log.csv
├── reports/plots/
│   ├── README.md                         # index of batches
│   ├── global/                           # not batch-specific
│   │   ├── training_curve.png
│   │   └── strikenet_sample_errors.png
│   └── integration/{batch_id}/           # one folder per test_main.py run
│       ├── README.md                     # table: seed → source run path
│       ├── integration_summary.png
│       └── seed_{N}/
│           ├── trajectory.png
│           └── errors.png
├── tests/integration/{batch_id}/
│   ├── batch.log
│   └── seed_{N}/
│       ├── trajectory.csv      # source data for plots
│       ├── simulation.mp4
│       └── metadata.json
├── tests/static/{batch_id}/...
└── runs/manual/manual_seed{N}_{timestamp}/...
```

## Batch ID

Integration batches are named by **timestamp** when `test_main.py` starts:

```text
YYYYMMDD_HHMMSS
```

Example: `20260521_022824` → run started 2026-05-21 02:28:24.

**Always cite this ID** when comparing plots, videos, or logs.

## Linking plots ↔ raw runs

| You have | Look here |
|----------|-----------|
| Plot `.../plots/integration/20260521_022824/seed_10/trajectory.png` | Raw run `data/tests/integration/20260521_022824/seed_10/` |
| Video for that seed | `.../seed_10/simulation.mp4` |
| Numeric series for errors plot | `.../seed_10/trajectory.csv` columns `pos_err`, `heading_err` |
| Pass/fail for one seed | `.../seed_10/metadata.json` → `"success"` |
| Whole batch summary | `.../integration/20260521_022824/batch.log` |

Open `data/reports/plots/integration/{batch_id}/README.md` — it lists every seed and the exact source folder.

## Generating plots

```powershell
conda activate striker
cd D:\SNU\Semester_6\motion_planning\project_retry

# Latest integration batch
python scripts/generate_plots.py

# Specific batch (if you have multiple)
python scripts/generate_plots.py --batch 20260521_022824
```

**Global** figures (training curve, StrikeNet sample errors) always go to `plots/global/`.

**Per-batch** figures only appear under `plots/integration/{batch_id}/`.

## Old flat layout (removed)

Do **not** look for PNGs directly under `data/reports/plots/` (except `README.md`).

| Old path (deleted) | New path |
|--------------------|----------|
| `plots/training_curve.png` | `plots/global/training_curve.png` |
| `plots/strikenet_sample_errors.png` | `plots/global/strikenet_sample_errors.png` |
| `plots/trajectory_seed_10.png` | `plots/integration/{batch}/seed_10/trajectory.png` |
| `plots/errors_seed_7.png` | `plots/integration/{batch}/seed_7/errors.png` |

Re-run after integration tests: `python scripts/generate_plots.py --batch <batch_id>`.

## Git ignore

`data/.gitignore` ignores generated `*.png`, `*.csv`, `*.mp4`, etc. Structure is kept via `.gitkeep` files; artifacts stay local.

## trajectory.csv columns

| Column | Meaning |
|--------|---------|
| `step` | MPC step index |
| `N_rem` | Remaining horizon length |
| `car_x`, `car_y`, `car_theta`, `car_v` | Car state after step |
| `ball_x`, `ball_y` | Ball position after step |
| `u_acc`, `u_steer` | Applied control |
| `pos_err` | Distance car–ball |
| `heading_err` | |car θ − strike θ| (wrapped) |
| `solve_ms` | NMPC solve time |

## metadata.json (integration / manual runs)

Typical fields:

- `success`, `final_pos_err_m`, `final_heading_err_rad`
- `N_steps`, `T_final_s`, `ball_restitution`, `field_size_m`
- `strike_target`: `[x, y, θ]` used by NMPC (bounce-correct)
- `seed`, `ball_start`, `ball_vel`, `car_start` (when from tests/manual)

StrikeNet raw `[T, x, y, θ]` predictions are **not** stored in metadata today (only console output).
