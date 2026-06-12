<!--
DOC PLACEHOLDERS — see docs/README.md for token definitions and how to resolve them.
-->

# Data Layout and Report Plots — Phase 5+

Path helpers: [`src/data_layout.py`](../src/data_layout.py). This document describes directories, metadata schema, and how to link plots back to raw runs.

---

## Directory structure

```text
project_root/
├── data/
│   ├── dataset/
│   │   ├── strike_dataset.npy
│   │   └── dataset_stats.json
│   ├── training/
│   │   ├── training_log_legacy.csv
│   │   └── training_log_structured.csv
│   ├── reports/
│   │   ├── benchmarks/
│   │   │   └── scalability.csv          # variant column when --model-variant both
│   │   └── plots/
│   │       ├── global/
│   │       │   ├── training_curve.png
│   │       │   ├── strikenet_sample_errors.png
│   │       │   └── scalability_curve.png
│   │       ├── integration/{batch_id}/
│   │       │   ├── integration_summary.png
│   │       │   ├── decision_latency.png
│   │       │   ├── fallback_analysis.png    # hybrid batches only
│   │       │   ├── fallback_summary.md
│   │       │   ├── research_summary.md
│   │       │   └── seed_{N}/
│   │       └── comparison/{run_id}/         # compare_modes.py + analyze_comparison.py
│   │           ├── comparison.csv
│   │           ├── comparison_summary.md
│   │           ├── comparison_bars.png
│   │           ├── worth_it_summary.md
│   │           ├── pareto_success_vs_latency.png
│   │           ├── latency_by_config.png
│   │           ├── success_heatmap.png
│   │           ├── hybrid_path_breakdown.png
│   │           ├── accuracy_vs_time.png
│   │           ├── config_summary.csv
│   │           ├── paired_seeds.csv
│   │           └── win_matrix.csv
│   └── tests/
│       ├── integration/{batch_id}/          # test_main.py default output
│       │   ├── batch.log
│       │   ├── summary.json                 # aggregate metrics per batch
│       │   └── seed_{N}/
│       │       ├── trajectory.csv
│       │       ├── simulation.mp4
│       │       └── metadata.json
│       └── comparison/{run_id}/             # compare_modes.py
│           ├── analytic/
│           ├── neural_legacy/
│           ├── neural_structured/
│           ├── hybrid_legacy/
│           └── hybrid_structured/           # each subfolder = full integration batch
├── models/
│   ├── strategy_net_legacy.pth
│   ├── strategy_net_structured.pth
│   └── strategy_net.pth                     # backward-compat alias (legacy)
└── docs/                                      # live docs (archives: legacy/, legacy_2/)
```

---

## Batch and run IDs

Integration and comparison runs use timestamp folders:

```text
YYYYMMDD_HHMMSS
```

Resolve latest integration batch:

```powershell
python -c "from src.data_layout import latest_integration_batch; b=latest_integration_batch(); print(b)"
```

Comparison runs: newest folder under `data/tests/comparison/`.

---

## Linking plots to raw runs

| Artifact | Source |
| :--- | :--- |
| `integration/{batch}/seed_10/trajectory.png` | `data/tests/integration/{batch}/seed_10/` |
| Numeric series | `.../trajectory.csv` |
| Video | `.../simulation.mp4` |
| Per-seed metrics | `.../metadata.json` |
| Batch aggregate | `data/tests/integration/{batch}/summary.json` |
| 5-way comparison table | `data/reports/plots/comparison/{run}/comparison.csv` |
| Cost/benefit narrative | `data/reports/plots/comparison/{run}/worth_it_summary.md` |
| Pipeline rollup | `data/reports/pipeline_summaries/{timestamp}_pipeline_summary.md` |

---

## Generating report plots

```powershell
# Latest integration batch
python scripts/generate_plots.py
python scripts/analyze_results.py
python -m scripts.analyze_fallback          # hybrid batches only

# Specific batch (replace placeholder)
python scripts/generate_plots.py --batch {LATEST_INTEGRATION_BATCH}
python -m scripts.analyze_fallback --batch {LATEST_INTEGRATION_BATCH}

# Comparison report (after compare_modes.py)
# → data/reports/plots/comparison/{LATEST_COMPARISON_RUN}/
python -m scripts.analyze_comparison   # step 8: worth_it_summary.md, Pareto plots, etc.
```

---

## Output formats

### `trajectory.csv`

| Column | Meaning |
| :--- | :--- |
| `step`, `phase`, `N_rem` | Time index, `approach` / `post_strike`, remaining horizon |
| `car_x`, `car_y`, `car_theta`, `car_v` | Bicycle state |
| `ball_x`, `ball_y` | Ball position |
| `u_acc`, `u_steer` | NMPC controls |
| `pos_err`, `heading_err` | **Diagnostic** closest-approach / heading vs strike $\theta$ |
| `solve_ms` | NMPC solve time |

### `metadata.json` (per seed)

**Success and accuracy**

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `success` | bool | Strike-gated: `scored AND ball_struck` |
| `scored`, `ball_struck` | bool | Raw physics flags |
| `strike_point_pred_err_m` | float | $\|\text{strike\_target}_{xy} - \text{ball\_at\_contact}\|$; NaN if no contact |
| `strike_time_err_s` | float | Timing error vs $N_{steps}$ |
| `ball_at_strike` | list/null | Ball position at contact |

**Planner configuration**

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `planner_mode` | str | `"analytic"`, `"neural"`, or `"hybrid"` |
| `model_variant` | str/null | `"legacy"` or `"structured"`; null for analytic |
| `target_source` | str | `"analytic"`, `"analytic_infeasible"`, `"network"`, or `"fallback"` |

**Strike target and diagnostics**

| Field | Meaning |
| :--- | :--- |
| `strike_target` | `[x, y, theta]` chosen strike point |
| `contact_pos_err_m` / `final_pos_err_m` | Closest-approach distance (diagnostic) |
| `final_heading_err_rad` | Heading error at closest approach |
| `net_target_vs_ball_traj_m` | Legacy network: $\|\text{predicted target} - \text{ball on trajectory at } T\|$ (`net_vs_analytic_pos_m` deprecated alias) |
| `N_steps`, `T_final_s`, `strike_step` | Horizon and contact timing |

**Latency**

| Field | Meaning |
| :--- | :--- |
| `decision_latency_ms` | **Deployed path** wall-clock of `decide_strike_target()` — includes inference, ball rollout, scoring checks, and hybrid fallback sweep when it fires |
| `fallback_sweep_ms` | Portion of `decision_latency_ms` spent in the 36-heading scoring sweep (hybrid fallback only; 0 otherwise) |
| `strikenet_infer_ms` | 30-rep median CPU inference micro-benchmark (diagnostic reference) |
| `rollout_ms` | 30-rep median ball rollout micro-benchmark (structured variant diagnostic) |
| `infer_plus_rollout_ms` | Sum of infer + rollout micro-benchmarks (diagnostic) |
| `analytic_strategy_ms` | 30-rep median analytic search on same scene (diagnostic reference) |
| `speedup_factor` | `analytic_strategy_ms / decision_latency_ms` |
| `timing_device`, `timing_repeats` | Timing protocol |

**Pass/fail:** integration tests gate on **strike-gated success rate** (≥60%), not latency. NMPC control period is 100 ms/step (`dt=0.1`); per-step solve times are in `trajectory.csv` → `solve_ms`.

### `summary.json` (per integration batch)

Written by `scripts/test_main.py`:

```json
{
  "planner_mode": "hybrid",
  "model_variant": "legacy",
  "n": 100,
  "successes": 0,
  "success_rate": 0.0,
  "mean_pred_err_m": 0.0,
  "mean_decision_latency_ms": 0.0,
  "net_episodes": 0,
  "fb_episodes": 0,
  ...
}
```

Fill numeric fields from the batch produced by your pipeline run (see placeholder note at top).

### `comparison.csv` (per comparison run)

One row per config (`config_name`, `success_rate`, `mean_pred_err_m`, `mean_decision_latency_ms`, `median_decision_latency_ms`, `fallback_share`, …). Generated by `scripts/compare_modes.py` from per-seed `metadata.json`.

### `analyze_comparison` outputs (same `{run_id}` folder)

| Artifact | Contents |
| :--- | :--- |
| `worth_it_summary.md` | Narrative: which configs win on success vs latency trade-offs |
| `config_summary.csv` | Aggregated stats per config |
| `paired_seeds.csv` | Per-seed paired outcomes across configs |
| `win_matrix.csv` | Head-to-head success wins between configs |
| `pareto_success_vs_latency.png` | Success–latency Pareto front |
| `latency_by_config.png` | Distribution of deployed `decision_latency_ms` |
| `success_heatmap.png` | Success rate grid |
| `hybrid_path_breakdown.png` | Network vs fallback latency and success |
| `accuracy_vs_time.png` | Prediction error vs decision time |

### `dataset_stats.json`

Offline generation cost: `wall_clock_s`, `mean_search_s_per_valid_sample`, `generation_params`, etc.
