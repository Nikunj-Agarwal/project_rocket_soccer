<!--
DOC PLACEHOLDERS — see docs/README.md. Batch folders: {LATEST_INTEGRATION_BATCH}, {LATEST_COMPARISON_RUN}.
-->

# Data directory

All runtime artifacts live under `data/`. Path helpers: [`src/data_layout.py`](../src/data_layout.py).  
**Logic & assumptions:** [`docs/README.md`](../docs/README.md).

## Layout

```
data/
├── dataset/
│   ├── strike_dataset.npy
│   └── dataset_stats.json
├── training/
│   ├── training_log_legacy.csv
│   └── training_log_structured.csv
├── reports/
│   ├── benchmarks/scalability.csv
│   ├── pipeline_summaries/              # summarize_pipeline.py --save
│   └── plots/
│       ├── global/
│       ├── integration/{batch_id}/
│       └── comparison/{run_id}/       # compare_modes + analyze_comparison
│           ├── comparison.csv, comparison_bars.png
│           ├── worth_it_summary.md
│           ├── pareto_success_vs_latency.png
│           └── config_summary.csv, win_matrix.csv, ...
├── tests/
│   ├── integration/{batch_id}/        # test_main.py
│   │   ├── batch.log
│   │   ├── summary.json
│   │   └── seed_{N}/metadata.json, trajectory.csv, simulation.mp4
│   └── comparison/{run_id}/           # compare_modes.py
│       ├── analytic/
│       ├── neural_legacy/
│       ├── neural_structured/
│       ├── hybrid_legacy/
│       └── hybrid_structured/
├── runs/manual/                       # src/main.py --save-video
└── phase_archives/
```

## Commands (striker conda env)

```powershell
conda activate striker

python -m src.data_generator --num_samples 100000
python -m src.network --variant both

python scripts/test_network.py
python scripts/test_main.py --planner-mode hybrid --model-variant legacy --no-video
python scripts/compare_modes.py
python -m scripts.analyze_comparison

python scripts/generate_plots.py
python scripts/generate_plots.py --batch {LATEST_INTEGRATION_BATCH}

python scripts/summarize_pipeline.py --save

python src/main.py --seed 10 --planner-mode hybrid --model-variant structured --save-video
```

Or run everything: `.\run_pipeline.ps1 -NoVideo` from project root (8 steps + auto summary).

## Latency fields in `metadata.json`

| Field | Role |
| :--- | :--- |
| `decision_latency_ms` | Deployed path (full `decide_strike_target()` timing) |
| `fallback_sweep_ms` | Hybrid fallback 36-heading sweep only |
| `strikenet_infer_ms` | Diagnostic micro-benchmark (scalability) |

See [docs/DATA_AND_REPORTS.md](../docs/DATA_AND_REPORTS.md).

## Video output

Saves **`simulation.mp4`** (requires `imageio` + `imageio-ffmpeg`); falls back to **`simulation.gif`**.

## Obsolete paths

- `data/interception_run/`, `data/integration_test_results/`, `data/static_target_test/`
- `data/plots/` → use `data/reports/plots/`
- Root-level `data/strike_dataset.npy` → use `data/dataset/`
- Single `strategy_net.pth` only → use `models/strategy_net_legacy.pth` and `strategy_net_structured.pth`
