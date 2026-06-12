# Robot Soccer Striker — Motion Planning

Closed-loop striker: **StrikeNet** (legacy or structured variant) predicts when/how to intercept; **NMPC** drives the car; **World** simulates the kinematic bicycle and bouncing ball with elastic collisions.

**Documentation:** [docs/README.md](docs/README.md) — architecture, physics, pipeline, data layout, research paper (previous try vs current), and phase updates.

## Setup

```powershell
conda activate striker
pip install -r requirements.txt
pip install imageio imageio-ffmpeg
```

### GPU PyTorch (RTX 40-series / CUDA 12.6 driver)

```powershell
conda activate striker
pip uninstall torch torchvision torchaudio -y
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Full pipeline (recommended)

```powershell
conda activate striker
cd D:\SNU\Semester_6\motion_planning\project_retry

# All 8 eval steps: data → train both variants → sanity → integration → reports → benchmark → comparison → cost/benefit
.\run_pipeline.ps1 -NoVideo
```

Steps: see [docs/PIPELINE_LOGIC.md](docs/PIPELINE_LOGIC.md). Results paths use `{LATEST_INTEGRATION_BATCH}` and `{LATEST_COMPARISON_RUN}` — see [docs/README.md](docs/README.md) placeholder note.

Re-eval only (skip data + train):

```powershell
.\run_pipeline.ps1 -SkipData -SkipTrain -NoVideo
```

## Manual step-by-step

```powershell
# 1) Dataset
python -m src.data_generator --num_samples 100000

# 2) Train both StrikeNet variants
python -m src.network --variant both

# 3) Sanity check
python scripts/test_network.py

# 4) Integration test (default: hybrid + legacy, 100 seeds)
python scripts/test_main.py --no-video

# 5) Reports
python scripts/generate_plots.py
python scripts/analyze_results.py
python -m scripts.analyze_fallback    # hybrid batches only

# 6) Scalability (both variants; light by default in pipeline)
python -m scripts.benchmark_scalability --model-variant both --n-scenes 50 --repeats 10

# 7) Five-config comparison
python scripts/compare_modes.py

# 8) Cost/benefit analysis (Pareto, worth-it summary)
python -m scripts.analyze_comparison

# Pipeline summary (optional)
python scripts/summarize_pipeline.py --save
```

### Planner modes and variants

```powershell
python src/main.py --seed 10 --planner-mode analytic
python src/main.py --seed 10 --planner-mode neural --model-variant legacy
python src/main.py --seed 10 --planner-mode hybrid --model-variant structured --save-video

python scripts/test_main.py --planner-mode neural --model-variant structured --no-video
```

## Project layout

| Path | Role |
|------|------|
| `src/main.py` | `decide_strike_target()`; modes: `analytic`, `neural`, `hybrid` |
| `src/network.py` | StrikeNet variants: `legacy` (5-out), `structured` (3-out) |
| `src/planner.py` | `analytic_strike_plan`, `max_reach_distance` |
| `src/data_layout.py` | Paths for models, integration/comparison batches, plots |
| `scripts/compare_modes.py` | 5-config comparison harness |
| `scripts/analyze_comparison.py` | Cross-config cost/benefit after comparison |
| `scripts/summarize_pipeline.py` | Consolidated pipeline summary |
| `scripts/test_main.py` | Integration batches + `summary.json` |
| `models/strategy_net_{legacy,structured}.pth` | Trained weights |
| `data/` | Datasets, runs, tests, reports — [data/README.md](data/README.md) |
| `run_pipeline.ps1` / `run_pipeline.sh` | End-to-end 8-step eval pipeline |

## Phases

1. Simulator + NMPC  
2. Dataset + StrikeNet  
3. Real-time interception  
3.6. Ball bounce pipeline  
4. Report plots  
5. Strike & Score  
5+. Dual-model variants + 3-way planner comparison  

Details: `data/phase_archives/`
