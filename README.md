# Robot Soccer Striker — Motion Planning

Closed-loop striker: **StrikeNet** (legacy or structured variant) predicts when/how to intercept; **NMPC** drives the car; **World** simulates the kinematic bicycle and bouncing ball with elastic collisions.

**Documentation:** Detailed architectural, literature, and physics documentation is stored locally in the gitignored `docs/` directory (e.g., `docs/SYSTEM_OVERVIEW.md`, `docs/LITERATURE_REVIEWS.md`).

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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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

Steps: See the local `docs/PIPELINE_LOGIC.md` for details on the evaluation steps. Results paths use `{LATEST_INTEGRATION_BATCH}` and `{LATEST_COMPARISON_RUN}` (as detailed in the local `docs/README.md`).

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
| `scripts/benchmark_scalability.py` | Generates scalability curves (NMPC vs Network) |
| `scripts/analyze_fallback.py` | Analyzes fallback events in hybrid mode |
| `scripts/generate_plots.py` | Generates trajectory and integration summary plots |
| `scripts/analyze_results.py` | Generates diagnostic plots for latency and tracking |
| `scripts/summarize_pipeline.py` | Consolidated pipeline summary markdown |
| `scripts/test_main.py` | Integration batches + `summary.json` |
| `models/strategy_net_{legacy,structured}.pth` | Trained weights |
| `data/` | Datasets, runs, tests, reports — [data/README.md](data/README.md) |
| `run_pipeline.ps1` / `run_pipeline.sh` | End-to-end 8-step eval pipeline |

## Standalone Utilities (Not in main pipeline)

| Path | Role |
|------|------|
| `scripts/sweep_offset.py` | Sweeps the heuristic target offset parameter |
| `scripts/test_static_target.py` | Evaluates NMPC solver against static targets |
| `scripts/verify_label_fidelity.py` | Diagnostic tool to verify dataset label sanity |
| `scripts/inspect_pdf.py` | Helper script to inspect and extract PDF pages |
| `scripts/test_torch.py` | Simple PyTorch and CUDA availability check |

## Phases

1. Simulator + NMPC  
2. Dataset + StrikeNet  
3. Real-time interception  
3.6. Ball bounce pipeline  
4. Report plots  
5. Strike & Score  
5+. Dual-model variants + 3-way planner comparison  

Details: local `data/phase_archives/` (gitignored)
