#!/usr/bin/env bash
#
# run_pipeline.sh — Full StrikeNet + NMPC pipeline.
#
# Run this AFTER activating the conda env yourself, e.g.:
#     conda activate striker
#     bash run_pipeline.sh
#
# Pipeline (8 steps): data -> train both variants -> sanity -> integration
#   test (hybrid/legacy) -> reports + fallback analysis -> scalability benchmark
#   (both variants) -> 3-way x 2-variant comparison harness -> cross-method
#   cost/benefit ("worth it?") analysis.
#
# Notes:
#   * Steps 1 (data) and 2 (train) are MANDATORY if the dataset labels or
#     network schema changed; they overwrite strike_dataset.npy and both
#     strategy_net_{legacy,structured}.pth. Use --skip-data / --skip-train to
#     reuse existing ones.
#   * The scalability benchmark (step 6) runs LIGHT by default (50 scenes x 10
#     reps). Pass --full-bench for the full sweep.
#   * Designed for Git Bash / WSL on Windows or any Linux shell.
#
# Usage:
#   bash run_pipeline.sh                 # full pipeline (regen data + retrain)
#   bash run_pipeline.sh --skip-data     # reuse existing dataset
#   bash run_pipeline.sh --skip-data --skip-train   # only run evaluation + analysis
#   bash run_pipeline.sh --no-video      # faster integration test (no mp4s)
#   bash run_pipeline.sh --full-bench    # full scalability sweep (slower)
#   NUM_SAMPLES=50000 bash run_pipeline.sh   # override dataset size

set -euo pipefail

# ----------------------------------------------------------------------------
# Config / flags
# ----------------------------------------------------------------------------
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
SKIP_DATA=0
SKIP_TRAIN=0
NO_VIDEO=""
FULL_BENCH=0

for arg in "$@"; do
  case "$arg" in
    --skip-data)  SKIP_DATA=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --no-video)   NO_VIDEO="--no-video" ;;
    --full-bench) FULL_BENCH=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $arg" ; exit 1 ;;
  esac
done

# Move to the directory of this script (project root)
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
step() { echo; echo "====================================================================="; echo " $1"; echo "====================================================================="; }

# Sanity: warn if conda env not active
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  echo "WARNING: no conda env detected. Activate it first (e.g. 'conda activate striker')."
fi

# ----------------------------------------------------------------------------
# STEP 1 — Generate dataset (also writes data/dataset/dataset_stats.json)
# ----------------------------------------------------------------------------
if [ "$SKIP_DATA" -eq 0 ]; then
  step "STEP 1/8 — Generating dataset ($NUM_SAMPLES samples)"
  python -m src.data_generator --num_samples "$NUM_SAMPLES"
else
  step "STEP 1/8 — SKIPPED (reusing existing dataset)"
fi

# ----------------------------------------------------------------------------
# STEP 2 — Train StrikeNet
# ----------------------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  step "STEP 2/8 — Training StrikeNet (both variants)"
  python -m src.network --variant both
else
  step "STEP 2/8 — SKIPPED (reusing existing model)"
fi

# ----------------------------------------------------------------------------
# STEP 3 — Quick network sanity check
# ----------------------------------------------------------------------------
step "STEP 3/8 — Network sanity check"
python scripts/test_network.py

# ----------------------------------------------------------------------------
# STEP 4 — Integration test (100 seeds, 100-199) — the main evaluation
# ----------------------------------------------------------------------------
step "STEP 4/8 — Integration test (100 seeds, hybrid mode)"
python scripts/test_main.py $NO_VIDEO

# ----------------------------------------------------------------------------
# STEP 5 — Reports + diagnostics + fallback analysis
# -------------------------------------------------------------------------
step "STEP 5/8 — Plots, diagnostics, and fallback analysis"
python scripts/generate_plots.py
python scripts/analyze_results.py
python -m scripts.analyze_fallback

# ----------------------------------------------------------------------------
# STEP 6 — Scalability benchmark (light by default; --full-bench for full sweep)
# ----------------------------------------------------------------------------
if [ "$FULL_BENCH" -eq 1 ]; then
  step "STEP 6/8 — Scalability benchmark (both variants, full sweep)"
  python -m scripts.benchmark_scalability --model-variant both
else
  step "STEP 6/8 — Scalability benchmark (both variants, light: 50 scenes x 10 reps)"
  python -m scripts.benchmark_scalability --model-variant both --n-scenes 50 --repeats 10
fi

# ----------------------------------------------------------------------------
# STEP 7 — 3-Way Comparison Harness
# ----------------------------------------------------------------------------
step "STEP 7/8 — 3-Way Comparison Harness"
python scripts/compare_modes.py

# ----------------------------------------------------------------------------
# STEP 8 — Cross-method cost/benefit ("worth it?") analysis
# ----------------------------------------------------------------------------
step "STEP 8/8 — Cross-method cost/benefit analysis"
python -m scripts.analyze_comparison

step "PIPELINE SUMMARY"
python scripts/summarize_pipeline.py --save || echo "WARNING: summarize_pipeline.py failed (partial run?)"

step "PIPELINE COMPLETE"
echo "Results:"
echo "  - Integration batch : data/tests/integration/<latest>/"
echo "  - Diagnostics/plots : data/reports/plots/integration/<latest>/"
echo "  - Fallback analysis : data/reports/plots/integration/<latest>/fallback_analysis.png"
echo "  - Scalability       : data/reports/plots/global/scalability_curve.png"
echo "  - Dataset stats     : data/dataset/dataset_stats.json"
echo "  - Comparison batches: data/tests/comparison/<latest>/"
echo "  - Comparison report : data/reports/plots/comparison/<latest>/ (comparison.csv, comparison_summary.md, comparison_bars.png)"
echo "  - Worth-it analysis : data/reports/plots/comparison/<latest>/ (worth_it_summary.md, pareto_success_vs_latency.png, success_heatmap.png, ...)"
echo "  - Pipeline summary  : data/reports/pipeline_summaries/<latest>_pipeline_summary.md"
