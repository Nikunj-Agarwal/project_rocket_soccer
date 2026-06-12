#!/usr/bin/env bash
#
# run_pipeline.sh — Full StrikeNet + NMPC pipeline.
#
# Run this AFTER activating the conda env yourself, e.g.:
#     conda activate striker
#     bash run_pipeline.sh
#
# Notes:
#   * Steps 1 (data) and 2 (train) are MANDATORY if the dataset labels or
#     network schema changed; they overwrite strike_dataset.npy and
#     strategy_net.pth. Use --skip-data / --skip-train to reuse existing ones.
#   * Designed for Git Bash / WSL on Windows or any Linux shell.
#
# Usage:
#   bash run_pipeline.sh                 # full pipeline (regen data + retrain)
#   bash run_pipeline.sh --skip-data     # reuse existing dataset
#   bash run_pipeline.sh --skip-data --skip-train   # only run evaluation + analysis
#   bash run_pipeline.sh --no-video      # faster integration test (no mp4s)
#   NUM_SAMPLES=50000 bash run_pipeline.sh   # override dataset size

set -euo pipefail

# ----------------------------------------------------------------------------
# Config / flags
# ----------------------------------------------------------------------------
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
SKIP_DATA=0
SKIP_TRAIN=0
NO_VIDEO=""

for arg in "$@"; do
  case "$arg" in
    --skip-data)  SKIP_DATA=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --no-video)   NO_VIDEO="--no-video" ;;
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
  step "STEP 1/6 — Generating dataset ($NUM_SAMPLES samples)"
  python -m src.data_generator --num_samples "$NUM_SAMPLES"
else
  step "STEP 1/6 — SKIPPED (reusing existing dataset)"
fi

# ----------------------------------------------------------------------------
# STEP 2 — Train StrikeNet
# ----------------------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  step "STEP 2/6 — Training StrikeNet"
  python -m src.network
else
  step "STEP 2/6 — SKIPPED (reusing existing model)"
fi

# ----------------------------------------------------------------------------
# STEP 3 — Quick network sanity check
# ----------------------------------------------------------------------------
step "STEP 3/6 — Network sanity check"
python scripts/test_network.py

# ----------------------------------------------------------------------------
# STEP 4 — Integration test (50 seeds) — the main evaluation
# ----------------------------------------------------------------------------
step "STEP 4/6 — Integration test (50 seeds)"
python scripts/test_main.py $NO_VIDEO

# ----------------------------------------------------------------------------
# STEP 5 — Reports + diagnostics + fallback analysis
# ----------------------------------------------------------------------------
step "STEP 5/6 — Plots, diagnostics, and fallback analysis"
python scripts/generate_plots.py
python scripts/analyze_results.py
python -m scripts.analyze_fallback

# ----------------------------------------------------------------------------
# STEP 6 — Scalability benchmark
# ----------------------------------------------------------------------------
step "STEP 6/6 — Scalability benchmark"
python -m scripts.benchmark_scalability

step "PIPELINE COMPLETE"
echo "Results:"
echo "  - Integration batch : data/tests/integration/<latest>/"
echo "  - Diagnostics/plots : data/reports/plots/integration/<latest>/"
echo "  - Fallback analysis : data/reports/plots/integration/<latest>/fallback_analysis.png"
echo "  - Scalability       : data/reports/plots/global/scalability_curve.png"
echo "  - Dataset stats     : data/dataset/dataset_stats.json"
