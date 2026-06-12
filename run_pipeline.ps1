<#
.SYNOPSIS
    Full StrikeNet + NMPC pipeline for PowerShell on Windows.

.DESCRIPTION
    Run AFTER activating the conda env:
        conda activate striker
        cd D:\SNU\Semester_6\motion_planning\project_retry
        .\run_pipeline.ps1

    Pipeline (8 steps): data -> train both variants -> sanity -> integration
    test (hybrid/legacy) -> reports + fallback analysis -> scalability benchmark
    (both variants) -> 3-way x 2-variant comparison harness -> cross-method
    cost/benefit ("worth it?") analysis.

    Flags:
        -SkipData    Reuse existing strike_dataset.npy (skip Step 1)
        -SkipTrain   Reuse existing strategy_net_{legacy,structured}.pth (skip Step 2)
        -NoVideo     Skip MP4 rendering in integration test (faster)
        -FullBench   Run the FULL scalability benchmark (default is light:
                     --n-scenes 50 --repeats 10, which is much faster)

    Example:
        .\run_pipeline.ps1 -SkipData -SkipTrain -NoVideo
#>

param(
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$NoVideo,
    [switch]$FullBench
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to project root (where this script lives)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

function Step-Header($msg) {
    Write-Host ""
    Write-Host "=====================================================================" -ForegroundColor Cyan
    Write-Host " $msg" -ForegroundColor Cyan
    Write-Host "=====================================================================" -ForegroundColor Cyan
}

function Invoke-Step($desc, [scriptblock]$cmd) {
    Step-Header $desc
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FATAL] Step failed with exit code $LASTEXITCODE. Stopping." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Warn if conda env not active
if (-not $env:CONDA_DEFAULT_ENV) {
    Write-Warning "No conda env detected. Activate it first:  conda activate striker"
}

# -------------------------------------------------------------------------
# STEP 1 - Generate dataset
# -------------------------------------------------------------------------
if (-not $SkipData) {
    Invoke-Step "STEP 1/8 - Generating dataset (100000 samples)" {
        python -m src.data_generator --num_samples 100000
    }
} else {
    Step-Header "STEP 1/8 - SKIPPED (reusing existing dataset)"
}

# -------------------------------------------------------------------------
# STEP 2 - Train StrikeNet
# -------------------------------------------------------------------------
if (-not $SkipTrain) {
    Invoke-Step "STEP 2/8 - Training StrikeNet (both variants)" {
        python -m src.network --variant both
    }
} else {
    Step-Header "STEP 2/8 - SKIPPED (reusing existing model)"
}

# -------------------------------------------------------------------------
# STEP 3 - Network sanity check
# -------------------------------------------------------------------------
Invoke-Step "STEP 3/8 - Network sanity check" {
    python scripts/test_network.py
}

# -------------------------------------------------------------------------
# STEP 4 - Integration test (100 seeds)
# -------------------------------------------------------------------------
$testArgs = @()
if ($NoVideo) { $testArgs += "--no-video" }

Invoke-Step "STEP 4/8 - Integration test (100 seeds, hybrid mode)" {
    python scripts/test_main.py @testArgs
}

# -------------------------------------------------------------------------
# STEP 5 - Reports + fallback analysis
# -------------------------------------------------------------------------
Invoke-Step "STEP 5/8 - Plots, diagnostics, and fallback analysis" {
    python scripts/generate_plots.py
    python scripts/analyze_results.py
    python -m scripts.analyze_fallback
}

# -------------------------------------------------------------------------
# STEP 6 - Scalability benchmark (light by default; -FullBench for the full sweep)
# -------------------------------------------------------------------------
$benchArgs = @("--model-variant", "both")
if (-not $FullBench) { $benchArgs += "--n-scenes", "50", "--repeats", "10" }
$benchLabel = if ($FullBench) { "full sweep" } else { "light: 50 scenes x 10 reps" }

Invoke-Step "STEP 6/8 - Scalability benchmark (both variants, $benchLabel)" {
    python -m scripts.benchmark_scalability @benchArgs
}

# -------------------------------------------------------------------------
# STEP 7 - 3-Way Comparison Harness
# -------------------------------------------------------------------------
Invoke-Step "STEP 7/8 - 3-Way Comparison Harness" {
    python scripts/compare_modes.py
}

# -------------------------------------------------------------------------
# STEP 8 - Cross-method cost/benefit ("worth it?") analysis
# -------------------------------------------------------------------------
Invoke-Step "STEP 8/8 - Cross-method cost/benefit analysis" {
    python -m scripts.analyze_comparison
}

# -------------------------------------------------------------------------
# Consolidated summary (non-fatal if something is still missing)
# -------------------------------------------------------------------------
Step-Header "PIPELINE SUMMARY"
python scripts/summarize_pipeline.py --save
if ($LASTEXITCODE -ne 0) {
    Write-Warning "summarize_pipeline.py exited with code $LASTEXITCODE (partial run?)"
}

# -------------------------------------------------------------------------
Step-Header "PIPELINE COMPLETE"
Write-Host "Results:"
Write-Host "  Integration batch : data\tests\integration\<latest>\"
Write-Host "  Reports / plots   : data\reports\plots\integration\<latest>\"
Write-Host "  Fallback analysis : data\reports\plots\integration\<latest>\fallback_analysis.png"
Write-Host "  Scalability curve : data\reports\plots\global\scalability_curve.png"
Write-Host "  Dataset stats     : data\dataset\dataset_stats.json"
Write-Host "  Comparison batches: data\tests\comparison\<latest>\"
Write-Host "  Comparison report : data\reports\plots\comparison\<latest>\ (comparison.csv, comparison_summary.md, comparison_bars.png)"
Write-Host "  Worth-it analysis : data\reports\plots\comparison\<latest>\ (worth_it_summary.md, pareto_success_vs_latency.png, success_heatmap.png, ...)"
Write-Host "  Pipeline summary  : data\reports\pipeline_summaries\<latest>_pipeline_summary.md"
