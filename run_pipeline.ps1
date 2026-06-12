<#
.SYNOPSIS
    Full StrikeNet + NMPC pipeline for PowerShell on Windows.

.DESCRIPTION
    Run AFTER activating the conda env:
        conda activate striker
        cd D:\SNU\Semester_6\motion_planning\project_retry
        .\run_pipeline.ps1

    Flags:
        -SkipData    Reuse existing strike_dataset.npy (skip Step 1)
        -SkipTrain   Reuse existing strategy_net.pth   (skip Step 2)
        -NoVideo     Skip MP4 rendering in integration test (faster)
        -LightBench  Use --n-scenes 50 --repeats 10 for scalability benchmark

    Example:
        .\run_pipeline.ps1 -SkipData -SkipTrain -NoVideo -LightBench
#>

param(
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$NoVideo,
    [switch]$LightBench
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
    Invoke-Step "STEP 1/6 - Generating dataset (100000 samples)" {
        python -m src.data_generator --num_samples 100000
    }
} else {
    Step-Header "STEP 1/6 - SKIPPED (reusing existing dataset)"
}

# -------------------------------------------------------------------------
# STEP 2 - Train StrikeNet
# -------------------------------------------------------------------------
if (-not $SkipTrain) {
    Invoke-Step "STEP 2/6 - Training StrikeNet" {
        python -m src.network
    }
} else {
    Step-Header "STEP 2/6 - SKIPPED (reusing existing model)"
}

# -------------------------------------------------------------------------
# STEP 3 - Network sanity check
# -------------------------------------------------------------------------
Invoke-Step "STEP 3/6 - Network sanity check" {
    python scripts/test_network.py
}

# -------------------------------------------------------------------------
# STEP 4 - Integration test (100 seeds)
# -------------------------------------------------------------------------
$testArgs = @()
if ($NoVideo) { $testArgs += "--no-video" }

Invoke-Step "STEP 4/6 - Integration test (100 seeds)" {
    python scripts/test_main.py @testArgs
}

# -------------------------------------------------------------------------
# STEP 5 - Reports + fallback analysis
# -------------------------------------------------------------------------
Invoke-Step "STEP 5/6 - Plots, diagnostics, and fallback analysis" {
    python scripts/generate_plots.py
    python scripts/analyze_results.py
    python -m scripts.analyze_fallback
}

# -------------------------------------------------------------------------
# STEP 6 - Scalability benchmark
# -------------------------------------------------------------------------
$benchArgs = @()
if ($LightBench) { $benchArgs += "--n-scenes", "50", "--repeats", "10" }

Invoke-Step "STEP 6/6 - Scalability benchmark" {
    python -m scripts.benchmark_scalability @benchArgs
}

# -------------------------------------------------------------------------
Step-Header "PIPELINE COMPLETE"
Write-Host "Results:"
Write-Host "  Integration batch : data\tests\integration\<latest>\"
Write-Host "  Reports / plots   : data\reports\plots\integration\<latest>\"
Write-Host "  Fallback analysis : data\reports\plots\integration\<latest>\fallback_analysis.png"
Write-Host "  Scalability curve : data\reports\plots\global\scalability_curve.png"
Write-Host "  Dataset stats     : data\dataset\dataset_stats.json"
