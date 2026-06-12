"""
compare_modes.py — Phase 5
Runs all 5 planner/model configs on shared seeds and produces a combined report.
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_layout import new_comparison_run, plots_comparison_dir, model_path_for_variant

CONFIGS = [
    ("analytic", "analytic", None),
    ("neural_legacy", "neural", "legacy"),
    ("neural_structured", "neural", "structured"),
    ("hybrid_legacy", "hybrid", "legacy"),
    ("hybrid_structured", "hybrid", "structured")
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 200)))
    # Videos are off by default (5 configs x 100 seeds is a lot of mp4s); pass
    # --video to re-enable per-seed recordings.
    parser.add_argument("--video", dest="no_video", action="store_false", default=True,
                        help="Record per-seed simulation videos (off by default for speed)")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    
    comp_dir = new_comparison_run(args.run_id)
    print(f"Comparison run directory: {comp_dir}")
    
    for name, mode, variant in CONFIGS:
        if variant:
            mpath = model_path_for_variant(variant)
            if not os.path.exists(mpath):
                print(f"WARNING: Model {mpath} missing. Train it first: `python -m src.network --variant {variant}`")
                print(f"Skipping config: {name}")
                continue
                
        print(f"\n=== Running {name} ===")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "test_main.py"),
            "--planner-mode", mode,
            "--batch-dir", os.path.join(comp_dir, name),
        ]
        if variant:
            cmd.extend(["--model-variant", variant])
        if args.no_video:
            cmd.append("--no-video")
        
        cmd.extend(["--seeds"] + [str(s) for s in args.seeds])
        
        subprocess.run(cmd, check=False)
        
    # Aggregate
    results = []
    for name, mode, variant in CONFIGS:
        sum_file = os.path.join(comp_dir, name, "summary.json")
        if os.path.exists(sum_file):
            with open(sum_file, "r") as f:
                d = json.load(f)
                d["config_name"] = name
                if d["fb_episodes"] is not None and d["n"] > 0:
                    d["fallback_share"] = d["fb_episodes"] / d["n"]
                else:
                    d["fallback_share"] = 0.0
                results.append(d)
                
    if not results:
        print("No results to aggregate.")
        return
        
    df = pd.DataFrame(results)
    
    # Save CSV
    plots_dir = plots_comparison_dir(comp_dir.name)
    csv_path = os.path.join(plots_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Save MD
    md_path = os.path.join(plots_dir, "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("# 3-Way Planner Comparison\n\n")
        f.write("## Configs Evaluated\n")
        for r in results:
            f.write(f"- **{r['config_name']}**: SR={r['success_rate']*100:.1f}%, ")
            f.write(f"Err={r['mean_pred_err_m']:.3f}m, Lat={r['mean_decision_latency_ms']:.1f}ms, ")
            f.write(f"FB Share={r['fallback_share']*100:.1f}%\n")
        
        f.write("\n## Observations\n")
        f.write("- **Neural vs Analytic Gap**: Note any drop in SR from analytic to pure neural.\n")
        f.write("- **Position Error Circularity**: `neural_structured` should have near 0 err (matches analytic), whereas `neural_legacy` predicts off-trajectory points.\n")
        f.write("- **Hybrid latency**: `decision_latency_ms` is wall-clock of the deployed planner path, including the 36-heading fallback sweep when it fires.\n")
        f.write("- **Scalability benchmark**: compares analytic search vs network infer (+rollout for structured); also reports hybrid fallback sweep cost separately.\n")

    # Plot Bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = df["config_name"].tolist()
    sr = df["success_rate"] * 100
    err = df["mean_pred_err_m"]
    lat = df["mean_decision_latency_ms"]
    
    axes[0].bar(names, sr, color='skyblue')
    axes[0].set_title("Success Rate (%)")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0, 100)
    
    axes[1].bar(names, err, color='salmon')
    axes[1].set_title("Mean Strike Pred Err (m)")
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(names, lat, color='lightgreen')
    axes[2].set_title("Mean Decision Latency (ms)")
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_yscale("log")
    
    plt.tight_layout()
    png_path = os.path.join(plots_dir, "comparison_bars.png")
    plt.savefig(png_path)
    print(f"\nComparison outputs saved to: {plots_dir}")
    
    # Print ascii table
    print("\n" + df[["config_name", "success_rate", "mean_pred_err_m", "mean_decision_latency_ms", "fallback_share"]].to_string(index=False))

if __name__ == "__main__":
    main()
