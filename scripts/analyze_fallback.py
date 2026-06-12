"""
analyze_fallback.py — Network vs. analytic-fallback breakdown for an integration batch.

For each seed in a batch it reads metadata.json (target_source, success, latency)
and the trajectory.csv (strike-time position / heading error), then reports how
often the learned StrikeNet plan was trusted end-to-end versus when the analytic
fallback took over, and how each path performed.

Outputs (into data/reports/plots/integration/{batch_id}/):
  - fallback_analysis.png   (multi-panel comparison figure)
  - fallback_summary.csv    (per-seed raw table)
  - fallback_summary.md     (human-readable summary)

Usage:
  python -m scripts.analyze_fallback                # latest batch
  python -m scripts.analyze_fallback --batch <id>   # specific batch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_layout import (
    list_integration_batches,
    plots_batch_dir,
    iter_integration_seed_runs,
    TRAJECTORY_CSV,
    RUN_METADATA,
)


def load_fallback_data(batch_dir: Path) -> pd.DataFrame:
    """Build a per-seed dataframe with target_source, success, errors, latency."""
    rows = []
    for seed_str, run_dir in iter_integration_seed_runs(batch_dir):
        meta_path = run_dir / RUN_METADATA
        traj_path = run_dir / TRAJECTORY_CSV
        if not meta_path.is_file():
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # --- Strike-gated success (Issue 1 fix) ---
        scored = bool(meta.get("scored", meta.get("success", False)))
        ball_struck = bool(meta.get("ball_struck", False))
        success = scored and ball_struck

        # --- Headline metric: predicted target error (Issue 2) ---
        # Prefer logged value; fall back to contact-distance for old batches.
        strike_point_pred_err_m = meta.get("strike_point_pred_err_m", None)
        if strike_point_pred_err_m is None:
            strike_point_pred_err_m = meta.get("final_pos_err_m", np.nan)

        # --- Legacy contact distance (diagnostic, not a pass criterion) ---
        contact_dist = meta.get("final_pos_err_m", np.nan)
        contact_heading_err = meta.get("final_heading_err_rad", np.nan)
        if traj_path.is_file():
            df = pd.read_csv(traj_path)
            if not df.empty and "pos_err" in df.columns:
                ci = df["pos_err"].idxmin()
                contact_dist = float(df.loc[ci, "pos_err"])
                if "heading_err" in df.columns:
                    contact_heading_err = float(df.loc[ci, "heading_err"])

        rows.append({
            "seed": int(seed_str),
            "target_source": meta.get("target_source", "unknown"),
            "success": success,
            "scored": scored,
            "ball_struck": ball_struck,
            "strike_point_pred_err_m": strike_point_pred_err_m,
            "strike_pos_err": contact_dist,          # kept for backward-compat plots
            "strike_heading_err": contact_heading_err,
            "net_vs_analytic_pos_m": meta.get("net_vs_analytic_pos_m", np.nan),
            "strikenet_infer_ms": meta.get("strikenet_infer_ms", np.nan),
            "decision_latency_ms": meta.get("decision_latency_ms", np.nan),
            "fallback_sweep_ms": meta.get("fallback_sweep_ms", np.nan),
            "analytic_strategy_ms": meta.get("analytic_strategy_ms", np.nan),
            "speedup_factor": meta.get("speedup_factor", np.nan),
        })

    if not rows:
        print("Error: no metadata found in batch.")
        sys.exit(1)

    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _source_stats(df: pd.DataFrame) -> dict:
    """Aggregate success rate and errors per target_source."""
    out = {}
    for src in ("network", "fallback"):
        sub = df[df["target_source"] == src]
        n = len(sub)
        # Use the new headline metric where available; fall back to contact dist
        pred_col = "strike_point_pred_err_m" if "strike_point_pred_err_m" in sub.columns else "strike_pos_err"
        out[src] = {
            "n": n,
            "share": n / len(df) if len(df) else 0.0,
            "scored": int(sub["success"].sum()),
            "success_rate": (sub["success"].mean() if n else 0.0),
            "mean_pos_err": (sub[pred_col].mean() if n else np.nan),
            "mean_heading_err": (sub["strike_pos_err"].mean() if n else np.nan),  # contact dist
        }
    return out


def plot_fallback_analysis(df: pd.DataFrame, stats: dict, output_dir: Path, batch_id: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"StrikeNet (network) vs. Analytic Fallback — Batch {batch_id}",
                 fontsize=14, fontweight="bold")

    sources = ["network", "fallback"]
    colors = ["#3b7dd8", "#e07b39"]

    # (1) Episode share + success counts
    ax = axes[0, 0]
    counts = [stats[s]["n"] for s in sources]
    scored = [stats[s]["scored"] for s in sources]
    x = np.arange(len(sources))
    ax.bar(x, counts, width=0.55, color=colors, alpha=0.45, label="Total episodes")
    ax.bar(x, scored, width=0.55, color=colors, alpha=0.95, label="Scored")
    for i, s in enumerate(sources):
        ax.text(i, counts[i] + 0.4, f"{scored[i]}/{counts[i]}\n({stats[s]['success_rate']*100:.0f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_ylabel("Episodes")
    ax.set_title("Episode count and goals scored by source")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (2) Success rate comparison
    ax = axes[0, 1]
    rates = [stats[s]["success_rate"] * 100 for s in sources]
    overall = df["success"].mean() * 100
    ax.bar(x, rates, width=0.55, color=colors, alpha=0.9)
    ax.axhline(overall, color="gray", ls="--", lw=1.5, label=f"Overall {overall:.0f}%")
    for i in range(len(sources)):
        ax.text(i, rates[i] + 1.0, f"{rates[i]:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sources])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Goal-scoring success rate by source")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (3) Strike position error distribution by source
    ax = axes[1, 0]
    data_pos = [df[df["target_source"] == s]["strike_pos_err"].dropna().values for s in sources]
    bp = ax.boxplot(data_pos, tick_labels=[s.capitalize() for s in sources], patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Strike position error (m)")
    ax.set_title("Strike position error (closest approach) by source")
    ax.grid(True, axis="y", alpha=0.3)

    # (4) Strike heading error distribution by source
    ax = axes[1, 1]
    data_head = [df[df["target_source"] == s]["strike_heading_err"].dropna().values for s in sources]
    bp = ax.boxplot(data_head, tick_labels=[s.capitalize() for s in sources], patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Strike heading error (rad)")
    ax.set_title("Strike heading error by source")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "fallback_analysis.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  Saved fallback analysis figure to: {out_path}")


def write_summary(df: pd.DataFrame, stats: dict, output_dir: Path, batch_id: str):
    csv_path = output_dir / "fallback_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved per-seed table to: {csv_path}")

    n = len(df)
    overall = df["success"].mean() * 100
    unstruck = int(df.get("ball_struck", df["success"] * 0).eq(False).sum()) if "ball_struck" in df.columns else 0
    unstruck = int((df["scored"] & ~df["ball_struck"]).sum()) if "scored" in df.columns and "ball_struck" in df.columns else 0
    net, fb = stats["network"], stats["fallback"]

    import datetime
    eval_date = datetime.date.today().isoformat()

    md = f"""# Network vs. Fallback Analysis — Batch `{batch_id}`

**Evaluation date:** {eval_date}
**Sample size:** {n} runs
**Overall success rate:** {overall:.1f}% (strike-gated: goals without car contact excluded)
**Goals without car contact (excluded):** {unstruck}

## Source breakdown

Success is strike-gated: a run is a success only if the car struck the ball AND the
ball entered the goal. `Mean pred err` is `strike_point_pred_err_m` (predicted target
vs ball position at contact); `Mean contact dist` is the legacy closest-approach
diagnostic (always < CONTACT_RADIUS = 0.35 m when a strike occurred).

| Source | Episodes | Share | Scored | Success rate | Mean pred err (m) | Mean contact dist (m) |
|--------|---------:|------:|-------:|-------------:|------------------:|----------------------:|
| Network (StrikeNet trusted) | {net['n']} | {net['share']*100:.0f}% | {net['scored']} | {net['success_rate']*100:.1f}% | {net['mean_pos_err']:.3f} | {net['mean_heading_err']:.3f} |
| Fallback (analytic) | {fb['n']} | {fb['share']*100:.0f}% | {fb['scored']} | {fb['success_rate']*100:.1f}% | {fb['mean_pos_err']:.3f} | {fb['mean_heading_err']:.3f} |

## Interpretation

- StrikeNet's plan was trusted end-to-end in **{net['n']}/{n}** episodes ({net['share']*100:.0f}%); the analytic fallback engaged in the remaining **{fb['n']}/{n}** ({fb['share']*100:.0f}%).
- Network-driven success: **{net['success_rate']*100:.1f}%**; fallback success: **{fb['success_rate']*100:.1f}%**.
- The fallback only engages when the predicted plan provably cannot score in the rollout check, so a high fallback success rate is expected (it uses the exact analytic search). A large gap in favour of the fallback indicates the network's *position* prediction is still the accuracy bottleneck.
"""
    md_path = output_dir / "fallback_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved summary report to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Network vs analytic-fallback batch analysis.")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch ID under data/tests/integration/. Defaults to latest.")
    args = parser.parse_args()

    if args.batch:
        batch_dir = PROJECT_ROOT / "data" / "tests" / "integration" / args.batch
        if not batch_dir.is_dir():
            print(f"Error: batch {args.batch} not found.")
            sys.exit(1)
    else:
        batches = list_integration_batches()
        if not batches:
            print("Error: no integration batches found.")
            sys.exit(1)
        batch_dir = batches[0]

    batch_id = batch_dir.name
    output_dir = plots_batch_dir(batch_id)

    print(f"Analyzing fallback behaviour for batch: {batch_id}")
    df = load_fallback_data(batch_dir)

    # This analysis only makes sense when the network was actually consulted and
    # could fall back to analytic search (i.e. hybrid mode). Analytic-only and
    # neural-only batches have no "network"/"fallback" split, so the per-source
    # plots would be empty/garbage. Exit gracefully instead of crashing.
    sources_present = set(df["target_source"].unique())
    if not (sources_present & {"network", "fallback"}):
        modes = ", ".join(sorted(sources_present)) or "unknown"
        print(f"  Fallback analysis not applicable: batch has no network/fallback "
              f"episodes (target_source = {modes}).")
        print("  This is expected for analytic-only or neural-only batches.")
        return

    stats = _source_stats(df)

    plot_fallback_analysis(df, stats, output_dir, batch_id)
    write_summary(df, stats, output_dir, batch_id)

    print("\n" + "=" * 58)
    print("  NETWORK vs FALLBACK SUMMARY")
    print("=" * 58)
    print(f"  Batch              : {batch_id}")
    print(f"  Total runs         : {len(df)}")
    print(f"  Overall success    : {df['success'].mean()*100:.1f}%")
    print(f"  Network episodes   : {stats['network']['n']:>2d}  "
          f"success {stats['network']['success_rate']*100:.1f}%")
    print(f"  Fallback episodes  : {stats['fallback']['n']:>2d}  "
          f"success {stats['fallback']['success_rate']*100:.1f}%")
    print("=" * 58)


if __name__ == "__main__":
    main()
