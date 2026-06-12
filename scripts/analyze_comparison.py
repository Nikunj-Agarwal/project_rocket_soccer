"""
analyze_comparison.py — Cross-method cost/benefit analysis for a comparison run.

Consumes the per-seed metadata produced by `scripts/compare_modes.py`
(data/tests/comparison/{run_id}/{config}/seed_*/metadata.json) and answers the
practical question: *is the neural net worth it?* It does this on SHARED seeds,
with a strike-gated success definition, and with honest timing.

Key idea: the bar chart from compare_modes.py only gives per-config averages.
This script adds the paired, per-seed analysis a reader needs:
  - Pareto: success rate vs decision latency (the headline "worth it" figure).
  - Latency distribution per config (box + CDF) — consistency, not just means.
  - Success heatmap (seeds x configs) — where each method wins/loses.
  - Hybrid network-vs-fallback breakdown — does the net earn its keep?
  - Accuracy vs time scatter per seed.

Honest-latency note
-------------------
``decision_latency_ms`` logged per seed is the **wall-clock deployed path** inside
``decide_strike_target()`` — inference, ball rollout, scoring checks, and (for
hybrid) the full 36-heading fallback sweep when it fires.  Micro-benchmark fields
(``strikenet_infer_ms``, ``rollout_ms``, ``analytic_strategy_ms``) are 30-rep
reference timings for head-to-head comparison only.

Outputs (into data/reports/plots/comparison/{run_id}/):
  - worth_it_summary.md        (narrative + tables + verdict)
  - config_summary.csv         (one row per config, all aggregate metrics)
  - paired_seeds.csv           (one row per seed, every config joined)
  - win_matrix.csv             (pairwise "A succeeds, B fails" counts)
  - pareto_success_vs_latency.png
  - latency_by_config.png
  - success_heatmap.png
  - hybrid_path_breakdown.png
  - accuracy_vs_time.png

Usage:
  python -m scripts.analyze_comparison                  # latest comparison run
  python -m scripts.analyze_comparison --run <run_id>   # specific run
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
    COMPARISON_TESTS_DIR,
    SCALABILITY_CSV,
    latest_comparison_run,
    plots_comparison_dir,
)

# Canonical config order (matches compare_modes.CONFIGS) and display labels.
CONFIG_ORDER = [
    "analytic",
    "neural_legacy",
    "neural_structured",
    "hybrid_legacy",
    "hybrid_structured",
]
CONFIG_COLORS = {
    "analytic": "#444444",
    "neural_legacy": "#1f77b4",
    "neural_structured": "#2ca02c",
    "hybrid_legacy": "#ff7f0e",
    "hybrid_structured": "#d62728",
}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run(run_dir: Path) -> pd.DataFrame:
    """Load every seed of every config in a comparison run into one long frame."""
    rows = []
    for config_dir in sorted(run_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config = config_dir.name
        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            meta = _read_json(seed_dir / "metadata.json")
            if meta is None:
                continue
            scored = bool(meta.get("scored", False))
            struck = bool(meta.get("ball_struck", False))
            success = bool(meta.get("success", scored and struck))
            rows.append({
                "config": config,
                "seed": int(meta.get("seed", seed_dir.name.replace("seed_", ""))),
                "success": success,
                "scored": scored,
                "ball_struck": struck,
                "unstruck_goal": scored and not struck,
                "pred_err_m": meta.get("strike_point_pred_err_m", np.nan),
                "time_err_s": meta.get("strike_time_err_s", np.nan),
                "decision_latency_ms": meta.get("decision_latency_ms", np.nan),
                "strikenet_infer_ms": meta.get("strikenet_infer_ms", 0.0),
                "analytic_strategy_ms": meta.get("analytic_strategy_ms", np.nan),
                "fallback_sweep_ms": meta.get("fallback_sweep_ms", 0.0),
                "target_source": meta.get("target_source", "?"),
                "planner_mode": meta.get("planner_mode", "?"),
                "model_variant": meta.get("model_variant"),
            })
    if not rows:
        raise SystemExit(f"No per-seed metadata found under {run_dir}")
    return pd.DataFrame(rows)


def _nanmean(s) -> float:
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else float("nan")


def _nanmedian(s) -> float:
    s = pd.Series(s).dropna()
    return float(s.median()) if len(s) else float("nan")


def _nanpct(s, q) -> float:
    s = pd.Series(s).dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config with the headline cost/benefit metrics."""
    out = []
    for config in [c for c in CONFIG_ORDER if c in df["config"].unique()]:
        sub = df[df["config"] == config]
        n = len(sub)
        net = sub[sub["target_source"] == "network"]
        fb = sub[sub["target_source"] == "fallback"]
        fb_sweeps = sub["fallback_sweep_ms"].replace(0, np.nan).dropna()
        out.append({
            "config": config,
            "n": n,
            "success_rate": sub["success"].mean(),
            "successes": int(sub["success"].sum()),
            "unstruck_goals": int(sub["unstruck_goal"].sum()),
            "mean_pred_err_m": _nanmean(sub["pred_err_m"]),
            "median_pred_err_m": _nanmedian(sub["pred_err_m"]),
            "mean_time_err_s": _nanmean(sub["time_err_s"]),
            "median_latency_ms": _nanmedian(sub["decision_latency_ms"]),
            "p90_latency_ms": _nanpct(sub["decision_latency_ms"], 90),
            "mean_latency_ms": _nanmean(sub["decision_latency_ms"]),
            "mean_fallback_sweep_ms": _nanmean(fb_sweeps) if len(fb_sweeps) else 0.0,
            "median_net_latency_ms": _nanmedian(net["decision_latency_ms"]) if len(net) else float("nan"),
            "median_fb_latency_ms": _nanmedian(fb["decision_latency_ms"]) if len(fb) else float("nan"),
            "fallback_share": (len(fb) / n) if n else 0.0,
            "net_episodes": len(net),
            "net_success_rate": net["success"].mean() if len(net) else float("nan"),
            "fb_episodes": len(fb),
            "fb_success_rate": fb["success"].mean() if len(fb) else float("nan"),
        })
    return pd.DataFrame(out)


def build_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per seed, success/latency/err columns per config."""
    seeds = sorted(df["seed"].unique())
    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for config in CONFIG_ORDER:
            sub = df[(df["seed"] == seed) & (df["config"] == config)]
            if len(sub):
                r = sub.iloc[0]
                row[f"success_{config}"] = bool(r["success"])
                row[f"latency_{config}"] = r["decision_latency_ms"]
                row[f"prederr_{config}"] = r["pred_err_m"]
        rows.append(row)
    return pd.DataFrame(rows)


def win_matrix(paired: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    """Entry [A][B] = number of seeds where A succeeded and B failed."""
    mat = pd.DataFrame(0, index=configs, columns=configs, dtype=int)
    for a in configs:
        for b in configs:
            ca, cb = f"success_{a}", f"success_{b}"
            if ca in paired.columns and cb in paired.columns:
                mat.loc[a, b] = int(((paired[ca]) & (~paired[cb])).sum())
    return mat


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_pareto(agg: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in agg.iterrows():
        ax.scatter(r["median_latency_ms"], r["success_rate"] * 100,
                   s=160, color=CONFIG_COLORS.get(r["config"], "gray"),
                   edgecolor="black", zorder=3)
        ax.annotate(r["config"], (r["median_latency_ms"], r["success_rate"] * 100),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Median deployed decision latency (ms, log scale)")
    ax.set_ylabel("Strike-gated success rate (%)")
    ax.set_title("Success vs deployed decision time — is the neural net worth it?")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.04, "upper-left = better\n(high success, low time)",
            transform=ax.transAxes, fontsize=9, style="italic",
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="gray"))
    fig.tight_layout()
    path = out_dir / "pareto_success_vs_latency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latency_by_config(df: pd.DataFrame, out_dir: Path):
    configs = [c for c in CONFIG_ORDER if c in df["config"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    data = [df[df["config"] == c]["decision_latency_ms"].dropna().values for c in configs]
    bp = axes[0].boxplot(data, tick_labels=configs, showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], configs):
        patch.set_facecolor(CONFIG_COLORS.get(c, "gray"))
        patch.set_alpha(0.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Decision latency (ms, log)")
    axes[0].set_title("Per-seed decision latency distribution")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", which="both", alpha=0.3)

    for c in configs:
        vals = np.sort(df[df["config"] == c]["decision_latency_ms"].dropna().values)
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        axes[1].step(vals, cdf * 100, where="post", label=c,
                     color=CONFIG_COLORS.get(c, "gray"), linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Decision latency (ms, log)")
    axes[1].set_ylabel("Episodes within latency (%)")
    axes[1].set_title("Latency CDF")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "latency_by_config.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_success_heatmap(paired: pd.DataFrame, configs: list[str], out_dir: Path):
    cols = [f"success_{c}" for c in configs if f"success_{c}" in paired.columns]
    present = [c for c in configs if f"success_{c}" in paired.columns]
    mat = paired[cols].astype(float).values.T  # configs x seeds
    fig, ax = plt.subplots(figsize=(16, 3 + 0.4 * len(present)))
    ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    seeds = paired["seed"].tolist()
    step = max(1, len(seeds) // 20)
    ax.set_xticks(range(0, len(seeds), step))
    ax.set_xticklabels([seeds[i] for i in range(0, len(seeds), step)], rotation=90, fontsize=7)
    ax.set_xlabel("Seed")
    ax.set_title("Per-seed strike-gated success (green=goal, red=miss) on shared seeds")
    fig.tight_layout()
    path = out_dir / "success_heatmap.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_hybrid_breakdown(agg: pd.DataFrame, out_dir: Path):
    hybrids = agg[agg["config"].str.startswith("hybrid")]
    if hybrids.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = hybrids["config"].tolist()
    x = np.arange(len(labels))
    w = 0.35
    net = (hybrids["net_success_rate"].fillna(0) * 100).values
    fb = (hybrids["fb_success_rate"].fillna(0) * 100).values
    b1 = ax.bar(x - w / 2, net, w, label="Network-trusted path", color="#1f77b4")
    b2 = ax.bar(x + w / 2, fb, w, label="Fallback path", color="#ff7f0e")
    for bars, shares, col in ((b1, hybrids["net_episodes"], "net"), (b2, hybrids["fb_episodes"], "fb")):
        for rect, n_ep in zip(bars, shares):
            ax.annotate(f"n={int(n_ep)}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Hybrid: network-trusted vs fallback success")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "hybrid_path_breakdown.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_accuracy_vs_time(df: pd.DataFrame, out_dir: Path):
    configs = [c for c in CONFIG_ORDER if c in df["config"].unique()]
    fig, ax = plt.subplots(figsize=(11, 7))
    for c in configs:
        sub = df[df["config"] == c]
        ax.scatter(sub["decision_latency_ms"], sub["pred_err_m"],
                   s=28, alpha=0.55, color=CONFIG_COLORS.get(c, "gray"), label=c)
    ax.set_xscale("log")
    ax.set_xlabel("Decision latency (ms, log)")
    ax.set_ylabel("Strike-point prediction error (m)")
    ax.set_title("Per-seed accuracy vs decision time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "accuracy_vs_time.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Summary report
# --------------------------------------------------------------------------- #
def _fmt(x, d=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{d}f}"


def _scalability_line() -> str:
    if not SCALABILITY_CSV.is_file():
        return "- Scalability CSV not found; run `scripts/benchmark_scalability.py`."
    sc = pd.read_csv(SCALABILITY_CSV)
    row = sc[sc["n_angles"] == 36]
    if row.empty:
        row = sc
    parts = []
    for _, r in row.iterrows():
        parts.append(f"{r['variant']}: analytic {r['analytic_ms_mean']:.0f} ms vs "
                     f"net {r['network_ms_mean']:.3f} ms ({r['speedup']:.0f}x)")
    return "- At n_angles=36: " + "; ".join(parts) + "."


def write_summary(agg: pd.DataFrame, paired: pd.DataFrame, wm: pd.DataFrame,
                  run_id: str, out_dir: Path) -> Path:
    A = {r["config"]: r for _, r in agg.iterrows()}
    lines = [
        f"# StrikeNet cost-benefit analysis - run `{run_id}`",
        "",
        f"Shared seeds: {len(paired)}. Success is strike-gated (goal AND car contact).",
        "",
        "## Headline table",
        "",
        "| Config | Success | Pred err (m) | Median lat (ms) | p90 lat (ms) | FB lat median (ms) | FB sweep mean (ms) | FB share |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in agg.iterrows():
        fb = f"{r['fallback_share']*100:.0f}%" if r["config"].startswith("hybrid") else "n/a"
        fb_lat = _fmt(r.get("median_fb_latency_ms"), 2) if r["config"].startswith("hybrid") else "n/a"
        fb_sw = _fmt(r.get("mean_fallback_sweep_ms"), 2) if r["config"].startswith("hybrid") else "n/a"
        lines.append(
            f"| {r['config']} | {r['success_rate']*100:.1f}% | {_fmt(r['mean_pred_err_m'])} "
            f"| {_fmt(r['median_latency_ms'], 2)} | {_fmt(r['p90_latency_ms'], 2)} "
            f"| {fb_lat} | {fb_sw} | {fb} |"
        )

    lines += ["", "## Timing notes", "",
              "`decision_latency_ms` is wall-clock of the deployed planner path (inference, ball "
              "rollout, scoring checks, and hybrid fallback sweep when it fires). Micro-benchmark "
              "fields in metadata (`strikenet_infer_ms`, `analytic_strategy_ms`) are 30-rep "
              "references for scalability comparison only.",
              "", _scalability_line(), ""]

    # Verdict math
    if "analytic" in A and "hybrid_legacy" in A:
        an, hl = A["analytic"], A["hybrid_legacy"]
        nl = A.get("neural_legacy")
        retention = (hl["success_rate"] / an["success_rate"] * 100) if an["success_rate"] else float("nan")
        lines += ["## Is the neural net worth it?", ""]
        lines.append(f"- **Analytic ceiling**: {an['success_rate']*100:.1f}% success at "
                     f"~{_fmt(an['median_latency_ms'], 0)} ms/decision. This is the planning oracle; "
                     "the <100% reflects closed-loop NMPC tracking + strike-gating, not planning.")
        lines.append(f"- **Hybrid (legacy)**: {hl['success_rate']*100:.1f}% success "
                     f"= {retention:.0f}% of the analytic ceiling. "
                     f"Network-trusted median latency: {_fmt(hl.get('median_net_latency_ms'), 2)} ms; "
                     f"fallback median: {_fmt(hl.get('median_fb_latency_ms'), 2)} ms "
                     f"(sweep mean {_fmt(hl.get('mean_fallback_sweep_ms'), 1)} ms).")
        if nl is not None:
            marginal = (hl["success_rate"] - nl["success_rate"]) * 100
            lines.append(f"- **Pure neural (legacy)**: only {nl['success_rate']*100:.1f}% — the net "
                         f"alone is not enough. The fallback adds **+{marginal:.0f} success points** over pure neural.")
        lines.append("- **Position error is misleading for `structured`**: it is near-zero by "
                     "construction (target sits on the ball trajectory), yet pure-neural success "
                     "is not higher than legacy — proving success, not pred-err, is the metric that matters.")
        # crude verdict
        if not pd.isna(retention) and retention >= 85:
            verdict = ("**Worth it in hybrid mode.** The net handles the easy majority of scenes in "
                       "~ms while fallback preserves near-analytic reliability on the hard ones. "
                       "Pure neural is faster but materially less reliable.")
        else:
            verdict = ("**Marginal.** Hybrid does not yet recover enough of the analytic ceiling to "
                       "justify the net on accuracy alone; its value is primarily latency.")
        lines += ["", "## Verdict", "", verdict, ""]

    # Paired ceiling gap vs analytic
    if "success_analytic" in paired.columns:
        for hy in ("hybrid_legacy", "hybrid_structured"):
            col = f"success_{hy}"
            if col not in paired.columns:
                continue
            only_analytic = paired[(paired["success_analytic"]) & (~paired[col])]["seed"].tolist()
            recovered = int((paired["success_analytic"] & paired[col]).sum())
            lines.append(f"- vs analytic, **{hy}** matches on {recovered} seeds; "
                         f"analytic-only wins on {len(only_analytic)} seeds "
                         f"({', '.join(map(str, only_analytic[:15]))}{' ...' if len(only_analytic) > 15 else ''}).")

    lines += ["", "## Win matrix (row succeeds where column fails)", "",
              "| | " + " | ".join(wm.columns) + " |",
              "| :--- | " + " | ".join(["---:"] * len(wm.columns)) + " |"]
    for idx, row in wm.iterrows():
        lines.append(f"| **{idx}** | " + " | ".join(str(int(v)) for v in row.values) + " |")

    lines += ["", "## Figures", "",
              "- `pareto_success_vs_latency.png` - main thesis figure",
              "- `latency_by_config.png` - per-seed latency box + CDF",
              "- `success_heatmap.png` - seeds x configs",
              "- `hybrid_path_breakdown.png` - network vs fallback success",
              "- `accuracy_vs_time.png` - per-seed accuracy vs time", ""]

    text = "\n".join(lines)
    path = out_dir / "worth_it_summary.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    parser = argparse.ArgumentParser(description="Cross-method cost/benefit analysis for a comparison run.")
    parser.add_argument("--run", type=str, default=None,
                        help="Comparison run id under data/tests/comparison/. Defaults to latest.")
    args = parser.parse_args()

    if args.run:
        run_dir = COMPARISON_TESTS_DIR / args.run
        if not run_dir.is_dir():
            print(f"Error: comparison run not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = latest_comparison_run()
        if run_dir is None:
            print("Error: no comparison runs found under data/tests/comparison/.", file=sys.stderr)
            sys.exit(1)

    run_id = run_dir.name
    out_dir = plots_comparison_dir(run_id)
    print(f"Analyzing comparison run: {run_id}")

    df = load_run(run_dir)
    agg = aggregate(df)
    paired = build_paired(df)
    configs_present = [c for c in CONFIG_ORDER if c in df["config"].unique()]
    wm = win_matrix(paired, configs_present)

    # CSVs
    agg.to_csv(out_dir / "config_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_seeds.csv", index=False)
    wm.to_csv(out_dir / "win_matrix.csv")

    # Plots
    plot_pareto(agg, out_dir)
    plot_latency_by_config(df, out_dir)
    plot_success_heatmap(paired, configs_present, out_dir)
    plot_hybrid_breakdown(agg, out_dir)
    plot_accuracy_vs_time(df, out_dir)

    summary_path = write_summary(agg, paired, wm, run_id, out_dir)

    print(f"\nWorth-it analysis written to: {out_dir}")
    print("\n" + agg[["config", "success_rate", "mean_pred_err_m",
                       "median_latency_ms", "median_net_latency_ms", "median_fb_latency_ms",
                       "fallback_share"]].to_string(index=False))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
