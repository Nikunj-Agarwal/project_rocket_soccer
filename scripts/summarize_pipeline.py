"""
summarize_pipeline.py — Consolidated report after a full run_pipeline execution.

Prints a single terminal summary of dataset stats, training, integration batch,
reports, scalability benchmark, and 5-config comparison. Optionally saves to
data/reports/pipeline_summaries/{timestamp}_pipeline_summary.md.

Usage:
  python scripts/summarize_pipeline.py
  python scripts/summarize_pipeline.py --save
  python scripts/summarize_pipeline.py --integration-batch 20260613_120000 --comparison-run 20260613_140000 --save
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_layout import (
    COMPARISON_TESTS_DIR,
    DATASET_STATS,
    INTEGRATION_TESTS_DIR,
    PIPELINE_SUMMARIES_DIR,
    RUN_METADATA,
    SCALABILITY_CSV,
    STRIKE_NET_LEGACY,
    STRIKE_NET_STRUCTURED,
    TRAINING_LOG_LEGACY,
    TRAINING_LOG_STRUCTURED,
    ensure_dir,
    iter_integration_seed_runs,
    latest_comparison_run,
    latest_integration_batch,
    plots_comparison_dir,
    plots_batch_dir,
)


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{100.0 * float(x):.1f}%"


def _fmt_float(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _model_line(path: Path) -> str:
    if not path.is_file():
        return f"  MISSING  {path.relative_to(PROJECT_ROOT)}"
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    size_kb = stat.st_size / 1024
    return f"  OK       {path.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB, {mtime})"


def _aggregate_from_metadata(batch_dir: Path) -> dict | None:
    """Fallback when summary.json is missing (older batches)."""
    rows = []
    for seed_str, run_dir in iter_integration_seed_runs(batch_dir):
        meta = _read_json(run_dir / RUN_METADATA)
        if meta:
            rows.append(meta)
    if not rows:
        return None
    n = len(rows)
    successes = sum(1 for m in rows if m.get("success"))
    pred_errs = [m["strike_point_pred_err_m"] for m in rows
                 if m.get("strike_point_pred_err_m") is not None
                 and not (isinstance(m["strike_point_pred_err_m"], float) and pd.isna(m["strike_point_pred_err_m"]))]
    latencies = [m["decision_latency_ms"] for m in rows
                 if m.get("decision_latency_ms") is not None]
    net = [m for m in rows if m.get("target_source") == "network"]
    fb = [m for m in rows if m.get("target_source") == "fallback"]
    return {
        "planner_mode": rows[0].get("planner_mode", "?"),
        "model_variant": rows[0].get("model_variant"),
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "mean_pred_err_m": float(pd.Series(pred_errs).mean()) if pred_errs else float("nan"),
        "median_pred_err_m": float(pd.Series(pred_errs).median()) if pred_errs else float("nan"),
        "mean_decision_latency_ms": float(pd.Series(latencies).mean()) if latencies else float("nan"),
        "net_episodes": len(net),
        "net_success": sum(1 for m in net if m.get("success")),
        "fb_episodes": len(fb),
        "fb_success": sum(1 for m in fb if m.get("success")),
        "unstruck_goals": sum(1 for m in rows if m.get("scored") and not m.get("ball_struck")),
        "_from_metadata": True,
    }


def _training_tail(log_path: Path, variant: str) -> list[str]:
    lines = [f"### Training - {variant}"]
    if not log_path.is_file():
        lines.append(f"- Log not found: `{log_path.relative_to(PROJECT_ROOT)}`")
        return lines
    df = pd.read_csv(log_path)
    if df.empty:
        lines.append("- Log is empty.")
        return lines
    last = df.iloc[-1]
    best = df.loc[df["test_loss"].idxmin()] if "test_loss" in df.columns else last
    lines.append(f"- Epochs logged: {len(df)}")
    lines.append(f"- Final train_loss: {_fmt_float(last.get('train_loss', None), 4)}  "
                 f"test_loss: {_fmt_float(last.get('test_loss', None), 4)}")
    lines.append(f"- Best test_loss: {_fmt_float(best.get('test_loss', None), 4)} "
                 f"(epoch {int(best.get('epoch', 0))})")
    lines.append(f"- Log: `{log_path.relative_to(PROJECT_ROOT)}`")
    return lines


def _integration_section(batch_dir: Path | None) -> list[str]:
    lines = ["## Integration test batch"]
    if batch_dir is None:
        lines.append("- No integration batch found under `data/tests/integration/`.")
        return lines

    bid = batch_dir.name
    lines.append(f"- Batch ID: `{bid}`")
    lines.append(f"- Raw data: `data/tests/integration/{bid}/`")

    summary = _read_json(batch_dir / "summary.json")
    if summary is None:
        summary = _aggregate_from_metadata(batch_dir)
        if summary:
            lines.append("- *(aggregated from per-seed metadata.json; no summary.json)*")

    if summary:
        mode = summary.get("planner_mode", "?")
        variant = summary.get("model_variant") or "n/a"
        n = summary.get("n", "?")
        sr = summary.get("success_rate")
        passed = sr is not None and float(sr) >= 0.6
        lines.append(f"- Config: **{mode}** / **{variant}**  |  seeds: {n}")
        lines.append(f"- Success rate: **{_fmt_pct(sr)}** ({summary.get('successes', '?')}/{n})  "
                     f"-> {'PASS' if passed else 'FAIL'} (threshold 60%)")
        lines.append(f"- Mean pred err (m): {_fmt_float(summary.get('mean_pred_err_m'))}  "
                     f"| median: {_fmt_float(summary.get('median_pred_err_m'))}")
        lines.append(f"- Mean decision latency (ms): {_fmt_float(summary.get('mean_decision_latency_ms'), 2)}")
        if summary.get("net_episodes") is not None:
            lines.append(f"- Network episodes: {summary.get('net_episodes')}  "
                         f"(scored {summary.get('net_success', 0)})")
            lines.append(f"- Fallback episodes: {summary.get('fb_episodes')}  "
                         f"(scored {summary.get('fb_success', 0)})")
        lines.append(f"- Unstruck goals (excluded): {summary.get('unstruck_goals', 0)}")
    else:
        lines.append("- `summary.json` not found in batch directory.")

    plot_dir = plots_batch_dir(bid)
    report_files = [
        plot_dir / "research_summary.md",
        plot_dir / "fallback_summary.md",
        plot_dir / "fallback_analysis.png",
        plot_dir / "integration_summary.png",
    ]
    lines.append("- Reports:")
    for p in report_files:
        tag = "OK" if p.is_file() else "--"
        lines.append(f"  - [{tag}] `{p.relative_to(PROJECT_ROOT)}`")
    return lines


def _comparison_section(comp_dir: Path | None) -> list[str]:
    lines = ["## Five-config comparison"]
    if comp_dir is None:
        lines.append("- No comparison run found under `data/tests/comparison/`.")
        return lines

    rid = comp_dir.name
    lines.append(f"- Run ID: `{rid}`")
    lines.append(f"- Raw batches: `data/tests/comparison/{rid}/`")

    plots_dir = plots_comparison_dir(rid)
    csv_path = plots_dir / "comparison.csv"
    if not csv_path.is_file():
        lines.append("- `comparison.csv` not found (comparison may still be running or failed).")
        subdirs = [d.name for d in comp_dir.iterdir() if d.is_dir()]
        if subdirs:
            lines.append(f"- Subdirs present: {', '.join(sorted(subdirs))}")
        return lines

    df = pd.read_csv(csv_path)
    lines.append(f"- Report: `{plots_dir.relative_to(PROJECT_ROOT)}/`")
    lines.append("")
    lines.append("| Config | Success | Pred err (m) | Latency (ms) | FB share |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for _, row in df.iterrows():
        fb = row.get("fallback_share", 0)
        fb_str = _fmt_pct(fb) if pd.notna(fb) and float(fb) > 0 else "n/a"
        lines.append(
            f"| {row.get('config_name', '?')} "
            f"| {_fmt_pct(row.get('success_rate'))} "
            f"| {_fmt_float(row.get('mean_pred_err_m'))} "
            f"| {_fmt_float(row.get('mean_decision_latency_ms'), 2)} "
            f"| {fb_str} |"
        )
    return lines


def _scalability_section() -> list[str]:
    lines = ["## Scalability benchmark"]
    if not SCALABILITY_CSV.is_file():
        lines.append(f"- Not found: `{SCALABILITY_CSV.relative_to(PROJECT_ROOT)}`")
        return lines

    df = pd.read_csv(SCALABILITY_CSV)
    lines.append(f"- CSV: `{SCALABILITY_CSV.relative_to(PROJECT_ROOT)}`")
    if "variant" in df.columns:
        for variant in df["variant"].unique():
            sub = df[df["variant"] == variant]
            net = sub["network_ms_mean"].iloc[0] if len(sub) else float("nan")
            n36 = sub[sub["n_angles"] == 36]
            if len(n36):
                speedup = n36["speedup"].iloc[0]
                analytic = n36["analytic_ms_mean"].iloc[0]
                line = (f"- **{variant}**: network {_fmt_float(net, 2)} ms  "
                        f"| analytic@36 {_fmt_float(analytic, 1)} ms  "
                        f"| speedup {_fmt_float(speedup, 1)}x")
                if "fallback_sweep_ms_mean" in n36.columns:
                    fb_sweep = pd.to_numeric(n36["fallback_sweep_ms_mean"], errors="coerce").iloc[0]
                    worst = pd.to_numeric(n36.get("deployed_hybrid_worst_ms"), errors="coerce").iloc[0] if "deployed_hybrid_worst_ms" in n36.columns else float("nan")
                    if pd.notna(fb_sweep):
                        line += (f"  | hybrid fallback sweep {_fmt_float(fb_sweep, 1)} ms  "
                                 f"| hybrid worst-case {_fmt_float(worst, 1)} ms")
                lines.append(line)
    else:
        row36 = df[df["n_angles"] == 36]
        if len(row36):
            r = row36.iloc[0]
            lines.append(f"- Network: {_fmt_float(r.get('network_ms_mean'), 2)} ms  "
                         f"| analytic@36: {_fmt_float(r.get('analytic_ms_mean'), 1)} ms  "
                         f"| speedup: {_fmt_float(r.get('speedup'), 1)}x")
    plot = PROJECT_ROOT / "data/reports/plots/global/scalability_curve.png"
    if plot.is_file():
        lines.append(f"- Plot: `{plot.relative_to(PROJECT_ROOT)}`")
    return lines


def build_summary(
    integration_batch: Path | None,
    comparison_run: Path | None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts: list[str] = [
        "# Pipeline run summary",
        f"Generated: {now}",
        "",
        "## Resolved paths",
        f"- Integration batch: `{integration_batch.name if integration_batch else 'NONE'}`",
        f"- Comparison run: `{comparison_run.name if comparison_run else 'NONE'}`",
        "",
        "## Models",
        _model_line(STRIKE_NET_LEGACY),
        _model_line(STRIKE_NET_STRUCTURED),
        "",
        "## Dataset generation",
    ]

    stats = _read_json(DATASET_STATS)
    if stats:
        parts.append(f"- Samples: {stats.get('num_samples', '?')}  "
                     f"(acceptance {_fmt_pct(stats.get('acceptance_rate'))})")
        parts.append(f"- Wall clock: {_fmt_float(stats.get('wall_clock_s'), 1)} s  "
                     f"| workers: {stats.get('num_workers', '?')}")
        parts.append(f"- Median search / valid sample: "
                     f"{_fmt_float(stats.get('median_search_s_per_valid_sample'), 3)} s")
        parts.append(f"- Stats: `{DATASET_STATS.relative_to(PROJECT_ROOT)}`")
    else:
        parts.append(f"- `{DATASET_STATS.relative_to(PROJECT_ROOT)}` not found.")

    parts.append("")
    parts.extend(_training_tail(TRAINING_LOG_LEGACY, "legacy"))
    parts.append("")
    parts.extend(_training_tail(TRAINING_LOG_STRUCTURED, "structured"))
    parts.append("")
    parts.extend(_integration_section(integration_batch))
    parts.append("")
    parts.extend(_comparison_section(comparison_run))
    parts.append("")
    parts.extend(_scalability_section())
    parts.extend([
        "",
        "## Quick commands",
        "```powershell",
        f"python scripts/generate_plots.py --batch {integration_batch.name if integration_batch else '<batch_id>'}",
        f"python -m scripts.analyze_fallback --batch {integration_batch.name if integration_batch else '<batch_id>'}",
        "```",
    ])
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidated pipeline run summary")
    parser.add_argument("--integration-batch", type=str, default=None,
                        help="Integration batch ID (default: latest)")
    parser.add_argument("--comparison-run", type=str, default=None,
                        help="Comparison run ID (default: latest)")
    parser.add_argument("--save", action="store_true",
                        help="Write markdown report to data/reports/pipeline_summaries/")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print to terminal (only meaningful with --save)")
    args = parser.parse_args()

    if args.integration_batch:
        batch_dir = INTEGRATION_TESTS_DIR / args.integration_batch
        if not batch_dir.is_dir():
            print(f"Error: integration batch not found: {batch_dir}", file=sys.stderr)
            return 1
    else:
        batch_dir = latest_integration_batch()

    if args.comparison_run:
        comp_dir = COMPARISON_TESTS_DIR / args.comparison_run
        if not comp_dir.is_dir():
            print(f"Error: comparison run not found: {comp_dir}", file=sys.stderr)
            return 1
    else:
        comp_dir = latest_comparison_run()

    text = build_summary(batch_dir, comp_dir)

    if not args.quiet:
        print("\n" + "=" * 72 + "\n")
        print(text)
        print("\n" + "=" * 72 + "\n")

    if args.save:
        out_dir = ensure_dir(PIPELINE_SUMMARIES_DIR)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{ts}_pipeline_summary.md"
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")

    if batch_dir is None and comp_dir is None:
        print("Warning: no integration or comparison batches found.", file=sys.stderr)
        return 1
    if comp_dir is None:
        print("Note: no comparison run yet (step 7 may still be running).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
