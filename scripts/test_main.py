"""
test_main.py — Phase 5 Integration Test

Runs the real-time simulation loop for distinct random seeds (including wall-bounce cases).
Evaluates goal scoring success, on-field final states, and strike precision metrics.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import logging
import json
import subprocess
import pandas as pd
import multiprocessing
import concurrent.futures

# Force non-interactive matplotlib backend BEFORE importing anything else
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.ball_physics import (
    DEFAULT_BALL_DT,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_FIELD_H,
    DEFAULT_FIELD_W,
)
from src.data_layout import (
    BATCH_LOG,
    new_integration_batch,
    integration_seed_run,
)

# Default batch: 100 seeds for a comprehensive evaluation
DEFAULT_INTEGRATION_SEEDS = list(range(100, 200))


def setup_logging(batch_dir: str) -> logging.Logger:
    """Create a logger that writes to both batch.log and stdout."""
    os.makedirs(batch_dir, exist_ok=True)
    log_path = os.path.join(batch_dir, BATCH_LOG)

    logger = logging.getLogger("integration_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s"))

    # Console handler
    import logging as logging_module
    class TqdmLoggingHandler(logging_module.Handler):
        def __init__(self, level=logging_module.NOTSET):
            super().__init__(level)
        def emit(self, record):
            try:
                msg = self.format(record)
                from tqdm import tqdm
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    ch = TqdmLoggingHandler()
    ch.setLevel(logging_module.WARNING) # Only warnings/errors to console to keep progress bar clean
    ch.setFormatter(logging_module.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger

def _run_single_seed(seed: int, batch_dir: str, planner_mode: str, model_variant: str, no_video: bool) -> dict:
    """Worker function to run a single seed in an isolated subprocess."""
    try:
        seed_run_dir = integration_seed_run(batch_dir, seed)
        
        # Setup command to run main.py as a subprocess
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "main.py"),
            "--seed", str(seed),
            "--run-dir", str(seed_run_dir),
            "--planner-mode", planner_mode,
        ]
        if planner_mode != "analytic":
            cmd.extend(["--model-variant", model_variant])
            
        if not no_video:
            cmd.append("--save-video")
            
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Check for crash
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed with exit code {result.returncode}.\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
        
        # Read metadata.json and trajectory.csv
        meta_path = os.path.join(seed_run_dir, "metadata.json")
        traj_path = os.path.join(seed_run_dir, "trajectory.csv")
        
        if not os.path.exists(meta_path) or not os.path.exists(traj_path):
            raise FileNotFoundError(f"Subprocess completed but missing metadata or trajectory file in {seed_run_dir}")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        df = pd.read_csv(traj_path)
        history = df.to_dict(orient="records")
        
        # success is strike-gated in main.py (scored AND ball_struck)
        success = meta.get("success", False)
        scored = meta.get("scored", success)        # raw physics flag
        ball_struck = meta.get("ball_struck", False)
        # Detect goals that entered without a car strike (excluded from success)
        unstruck_goal = bool(scored and not ball_struck)

        # --- Headline precision metric (Issue 2) ---
        # strike_point_pred_err_m is written by main.py; fall back to NaN for
        # old batches that don't have it yet.
        strike_point_pred_err_m = meta.get("strike_point_pred_err_m", float("nan"))
        decision_latency_ms = meta.get("decision_latency_ms", float("nan"))
        fallback_sweep_ms = meta.get("fallback_sweep_ms", 0.0)

        # --- Legacy closest-approach distance (diagnostic only) ---
        if history:
            closest_step = min(history, key=lambda h: h["pos_err"])
            contact_dist = closest_step["pos_err"]
            contact_heading_err = closest_step["heading_err"]
        else:
            contact_dist = meta.get("final_pos_err_m", 999.0)
            contact_heading_err = meta.get("final_heading_err_rad", 999.0)

        final_ball = np.array([history[-1]["ball_x"], history[-1]["ball_y"]]) if history else np.array([0,0])
        final_car = np.array([history[-1]["car_x"], history[-1]["car_y"]]) if history else np.array([0,0])
        
        # Ball is allowed to cross W if it was a goal
        ball_x_max = DEFAULT_FIELD_W + 2.0 if success else DEFAULT_FIELD_W
        on_field = (
            0.0 <= final_ball[0] <= ball_x_max
            and 0.0 <= final_ball[1] <= DEFAULT_FIELD_H
            and 0.0 <= final_car[0] <= DEFAULT_FIELD_W
            and 0.0 <= final_car[1] <= DEFAULT_FIELD_H
        )

        return {
            "seed": seed,
            "crashed": False,
            "success": success,
            "scored": scored,
            "ball_struck": ball_struck,
            "unstruck_goal": unstruck_goal,
            "on_field": on_field,
            "strike_point_pred_err_m": strike_point_pred_err_m,
            "decision_latency_ms": decision_latency_ms,
            "fallback_sweep_ms": fallback_sweep_ms,
            "contact_dist": contact_dist,
            "contact_heading_err": contact_heading_err,
            "final_ball": final_ball,
            "final_car": final_car,
            "target_source": meta.get("target_source", "unknown"),
            "error_msg": "",
        }
        
    except Exception as e:
        return {
            "seed": seed,
            "crashed": True,
            "success": False,
            "scored": False,
            "ball_struck": False,
            "unstruck_goal": False,
            "on_field": False,
            "strike_point_pred_err_m": float("nan"),
            "decision_latency_ms": float("nan"),
            "contact_dist": 999.0,
            "contact_heading_err": 999.0,
            "final_ball": np.array([0, 0]),
            "final_car": np.array([0, 0]),
            "target_source": "unknown",
            "error_msg": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Integration test with per-seed videos")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_INTEGRATION_SEEDS,
        help="Random seeds to run (default: 100 seeds, 100-199)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable saving simulation videos for faster runs",
    )
    parser.add_argument("--planner-mode", choices=["analytic","neural","hybrid"], default="hybrid")
    parser.add_argument("--model-variant", choices=["legacy","structured"], default="legacy")
    parser.add_argument("--batch-dir", type=str, default=None)
    args = parser.parse_args()
    seeds = args.seeds

    batch_dir = Path(args.batch_dir) if args.batch_dir else new_integration_batch()
    log = setup_logging(str(batch_dir))
    log.info(f"Batch directory: {batch_dir}")
    log.info(f"Seeds: {seeds}")
    
    successes = 0
    failures = 0
    unstruck_goals = 0
    pred_err_vals = []     # strike_point_pred_err_m (new headline metric)
    contact_dists = []     # closest-approach distance (diagnostic only)
    latencies = []         # decision_latency_ms (deployed path)
    fb_sweeps = []         # fallback_sweep_ms (hybrid only, when fallback fires)
    net_latencies = []
    fb_latencies = []

    # Fallback/network tracking
    net_total = 0;   net_success = 0
    fb_total  = 0;   fb_success  = 0

    log.info("=" * 65)
    log.info("  PHASE 5 INTEGRATION TEST — Strike & Score with Pursuit Warm-Start")
    log.info(f"  Bounce: restitution={DEFAULT_BALL_RESTITUTION}, dt={DEFAULT_BALL_DT}, field={DEFAULT_FIELD_W}x{DEFAULT_FIELD_H}")
    log.info("=" * 65)

    # CasADi/IPOPT and Matplotlib video rendering are memory-heavy.
    # 6 workers is enough to keep all cores busy while staying within typical RAM limits.
    # Lower to 4 if you observe OOM or IPOPT restoration errors.
    num_workers = min(6, multiprocessing.cpu_count())
    log.info(f"  Running concurrently with {num_workers} thread workers to preserve RAM.")

    from tqdm import tqdm
    pbar = tqdm(total=len(seeds), desc="Integration Test")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_seed = {
            executor.submit(_run_single_seed, seed, batch_dir, args.planner_mode, args.model_variant, args.no_video): seed
            for seed in seeds
        }

        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            res = future.result()
            
            source = res["target_source"]  # "network", "fallback", or "unknown"
            source_tag = f"[{source.upper():<8}]"

            log.info(f"\n--- Seed {seed} | {source_tag} ---")

            if not res["crashed"]:
                if not res["on_field"]:
                    log.warning(f"  [WARN] Final state left the field: ball={res['final_ball']}, car={res['final_car']}")

                if res["unstruck_goal"]:
                    unstruck_goals += 1
                    log.info(f"  {source_tag} [UNSTRUCK GOAL] Ball entered goal without car contact — excluded.")

                if res["success"] and res["on_field"]:
                    successes += 1
                    log.info(f"  {source_tag} [SUCCESS] Goal scored (with strike)!")
                else:
                    failures += 1
                    log.info(f"  {source_tag} [FAILED]: Missed goal or left field.")

                pred_err = res["strike_point_pred_err_m"]
                pred_err_str = f"{pred_err:.4f} m" if not np.isnan(pred_err) else "n/a"
                lat_str = f"{res['decision_latency_ms']:.2f} ms" if not np.isnan(res["decision_latency_ms"]) else "n/a"
                log.info(f"  Pred err (target vs contact): {pred_err_str}"
                         f" | contact dist (diag): {res['contact_dist']:.4f} m"
                         f" | decision latency: {lat_str}")
                if res.get("fallback_sweep_ms", 0) > 0:
                    log.info(f"  Fallback sweep: {res['fallback_sweep_ms']:.2f} ms")

                # Track per-source counts
                if source == "network":
                    net_total += 1
                    if res["success"] and res["on_field"]:
                        net_success += 1
                elif source == "fallback":
                    fb_total += 1
                    if res["success"] and res["on_field"]:
                        fb_success += 1
            else:
                log.error(f"  [CRASH] Run crashed with exception: {res['error_msg']}")
                failures += 1

            if not np.isnan(res["strike_point_pred_err_m"]):
                pred_err_vals.append(res["strike_point_pred_err_m"])
            if not np.isnan(res["decision_latency_ms"]):
                latencies.append(res["decision_latency_ms"])
                if source == "network":
                    net_latencies.append(res["decision_latency_ms"])
                elif source == "fallback":
                    fb_latencies.append(res["decision_latency_ms"])
            if res.get("fallback_sweep_ms", 0) > 0:
                fb_sweeps.append(res["fallback_sweep_ms"])
            contact_dists.append(res["contact_dist"])

            pbar.update(1)
            mean_pred = np.mean(pred_err_vals) if pred_err_vals else float("nan")
            pbar.set_postfix(success=successes, fail=failures, pred_err=f"{mean_pred:.2f}")

    pbar.close()

    # Summary
    log.info("")
    log.info("=" * 65)
    log.info("  INTEGRATION TEST SUMMARY")
    log.info("=" * 65)
    net_rate = net_success / net_total * 100 if net_total else 0.0
    fb_rate  = fb_success  / fb_total  * 100 if fb_total  else 0.0

    mean_pred_err = np.mean(pred_err_vals) if pred_err_vals else float("nan")
    median_pred_err = np.median(pred_err_vals) if pred_err_vals else float("nan")
    mean_contact = np.mean(contact_dists) if contact_dists else float("nan")
    mean_decision_latency_ms = np.mean(latencies) if latencies else float("nan")
    median_decision_latency_ms = np.median(latencies) if latencies else float("nan")
    p90_decision_latency_ms = float(np.percentile(latencies, 90)) if latencies else float("nan")
    mean_fallback_sweep_ms = np.mean(fb_sweeps) if fb_sweeps else 0.0
    median_net_latency_ms = np.median(net_latencies) if net_latencies else float("nan")
    median_fb_latency_ms = np.median(fb_latencies) if fb_latencies else float("nan")

    log.info(f"  Total runs       : {len(seeds)}")
    log.info(f"  Goals (Success)  : {successes} / {len(seeds)}  ({successes/len(seeds)*100:.1f}%)")
    log.info(f"  Misses (Fail)    : {failures} / {len(seeds)}")
    log.info(f"  Goals without strike (excluded): {unstruck_goals}")
    log.info(f"  Avg Pred Target Err (strike_point_pred_err_m): {mean_pred_err:.4f} m  [median: {median_pred_err:.4f} m]")
    log.info(f"  Avg Contact Dist (diagnostic only)           : {mean_contact:.4f} m")
    log.info(f"  Decision Latency (deployed path)             : mean {mean_decision_latency_ms:.3f} ms  "
             f"| median {median_decision_latency_ms:.3f} ms  | p90 {p90_decision_latency_ms:.3f} ms")
    if args.planner_mode == "hybrid" and fb_sweeps:
        log.info(f"  Fallback sweep (when engaged)                : mean {mean_fallback_sweep_ms:.1f} ms")
        log.info(f"  Latency by path — network median: {median_net_latency_ms:.3f} ms  "
                 f"| fallback median: {median_fb_latency_ms:.3f} ms")
    log.info("  (No hard latency pass/fail gate; NMPC control period is 100 ms/step.)")
    log.info("-" * 65)
    log.info("  NETWORK vs FALLBACK BREAKDOWN")
    log.info(f"  [NETWORK ] Episodes: {net_total:>3d}  |  Scored: {net_success:>3d}  |  Success rate: {net_rate:.1f}%")
    log.info(f"  [FALLBACK] Episodes: {fb_total:>3d}  |  Scored: {fb_success:>3d}  |  Success rate: {fb_rate:.1f}%")
    log.info("-" * 65)

    summary = {
      "planner_mode": args.planner_mode,
      "model_variant": (None if args.planner_mode == "analytic" else args.model_variant),
      "n": len(seeds),
      "successes": successes,
      "success_rate": successes/len(seeds) if len(seeds) > 0 else 0.0,
      "unstruck_goals": unstruck_goals,
      "mean_pred_err_m": float(mean_pred_err),
      "median_pred_err_m": float(median_pred_err),
      "mean_contact_dist_m": float(mean_contact),
      "net_episodes": net_total,
      "net_success": net_success,
      "fb_episodes": fb_total,
      "fb_success": fb_success,
      "mean_decision_latency_ms": float(mean_decision_latency_ms),
      "median_decision_latency_ms": float(median_decision_latency_ms),
      "p90_decision_latency_ms": float(p90_decision_latency_ms),
      "mean_fallback_sweep_ms": float(mean_fallback_sweep_ms),
      "median_net_latency_ms": float(median_net_latency_ms),
      "median_fb_latency_ms": float(median_fb_latency_ms),
    }
    with open(os.path.join(batch_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Pass criteria: at least 60% success (strike-gated).
    # The old closest-approach threshold (<= 0.35 m) is removed — it was tautological
    # because contact triggers at CONTACT_RADIUS = 0.35. Success rate is the gate.
    min_successes = max(1, int(np.ceil(0.6 * len(seeds))))
    passed = successes >= min_successes
    log.info(f"  Pass threshold     : {min_successes} / {len(seeds)} successes (60%)  "
             f"[network {net_total} eps, fallback {fb_total} eps]")

    if passed:
        log.info("  [PASSED] INTEGRATION TEST PASSED!")
        return 0
    else:
        log.info("  [FAILED] INTEGRATION TEST FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
