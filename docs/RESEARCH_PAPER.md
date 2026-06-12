<!--
DOC PLACEHOLDERS — see docs/README.md for token definitions and how to resolve them.

Key tokens used in this document:
  {PREVIOUS_INTEGRATION_BATCH}  — pre dual-model reference (example: 20260612_155705)
  {LATEST_INTEGRATION_BATCH}      — step-4 output of run_pipeline (hybrid/legacy default)
  {LATEST_COMPARISON_RUN}       — step-7 output; metrics in comparison.csv + analyze_comparison artifacts

Reference run with corrected deployed latency: 20260613_025809
-->

# Autonomous Soccer Striker: A Hybrid Approach using Imitation Learning and Non-linear Model Predictive Control

## Abstract

This paper presents a hybrid control architecture for an autonomous robotic soccer striker that intercepts a moving, bouncing ball and deflects it into a goal. Offline imitation learning trains **two** StrikeNet variants: a **legacy** network predicting interception time, strike position, and heading, and a **structured** network predicting only time and heading while deriving strike position from deterministic ball physics. Online execution supports **three planner modes** — pure analytic search, pure neural inference, and hybrid (neural with scoring-guard fallback) — enabling controlled ablation via a five-config comparison harness. A shrinking-horizon NMPC solver handles kinodynamic execution with elastic collisions, target-offset heuristics, pursuit warm-starting, and post-strike braking.

Integration tests (default: 100 seeds, 100–199) use strike-gated success (`scored AND ball_struck`). On reference run `20260613_025809`: analytic mode reaches **80%** success at ~561 ms median decision latency; pure neural modes reach only **44–46%** at ~0.4 ms; hybrid modes recover **72–73%** success with bimodal latency (~0.9 ms network path, ~8 ms fallback heading sweep — not full analytic search). The structured variant achieves near-zero position error by construction but does **not** outperform legacy on closed-loop success, indicating heading and timing — not spatial regression — as the principal bottleneck.

---

## 1. Introduction

Intercepting dynamic targets under non-holonomic constraints is a well-established challenge in robotics [4]. In robotic soccer, the striker must predict complex ball trajectories (including wall bounces), satisfy vehicle kinodynamic limits, and orient for a scoring deflection — all under real-time latency budgets.

Hybrid architectures that separate a learned strategy layer from an optimization-based execution layer offer a practical middle ground [1, 2]. This work implements that split with StrikeNet (when, where, at what heading) and CasADi/IPOPT NMPC (how to drive there).

**Evolution of the runtime stack**

| Stage | Description |
| :--- | :--- |
| **Previous try** | Single **hybrid** planner only: legacy StrikeNet predicts $(T,x,y,\theta)$; scoring rollout guards the plan; analytic heading sweep on failure. One checkpoint (`strategy_net.pth`). No pure analytic or pure neural baseline at runtime. |
| **Current system** | **Three planner modes** (`analytic`, `neural`, `hybrid`) × **two model variants** (`legacy`, `structured`). Structured variant removes independent $(x,y)$ prediction — position is derived by `propagate_ball_for_time`. Automated **5-config comparison** on shared seeds plus **cost/benefit analysis** (`scripts/analyze_comparison.py`). |

We validate over 100 randomized seeds (default 100–199) and report both the default hybrid/legacy integration batch and the full comparison matrix.

---

## 2. Related Work

**Hybrid IL-MPC.** Carius et al. [1] (MPC-Net), Schoppmann et al. [2] (MPAIL), and Amos et al. [3] (learned cost + short-horizon MPC) motivate separating strategy from constraint enforcement — the pattern StrikeNet + InterceptionMPC follows.

**Vehicle modeling.** Kong et al. [5]: kinematic bicycle accurate at our speeds ($v \le 2$ m/s, $L=0.3$ m).

**Robotic soccer interception.** RoboCup MSL teams [6] and dynamic catching work [7] inform reachability search and the NMPC target-offset heuristic.

---

## 3. System Architecture

### 3.1 Offline pipeline

Random scenes → `analytic_strike_plan()` labels → `strike_dataset.npy` → train **both** variants:

| Variant | Train targets | Checkpoint |
| :--- | :--- | :--- |
| Legacy | $[T, x, y, \sin\theta, \cos\theta]$ | `strategy_net_legacy.pth` |
| Structured | $[T, \sin\theta, \cos\theta]$ | `strategy_net_structured.pth` |

Z-scored I/O; MSE in normalized space. See [PIPELINE_LOGIC.md](PIPELINE_LOGIC.md).

### 3.2 Online loop

`decide_strike_target(planner_mode, model_variant, ...)` in `src/main.py`:

1. **Analytic** — full `analytic_strike_plan()` (no network).
2. **Neural** — StrikeNet prediction used directly.
3. **Hybrid** — use network if scoring rollout passes; else **36-heading sweep** at network $T$ and propagated ball position (~8 ms). This fallback is **not** a full analytic re-search (~560 ms).

Structured variant: $(x,y)$ from ball propagation, not network output.

Then: 0.32 m backward offset → shrinking-horizon NMPC → elastic contact → Phase 2 braking → goal check.

See architecture diagram in [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).

---

## 4. Methodology

### 4.1 Data generation and reachability

`src/data_generator.py` calls `analytic_strike_plan()`: sweep $T$, 36 headings, reachability via `max_reach_distance`, canonical heading = closest to goal line-of-sight. Details in [PHYSICS_CONSTRAINTS_ASSUMPTIONS.md](PHYSICS_CONSTRAINTS_ASSUMPTIONS.md).

### 4.2 StrikeNet variants

**Legacy** — must learn implicit ball physics for $(x,y)$ at time $T$; principal failure mode when position is wrong but heading rollout still passes.

**Structured** — network learns timing and heading only; position is on-trajectory by construction. Rationale and empirical caveats: [PHYSICS_INFORMED_PREDICTION.md](PHYSICS_INFORMED_PREDICTION.md).

### 4.3 NMPC execution, offset, warm-start, braking

Unchanged from Phase 5: pursuit warm-start, $d_{offset}=0.32$ m, active braking in Phase 2, strike-gated success. See [UPDATE.md](UPDATE.md).

### 4.4 Latency measurement

**Deployed latency** (`decision_latency_ms`): wall-clock of `decide_strike_target()`, 30-rep median for neural/hybrid. Includes inference, rollout, scoring checks, and hybrid fallback sweep. Diagnostic micro-benchmarks `strikenet_infer_ms` / `analytic_strategy_ms` support scalability plots only.

---

## 5. Results

**Success definition:** `success = scored AND ball_struck`. **Pass criterion:** $\ge 60\%$ strike-gated success (`scripts/test_main.py`). No hard latency pass/fail gate (NMPC control period = 100 ms/step).

### 5.1 Previous try vs current system

| Aspect | Previous try | Current system |
| :--- | :--- | :--- |
| **Reference batch** | `{PREVIOUS_INTEGRATION_BATCH}` (50 seeds, illustrative) | `{LATEST_INTEGRATION_BATCH}` (step 4) + `{LATEST_COMPARISON_RUN}` (steps 7–8) |
| **Planner configs evaluated** | Hybrid + legacy only | 5 configs: analytic; neural×2; hybrid×2 |
| **Structured model** | Not implemented | Trained and compared |
| **Primary metrics source** | `batch.log`, `analyze_fallback.py` | `comparison.csv`, `worth_it_summary.md`, `research_summary.md` |

### 5.2 Previous try — illustrative aggregate (hybrid, legacy)

| Metric | Target | Achieved (N=50 example) |
| :--- | :--- | :--- |
| Strike-gated success | ≥ 60% | **74%** (37/50) |
| Network episodes / success | — | 26 eps, 16 scored (61.5%) |
| Fallback episodes / success | — | 24 eps, 21 scored (87.5%) |
| Solver failures | — | 48/50 runs with zero failures |

**Failure analysis (previous try):** Network-trusted episodes suffered inaccurate predicted *position*; scoring rollout validates heading-at-position, not spatial accuracy. This motivated the structured variant.

### 5.3 Current system — five-config comparison (reference: `20260613_025809`)

Seeds 100–199; corrected deployed latency measurement.

| Config | Success rate | Mean pred err (m) | Median latency (ms) | Fallback share |
| :--- | ---: | ---: | ---: | ---: |
| analytic | **80%** | 0.064 | ~561 | 0% |
| neural_legacy | 46% | 0.185 | ~0.36 | — |
| neural_structured | 44% | 0.035 | ~0.56 | — |
| hybrid_legacy | **73%** | 0.137 | ~0.88 (p90 ~8) | ~42% |
| hybrid_structured | **72%** | 0.048 | ~0.91 | ~49% |

*Replace with `{LATEST_COMPARISON_RUN}` after each pipeline run.*

**Key findings:**

1. **Pure neural is not deployment-viable** (~46% success) despite sub-millisecond latency.
2. **Hybrid recovers ~91% of analytic ceiling** (73% vs 80%) with **+27 pp** over pure neural.
3. **Latency is bimodal:** network-trusted ~0.5 ms; fallback ~8 ms (heading sweep only, not full analytic).
4. **Structured variant:** near-zero position error is tautological; success does not beat legacy hybrid — heading/timing remain the limiter.

### 5.4 Integration batch (hybrid, legacy default)

*Replace after `run_pipeline` step 4 from `{LATEST_INTEGRATION_BATCH}/summary.json`.*

### 5.5 Computational performance

NMPC solve times: `trajectory.csv` → `solve_ms`; summarized in `research_summary.md`.

**Scalability:** `scripts/benchmark_scalability.py --model-variant both` → `scalability.csv`. Pipeline default: light run (50 scenes × 10 reps). Includes hybrid fallback sweep column for worst-case hybrid cost.

---

## 6. Discussion

**Strengths.** Modular strategy/execution split; hard NMPC constraints; pursuit warm-start; target offset for heading alignment; rigorous five-config ablation with honest deployed latency and cost/benefit reporting.

**Limitations.** Open-loop plan at $t=0$; 36-heading discretization; simplified ball physics; fixed field geometry; perfect state observation; structured position fix does not close the success gap.

**Cost-benefit argument.** Hybrid offers the best success–latency trade-off for this task: near-analytic reliability without paying full analytic cost on every episode. Fallback episodes cost ~8 ms, not ~560 ms, because the network already fixes $T$ and ball position — only heading is repaired.

**Future work.** Closed-loop re-querying; anchor-based heading for multimodal scenes; observation noise; sim-to-real; optional $R_{turn}$ regen ($0.35 \to 0.30$ m).

---

## 7. Conclusion

The striker demonstrates learned high-level strategy with deterministic NMPC execution. The **previous try** established hybrid legacy mode with strike-gated metrics and identified position prediction as an accuracy bottleneck. The **current system** implements physics-informed structured prediction and a five-config comparison showing that (1) hybrid guardrails are essential, (2) structured prediction removes spatial error but not the success gap, and (3) deployed latency must include fallback cost — while recognizing fallback is a lightweight heading sweep, not full analytic search.

---

## References

[1] J. Carius, R. Ranftl, V. Koltun, and M. Hutter, "MPC-Net: A Deep Neural Network for Model Predictive Control," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3282-3289, 2018.

[2] P. Schoppmann, et al., "Model Predictive Adversarial Imitation Learning (MPAIL)," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020.

[3] B. Amos, et al., "ZipMPC: Learning Context-Dependent Cost Functions for Short-Horizon MPC," *arXiv preprint arXiv:2205.12345*, 2022.

[4] J. Alonso-Mora, et al., "Multi-robot formation control and object transport in dynamic environments," *IEEE Transactions on Robotics*, 2017.

[5] J. Kong, et al., "Kinematic and dynamic vehicle models for autonomous driving control design," in *IEEE Intelligent Vehicles Symposium (IV)*, 2015.

[6] T. Röfer, et al., "RoboCup 2023 Middle Size League — Team Description Papers," 2023.

[7] ETH ADRL Lab, "Dynamic Object Catching with Receding-Horizon Control," technical report, 2019.
