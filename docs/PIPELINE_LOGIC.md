<!--
DOC PLACEHOLDERS — see docs/README.md for token definitions and how to resolve them.
-->

# Pipeline Logic — Phase 5 Striker

## Offline pipeline: data to models

```mermaid
flowchart TD
  A[Sample Random Scene] --> B{Reachability Sweep T}
  B -- Reachable --> C{Sweep 36 θ for Goal}
  C -- "≥1 scores" --> D["Pick scoring θ closest to goal LoS"]
  C -- None score --> E[Try next T / Reject]
  D --> F[strike_dataset.npy]
  F --> G1[Train legacy: T,x,y,sin,cos]
  F --> G2[Train structured: T,sin,cos]
  G1 --> H1[strategy_net_legacy.pth]
  G2 --> H2[strategy_net_structured.pth]
```

### 1. Data generation (`python -m src.data_generator`)

Label search is in `src/planner.py` as `analytic_strike_plan()`. Reachability uses `max_reach_distance(T)` with $a_{max}=2.0$ m/s², $v_{max}=2.0$ m/s, and turn-arc penalty $R_{turn}=0.35$ m (legacy default; exact $0.30$ m deferred — see [PHYSICS_INFORMED_PREDICTION.md](PHYSICS_INFORMED_PREDICTION.md)).

Per sample:
1. Random ball and car state.
2. For $T \in [0.5, 5.0]$ s (step $0.05$ s): propagate ball, sweep 36 headings, keep scoring **and** reachable candidates.
3. First feasible $T$; canonical heading = closest to goal line-of-sight.
4. Save 11 columns: `[ball_x, ball_y, ball_vx, ball_vy, car_x, car_y, car_theta, T, x, y, theta]`.

### 2. Training (`python -m src.network --variant {legacy|structured|both}`)

| Variant | Training targets | Checkpoint | Log |
| :--- | :--- | :--- | :--- |
| `legacy` | $[T, x, y, \sin\theta, \cos\theta]$ | `models/strategy_net_legacy.pth` | `data/training/training_log_legacy.csv` |
| `structured` | $[T, \sin\theta, \cos\theta]$ | `models/strategy_net_structured.pth` | `data/training/training_log_structured.csv` |

* Architecture: `Input(7) → 128 → 128 → 64 → Output(5 or 3)`.
* Z-scored input and output (train-split statistics in registered buffers).
* MSE in normalized space; early stopping (patience 20).

---

## Online loop: `decide_strike_target()` + two-phase execution

Entry point: `run_simulation(planner_mode, model_variant, ...)` in `src/main.py`.

### Strike target selection

```
decide_strike_target(planner_mode, model, model_variant, input7, ...)
```

| Mode | Logic |
| :--- | :--- |
| **analytic** | `analytic_strike_plan()` → $(T, x, y, \theta)$. If infeasible: ball at $T=2$s fallback. |
| **neural** | `model.predict()` → use prediction directly (`target_source = "network"`). |
| **hybrid** | Predict → scoring rollout; if pass use network, else heading sweep at propagated ball position (`target_source = "fallback"`). |

**Structured variant:** after predicting $(T, \theta)$, $x,y$ come from `propagate_ball_for_time` to $T_{final}$ — not from the network.

**Legacy variant:** $(x, y)$ predicted and clipped to field; `net_vs_analytic_pos_m` logged when `target_source == "network"`.

### Phase 1: NMPC interception

1. $N_{steps} = \mathrm{clip}(\mathrm{round}(T / \Delta t), 1, 50)$.
2. Offset target: $\mathbf{q}_{strike} = [x - 0.32\cos\theta,\ y - 0.32\sin\theta,\ \theta,\ v_{impact}]$.
3. Shrinking-horizon `InterceptionMPC` until contact (`dist < 0.35$ m) or horizon exhausted.

### Phase 2: Post-strike

Up to 80 steps with active braking; early exit on `scored`.

```mermaid
sequenceDiagram
  participant Main as main.py
  participant Decide as decide_strike_target
  participant Net as StrikeNet
  participant BP as ball_physics
  participant MPC as InterceptionMPC
  participant World as World

  Main->>Decide: planner_mode + model_variant
  alt analytic
    Decide->>Decide: analytic_strike_plan
  else neural or hybrid
    Decide->>Net: predict(inputs)
    Net-->>Decide: T (+ x,y,theta or T,theta)
    Decide->>BP: propagate_ball_for_time (structured or hybrid check)
    alt hybrid and plan does not score
      Decide->>BP: 36-heading sweep
    end
  end
  Decide-->>Main: strike target + target_source
  Main->>MPC: solve loop
  MPC->>World: step
```

### Latency metadata

Per run, `metadata.json` records:
* `strikenet_infer_ms`, `rollout_ms` (structured only), `decision_latency_ms`
* `analytic_strategy_ms` (timed reference, not control path unless `analytic` mode)
* `planner_mode`, `model_variant`

---

## Testing and reporting pipelines

### Integration test (`scripts/test_main.py`)

```powershell
python scripts/test_main.py --planner-mode hybrid --model-variant legacy
python scripts/test_main.py --planner-mode analytic --no-video
```

* Default: 100 seeds (100–199), hybrid + legacy.
* Writes `summary.json` per batch (success rate, `mean_pred_err_m`, `mean_decision_latency_ms`, network/fallback counts).
* **Pass:** strike-gated success $\ge 60\%$.

### Comparison harness (`scripts/compare_modes.py`)

Runs five configs on shared seeds (videos off by default):

| Config folder | Mode | Variant |
| :--- | :--- | :--- |
| `analytic` | analytic | — |
| `neural_legacy` | neural | legacy |
| `neural_structured` | neural | structured |
| `hybrid_legacy` | hybrid | legacy |
| `hybrid_structured` | hybrid | structured |

Outputs under `data/tests/comparison/{LATEST_COMPARISON_RUN}/` and `data/reports/plots/comparison/{run}/`.

### Other scripts

| Script | Role |
| :--- | :--- |
| `scripts/generate_plots.py` | Per-batch trajectory/error plots |
| `scripts/analyze_results.py` | `research_summary.md`; surfaces `planner_mode` / `model_variant` |
| `scripts/analyze_fallback.py` | Hybrid-only network vs fallback (graceful skip otherwise) |
| `scripts/benchmark_scalability.py` | `--model-variant both` for legacy vs structured latency |
| `scripts/test_network.py` | Sanity check both variants on dataset samples |

### Full pipeline (`run_pipeline.ps1` / `run_pipeline.sh`)

1. Data generation  
2. Train both variants (`--variant both`)  
3. Network sanity check  
4. Integration test (hybrid/legacy, optional `-NoVideo`)  
5. Plots + `analyze_results` + `analyze_fallback`  
6. Scalability benchmark (`--model-variant both`)  
7. `compare_modes.py`
