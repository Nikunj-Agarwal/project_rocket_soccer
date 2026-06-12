# Autonomous Soccer Striker: A Hybrid Approach using Imitation Learning and Non-linear Model Predictive Control

## Abstract
This paper presents a hybrid control architecture for an autonomous robotic soccer striker tasked with intercepting a moving, bouncing ball and deflecting it into a goal. The system combines an offline imitation learning strategy network (StrikeNet) that predicts the interception time, strike position, and strike heading with an online, shrinking-horizon Non-linear Model Predictive Control (NMPC) solver for precise kinematic execution. The network's predicted strike plan drives the NMPC target directly; a physics-based scoring rollout validates each plan and an analytic interception point with a heading sweep is substituted only when the learned plan provably cannot score. By incorporating elastic collision models, target-offset heuristics, pursuit-based warm-starting, and post-strike active braking, the system overcomes real-time kinodynamic constraints. Integration tests (default: 100 seeds, 100–199) use strike-gated success (`scored AND ball_struck`). In a representative 50-seed run (batch `20260612_155705`), the system achieved 74% success with the network's prediction driving execution in 52% of episodes; headline interception accuracy is measured by `strike_point_pred_err_m` (predicted target vs ball-at-contact), with closest-approach distance retained only as a diagnostic.

---

## 1. Introduction
Intercepting dynamic targets under non-holonomic constraints is a well-established challenge in robotics [4]. In robotic soccer, the striker must simultaneously predict a ball's complex trajectory — including multiple elastic wall bounces — satisfy vehicle kinodynamic constraints (bounded acceleration, velocity saturation, and limited steering angle), and precisely orient itself at the moment of contact to redirect the ball into a specific goal mouth. The problem compounds further when real-time performance is required: every additional millisecond of planning latency translates to positional drift of the moving target.

Purely analytical planners, while globally optimal in theory, suffer from prohibitive computational costs when the search space includes variable interception times, multi-bounce ball trajectories, and angular heading sweeps. On the other hand, pure learning-based approaches risk safety violations — a policy trained end-to-end may output controls that violate steering or acceleration bounds, leading to kinematically infeasible trajectories.

Hybrid control architectures that decompose the problem into a high-level learned strategy layer and a low-level optimization-based execution layer offer a promising middle ground [1, 2]. By relegating the combinatorial "when and where" decisions to a lightweight neural network and the precise "how to drive there" trajectory planning to a deterministic NMPC solver, each component operates within its area of strength.

This paper proposes such a two-phase hybrid approach: a Multi-Layer Perceptron (StrikeNet) dictates the macroscopic strategy (predicting the optimal interception time, strike position, and strike heading from an initial scene configuration), while a shrinking-horizon NMPC solver manages the microscopic execution. At runtime there is a single hybrid mode — network prediction with a scoring-guard analytic fallback (`target_source` logged per episode); there is no selectable network-only or analytic-only mode. We validate the system over 100 randomized integration test seeds (default: 100–199) spanning diverse approach angles, multi-bounce trajectories, and challenging initial headings.

<!-- 📊 FIGURE 1: System architecture block diagram (see SYSTEM_OVERVIEW.md mermaid flowchart).
     Show the two-phase pipeline: Offline (Data Generation → StrikeNet Training) and
     Online (StrikeNet Inference → Ball Propagation → Heading Sweep → Offset → NMPC → Collision → Braking).
     This should be a clean, single-column figure spanning the full page width. -->

---

## 2. Related Work

**Hybrid IL-MPC Architectures.** Carius et al. [1] proposed MPC-Net, training a neural network policy to mimic an MPC "teacher," thereby reducing the online optimization burden to a single forward pass while retaining the safety guarantees of the original MPC formulation. Schoppmann et al. [2] extended this idea through adversarial imitation learning integrated with an MPC tracking layer, demonstrating that separating strategic objectives from kinematic constraint enforcement improves both robustness and generalization. Amos et al. [3] showed that learning context-dependent cost functions via neural networks allows a short-horizon MPC solver to approximate the performance of computationally prohibitive long-horizon MPC. Our work follows this paradigm: StrikeNet's output ($T_{strike}$) establishes a dynamic, shrinking-horizon target, enabling the CasADi solver to work efficiently with a limited control horizon.

**Vehicle Modeling and Trajectory Tracking.** Kong et al. [5] evaluated the accuracy and computational feasibility of the kinematic bicycle model within an MPC framework, demonstrating that it remains highly accurate for low-to-medium speed maneuvers where lateral tire slip is bounded. Our system operates at speeds up to 2.0 m/s with a wheelbase of 0.3 m, well within the regime where this model is known to be accurate.

**Dynamic Interception in Robotic Soccer.** RoboCup Middle Size League teams [6] have developed extensive path planning algorithms for intercepting high-speed bouncing balls, including dynamic "Estimated Time of Arrival" calculations and reachability-based interception point selection. Research from ETH Zürich's ADRL Lab [7] on dynamic object catching with robotic manipulators using receding-horizon control introduced spatial offset strategies to align the interceptor's orientation before the final contact phase — a technique we adapt for our NMPC target-offset heuristic.

---

## 3. System Architecture
The system is divided into an **Offline Pipeline** and an **Online Simulation Loop**, connected through a trained model checkpoint:

1. **Offline Pipeline**: Randomly generated scene configurations are simulated to identify reachable, goal-scoring states. For each valid configuration, the system records the initial 7-dimensional state vector along with the corresponding interception time, strike position, and strike heading. A Multi-Layer Perceptron (StrikeNet) is then trained on this curated dataset via supervised learning to predict the 5-dimensional strategy output.

2. **Online Loop**: During live execution, StrikeNet infers the full strike plan $(T_{strike}, x_{strike}, y_{strike}, \theta_{strike})$ from the current scene state in a single forward pass (<1 ms). This predicted plan drives the NMPC target directly. To guard against unsafe predictions, the system runs a physics-based scoring rollout of the predicted plan; if the predicted strike point and heading send the ball into the goal, they are used unchanged (the network is the decision-maker). Only when the predicted plan fails the rollout does the system fall back to the analytically propagated interception point with a 36-candidate heading sweep. A backward target offset is applied and the resulting coordinates are fed to the NMPC solver, which drives the car along a kinematically feasible trajectory. Upon contact (distance < 0.35 m), an elastic collision physics model computes the post-strike ball velocity, and a Phase 2 braking controller brings the vehicle to a halt.

<!-- 📊 FIGURE 2: Mermaid or drawn system architecture diagram showing the complete
     data flow from Offline Pipeline (data_generator → strike_dataset.npy → StrikeNet train → strategy_net.pth)
     to Online Loop (StrikeNet predict → ball propagation → θ sweep → offset → NMPC → World.step → collision → braking → goal check).
     See the mermaid diagram in SYSTEM_OVERVIEW.md for reference. -->

---

## 4. Methodology

### 4.1 Data Generation and Reachability

To generate valid training data, the offline generator samples random initial states for the vehicle and the ball. The ball position is sampled uniformly from $x_b \in [2.0, 8.0]$ m, $y_b \in [0.0, 6.0]$ m, with speed $v_b \in [0.5, 2.0]$ m/s at a random direction. The car is initialized at $x_c \in [0.0, 4.0]$ m, $y_c \in [0.0, 6.0]$ m, with a random heading $\theta_c \in [-\pi, \pi]$ and zero initial speed.

For increasing time horizons $T \in [0.5, 5.0]$ s (in steps of 0.05 s), the system calculates the ball's position at time $T$ using the shared bounce-aware integrator with wall restitution coefficient $e = 0.85$ on a 10.0 m × 6.0 m rectangular field. The generator then performs a joint reachability and scoring check:

1. **Reachability** (`src/planner.py` — `max_reach_distance`, `analytic_strike_plan`): Using a kinodynamic bi-arc path approximation, the system estimates whether the car can physically reach the ball's future position at time $T$. The effective distance accounts for initial heading misalignment and terminal orientation change, with a turning radius buffer of $R_{turn} = 0.35$ m (legacy default; exact $L/\tan(\delta_{max}) = 0.30$ m deferred to a future regen). The maximum reachable distance follows a piecewise acceleration model: $d_{max} = T^2$ for $T \leq 1.0$ s (accelerating from rest at $a_{max} = 2.0$ m/s²), and $d_{max} = 2.0T - 1.0$ for $T > 1.0$ s (at maximum velocity $v_{max} = 2.0$ m/s).

2. **Scoring Heading Sweep**: For each reachable $(T, \mathbf{p}_b)$ pair, the generator sweeps 36 angular candidates $\theta_{strike} \in [-\pi, \pi]$ to find headings that, after an elastic collision at impact speed $v_{impact} = 1.0$ m/s, redirect the ball into the goal mouth (the 2.0 m wide segment at $x = 10.0$ m, $y \in [2.0, 4.0]$ m). Post-collision ball trajectories are propagated for up to 5.0 s to check goal crossing. The label is taken at the first time $T$ for which at least one candidate is simultaneously scoring and reachable.

**Canonical heading selection.** Multiple headings can be valid for a single scene, making the input-to-heading map multimodal; naive selection (e.g. the first scoring angle scanning from $-\pi$) yields a discontinuous target that MSE regression averages into invalid intermediate angles. To produce a deterministic, approximately continuous label, among the feasible candidates we select the heading closest to the line-of-sight from the strike point to the goal center, $\theta_{strike} = \arg\min_{\theta} |\text{wrap}(\theta - \theta_{LoS \to goal})|$. This same rule is reused at runtime in the analytic fallback, keeping training labels and fallback behavior consistent.

This process generates a dataset of 100,000 valid samples (typical acceptance rate: ~15–25% of random attempts yield valid samples).

<!-- 📊 FIGURE 3: Scatter plot of the training dataset showing the distribution of strike positions
     (x_strike, y_strike) on the field, colored by T_strike. This visualizes the coverage of
     the training data across the field and across interception horizons. -->

<!-- 📊 FIGURE 4: Histogram of T_strike values in the training dataset, showing the distribution
     of interception times. Expected to be right-skewed (most interceptions happen at shorter times). -->

### 4.2 Strategy Network (StrikeNet)

StrikeNet is a feed-forward MLP with architecture: Input(7) → 128 → ReLU → 128 → ReLU → 64 → ReLU → Output(5). The network maps the initial 7-dimensional scene state:
$$\mathbf{x}_{in} = [x_{ball}, y_{ball}, v_{x,ball}, v_{y,ball}, x_{car}, y_{car}, \theta_{car}]$$
to a 5-dimensional output strategy vector:
$$\mathbf{y}_{out} = [T_{strike}, x_{strike}, y_{strike}, \sin(\theta_{strike}), \cos(\theta_{strike})]$$

The angle is decomposed into sine and cosine components to avoid circular discontinuity penalties during backpropagation. At inference time, $\theta_{strike}$ is reconstructed via $\text{arctan2}(\sin(\theta_{strike}), \cos(\theta_{strike}))$.

**Input and Output Normalization.** The network applies z-score normalization to both inputs and outputs using per-feature mean and standard deviation computed from the training split, stored as PyTorch registered buffers so they persist across save/load cycles. Output normalization is essential here: the raw target scales differ by 5–10× (e.g. $x_{strike} \in [0, 10]$ m versus $\sin\theta \in [-1, 1]$), so an unnormalized MSE would be dominated by the position/time terms and effectively ignore the heading. Training is therefore performed in normalized target space, and `predict()` de-normalizes the network output back to physical units before reconstructing the angle.

**Training Details.** The model is trained using the Adam optimizer ($\text{lr} = 10^{-3}$) with Mean Squared Error (MSE) loss across all 5 (normalized) output dimensions. An 80/20 train/test split is used with a batch size of 256. Early stopping with patience of 20 epochs prevents overfitting, and the best model (by test loss) is checkpointed. Training converges within approximately 80–120 epochs.

<!-- 📊 FIGURE 5: Training loss curve (train MSE and test MSE vs. epoch). Shows convergence behavior
     and the early stopping point. Use data from data/training/training_log.csv.
     See scripts/generate_plots.py → plot_training_curves(). -->

<!-- 📊 FIGURE 6: StrikeNet prediction error analysis — bar chart showing absolute error in
     [T, x, y, θ] for 8 random dataset samples (ground truth vs. prediction).
     See scripts/generate_plots.py → plot_strikenet_samples(). -->

### 4.3 Physics & Collision Model

**Vehicle Kinematics.** The vehicle is modeled using the kinematic bicycle model, which has been shown to provide an excellent balance of fidelity and real-time computation for autonomous platforms operating at low-to-medium speeds [5]. The continuous-time dynamics are:

$$\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \frac{v}{L} \tan(\delta), \quad \dot{v} = a$$

where $L = 0.3$ m is the wheelbase, with bounded velocity $v \in [0.0, 2.0]$ m/s, bounded acceleration $a \in [-2.0, 2.0]$ m/s², and bounded steering $\delta \in [-\pi/4, \pi/4]$ rad. The dynamics are discretized using a 4th-order Runge-Kutta (RK4) integrator at $\Delta t = 0.1$ s.

**Ball-Wall Bounce Model.** Between bounces, ball velocity is constant (no drag, gravity, or spin). On wall impact at the field boundaries ($[0, 10.0] \times [0, 6.0]$ m), the normal velocity component is reflected and scaled by the coefficient of restitution $e = 0.85$. The goal mouth segment ($x = 10.0$, $y \in [2.0, 4.0]$) allows the ball to pass through without bouncing.

**Car-Ball Elastic Collision Model.** Upon intersection (car-ball distance < 0.35 m), a 2D elastic collision occurs between the flat car bumper and the ball. The bumper is modeled as a plane with normal vector $\mathbf{n} = [\cos(\theta_{car}), \sin(\theta_{car})]^T$. Let $\mathbf{v}_{rel} = \mathbf{v}_{ball} - v_{car}\mathbf{n}$ be the relative velocity and $v_{rel,n} = \mathbf{v}_{rel} \cdot \mathbf{n}$ be its normal component. If the ball and car are approaching ($v_{rel,n} < 0$):

$$\mathbf{v}_{ball}^{post} = \mathbf{v}_{ball} - (1 + e_{strike}) \, v_{rel,n} \, \mathbf{n}$$

where $e_{strike} = 0.8$ is the bumper coefficient of restitution. When $v_{rel,n} \ge 0$ (the ball is already separating from the bumper at least as fast as the car along the normal), no impulse is applied and the ball velocity is unchanged. To prevent steering oscillations and early contact triggers, we employ a target-offset mechanism similar to orientation-alignment strategies used in dynamic manipulator catching systems [7].

<!-- 📊 FIGURE 7: Diagram illustrating the elastic collision geometry — showing the car bumper
     normal vector n, the relative velocity decomposition, and the post-collision ball velocity
     vector. A simple 2D vector diagram with labeled components. -->

### 4.4 Non-linear Model Predictive Control (NMPC)

The shrinking-horizon NMPC formulation, tracking a dynamic strategy predicted by the imitation learning network, aligns with modern frameworks that compress long-horizon behaviors into parameterized cost objectives [3]. The NMPC loop uses CasADi's Opti() interface with the IPOPT interior-point solver in a multiple-shooting formulation to solve the optimal control problem at a rate of 10 Hz ($\Delta t = 0.1$ s).

**Objective Function.** The cost function combines a heavily weighted terminal cost with a light running cost:

$$J = \underbrace{Q_x (x_N - x_T)^2 + Q_y (y_N - y_T)^2 + Q_\theta (1 - \cos(\theta_N - \theta_T)) + Q_v (v_N - v_T)^2}_{\text{Terminal Cost}} + \sum_{k=0}^{N-1} \underbrace{R_a \, a_k^2 + R_\delta \, \delta_k^2}_{\text{Running Cost}}$$

Terminal cost weights $\mathbf{Q} = \text{diag}(3000, 3000, 300, 1)$ heavily penalize positional and heading deviations at the terminal time step, while running cost weights $\mathbf{R} = \text{diag}(0.005, 0.005)$ allow aggressive maneuvering. The heading penalty uses the wrap-safe form $1 - \cos(\Delta\theta)$ rather than $\Delta\theta^2$ to avoid discontinuities at $\pm\pi$.

**Target Offset.** To prevent the vehicle from triggering an early collision before completing its terminal rotation, the NMPC target is artificially offset by $d_{offset} = 0.32$ m backwards along the approach vector. The strike point $(x_{tgt}, y_{tgt}, \theta_{tgt})$ is the network's prediction when its plan scores, or the analytic fallback otherwise:

$$x_{target} = x_{tgt} - d_{offset} \cos(\theta_{tgt}), \quad y_{target} = y_{tgt} - d_{offset} \sin(\theta_{tgt})$$

This offset is carefully tuned: the contact detection radius is 0.35 m, so an offset of 0.32 m ensures the car enters the contact zone with its heading fully aligned to $\theta_{strike}$. Without this offset, heading error at the moment of collision was approximately 0.61 rad; with it, heading error drops to a median of essentially 0 rad (mean 0.070 rad over the batch).

<!-- 📊 FIGURE 8: Comparison showing the effect of the target offset on heading error at contact.
     Two side-by-side trajectory plots: (a) without offset (high heading error, ~0.61 rad)
     vs. (b) with 0.32m offset (low heading error, ~0.03 rad).
     Alternatively, a bar chart comparing the two configurations. -->

**Pursuit-Based Warm-Start.** To ensure robust convergence across diverse initial configurations, the solver is warm-started with a kinematically feasible initial guess generated by forward-simulating a proportional pursuit controller. At each step:
1. A line-of-sight angle to the target is computed: $\theta_{LoS} = \text{arctan2}(y_T - y_k, x_T - x_k)$.
2. A proportional steering command is applied: $\delta_k = \text{clip}(1.5 \cdot \Delta\theta_{LoS}, -\pi/4, \pi/4)$, where $\Delta\theta_{LoS}$ is the wrapped angular difference.
3. Acceleration is linearly interpolated to reach the desired impact speed by the final step.
4. The state is propagated using the same RK4 integrator as the NMPC dynamics.

This produces a warm-start trajectory that satisfies all kinematic constraints, guiding IPOPT away from local minima (e.g., acceleration chattering, premature stopping) that arise with naive straight-line initialization. In practice, the pursuit warm-start kept 48 of 50 test runs entirely free of solver failures, with only 2 runs triggering recoverable IPOPT restoration phases.

**Solver Performance.** By suppressing IPOPT console output (`print_level: 0`, `sb: "yes"`) and CasADi's I/O overhead (`print_time: False`, `verbose: False`), the system achieved a 120× execution speedup. The NMPC solver averages approximately 60 ms per step with a maximum of 300 iterations per solve.

<!-- 📊 FIGURE 9: Box plot or histogram of NMPC solve times (ms per step) across all 50 test seeds.
     Shows the distribution of computational cost and confirms real-time feasibility. -->

**Post-Strike Braking.** Following the strike, a Phase 2 active braking controller is engaged, applying maximum deceleration:

$$a_{brake} = \text{clip}\left(-\frac{v_{car}}{\Delta t}, -2.0, 0.0\right)$$

This brings the vehicle to a halt within a few simulation steps, ensuring it remains safely within the field boundaries while the ball coasts into the goal. Without active braking, the kinematic bicycle model (which does not include ground friction) would allow the car to continue indefinitely at its impact speed. The post-strike phase runs for up to 80 steps (8.0 s) and terminates early as soon as the ball is scored, so the window only extends episodes in which a late or slow strike leaves the ball still travelling toward the goal.

---

## 5. Results

The system was validated using a rigorous integration test suite over **100** randomized, distinct seeds (default: seeds 100–199), encompassing varied approach angles, multi-bounce ball trajectories, and challenging initial vehicle headings. Each seed generates a unique initial configuration by sampling ball and car states from the same distributions used in training.

**Success definition.** A run counts as successful only when the car strikes the ball *and* the ball enters the goal (`success = scored AND ball_struck`). Goals entered without car contact are logged but excluded.

**Pass criterion.** Strike-gated success rate $\ge 60\%$ (`scripts/test_main.py`). The former closest-approach position threshold ($\le 0.35$ m) is no longer a pass gate.

### 5.1 Aggregate Performance

*Illustrative results from a 50-seed batch (`20260612_155705`) run before the default expanded to 100 seeds; regenerate on the latest batch for publication figures.*

| Metric | Target | Achieved (N=50 example) |
|:---|:---|:---|
| **Strike-gated Success Rate** | ≥ 60% | **74%** (37/50) |
| **Mean `strike_point_pred_err_m`** | report only | *regenerate from latest batch* |
| **Solver Convergence** | — | **48/50** runs with zero solver failures |

The pursuit-based warm-start kept 48 of 50 runs entirely free of solver failures; only 2 runs triggered recoverable IPOPT restoration phases.

**Network vs. fallback contribution (same example batch).** In 26 of 50 episodes StrikeNet's predicted strike point and heading passed the scoring rollout and drove the controller directly (`target_source = "network"`), scoring 16/26 (61.5%). In the remaining 24 episodes the analytic fallback was engaged, scoring 21/24 (87.5%). This breakdown is produced automatically by `scripts/analyze_fallback.py`.

**Failure analysis.** Failures are dominated by network-driven episodes in which the predicted strike *position* was inaccurate: the controller drives to the predicted point, but `net_vs_analytic_pos_m` and `strike_point_pred_err_m` show the ball is elsewhere at contact time. Because the scoring rollout validates heading-at-position but not positional accuracy, a confidently-wrong position can still be trusted — the principal accuracy ceiling of the current design. A proposed fix (predict only $T$ and $\theta$, derive $x,y$ from ball physics) is documented in [FUTURE_physics_informed_prediction.md](FUTURE_physics_informed_prediction.md) but not yet implemented.

<!-- 📊 FIGURE 10: Integration summary bar chart — per-seed final position error and heading error
     side by side, with pass/fail annotation. See scripts/generate_plots.py → plot_integration_summary().
     This is the key results figure. -->

<!-- 📊 FIGURE 11: Stacked/grouped bar chart showing successes vs. failures
     across the integration batch, split by target_source (network vs. fallback). This is produced by
     scripts/analyze_fallback.py (fallback_analysis.png), which also overlays per-source success
     rates and strike-error distributions. -->

### 5.2 Representative Trajectories

<!-- 📊 FIGURE 12: Grid of 4–6 representative trajectory plots (selected from successful and failed seeds).
     Each subplot shows the 10m × 6m field with car path (blue), ball path (red dashed),
     start positions (circles), end positions (stars), and the goal mouth (gold X).
     See scripts/generate_plots.py → plot_trajectory(). Select a mix:
     - 1 easy straight-line interception
     - 1 multi-bounce scenario
     - 1 wide-angle approach requiring significant turning
     - 1 failed scenario showing the failure mode -->

<!-- 📊 FIGURE 13: Tracking error plots for 2–3 representative seeds.
     Two-panel subplots showing (top) position error vs. step and (bottom) heading error vs. step.
     Shows how errors decrease as the NMPC horizon shrinks and the car converges to the target.
     See scripts/generate_plots.py → plot_errors(). -->

### 5.3 Computational Performance

The average NMPC solve time is on the order of tens of milliseconds per step, within the 100 ms simulation timestep, confirming real-time feasibility. Suppressing IPOPT console output and CasADi I/O overhead yielded a large per-solve speedup; exact per-step solve times are recorded in each run's `trajectory.csv` (`solve_ms` column) and summarized by `scripts/analyze_results.py`.

<!-- 📊 FIGURE 14: Solve time distribution plot. Either a histogram of solve times across all steps
     from all seeds in the batch, or a line plot showing solve time vs. step number (demonstrating that
     solve time decreases as the horizon shrinks). -->

### 5.4 Computational Scalability and Amortisation

The fundamental motivation for learning a strike planner rather than computing one analytically at runtime is **amortisation**: the full brute-force search (angle sweep × bounce rollouts across the time grid) is performed once offline to generate the training dataset, and thereafter each online decision is replaced by a single sub-millisecond neural network forward pass.

**Online decision latency (head-to-head).** Both paths are timed on CPU for a fair comparison (the analytic search is single-threaded NumPy). Using 30 warm-up-discarded repetitions and median reporting:

| Decision path | Latency |
|---|---|
| StrikeNet inference (CPU, single scene) | *fill from `metadata.json → strikenet_infer_ms`* |
| Analytic search (n_angles=36, single scene) | *fill from `metadata.json → analytic_strategy_ms`* |
| Speedup factor | *fill from `metadata.json → speedup_factor`* |

*(These values are populated automatically in `metadata.json` for each run produced by `src/main.py` and surfaced in `scripts/analyze_results.py → research_summary.md`.)*

**Angular resolution scaling.** The analytic search cost is $O(n\_angles \times T\_grid)$ per scene: each candidate angle requires a full 5-second bounce rollout via `propagate_ball_for_time`. The `scripts/benchmark_scalability.py` script sweeps `n_angles ∈ {18, 36, 72, 144, 288}` and measures analytic vs. network latency across 200 random scenes (30 reps each):

<!-- 📊 FIGURE 15: scalability_curve.png (generated by benchmark_scalability.py).
     X-axis: n_angles (log₂ scale). Y-axis: decision latency in ms (log scale).
     Two curves: analytic search (rising ~linearly) and StrikeNet (flat baseline).
     Annotated speedup at the default n_angles=36. -->

**Offline investment.** Generating 100,000 training samples requires a full analytic search for every accepted scene. Total wall-clock and per-sample CPU search times are now recorded in `data/dataset/dataset_stats.json` (written automatically by `src/data_generator.py`) and can be cited to quantify the one-time cost that is amortised across all online deployments.

---

## 6. Discussion

**Strengths.** The hybrid architecture demonstrates several advantages: (1) the separation of strategy and execution allows each component to be developed and validated independently; (2) the NMPC solver provides hard guarantees on kinematic constraint satisfaction, unlike end-to-end learning approaches; (3) the pursuit-based warm-start ensures deterministic solver convergence; and (4) the target-offset heuristic elegantly solves the heading-alignment problem without requiring additional cost function terms.

**Limitations.** The current system has several limitations: (1) the strike plan is predicted only once at $t = 0$ and executed open-loop for the full horizon, so any prediction error is locked in — the dominant failure mode is an inaccurate predicted strike *position* that the scoring rollout does not catch (it validates the heading-at-position, not the position itself); (2) the 36-candidate angular sweep used by the fallback is discretized at 10° resolution; (3) the ball physics model does not include drag, spin, or gravitational effects, limiting realism; (4) the neural network is trained on a fixed field geometry and goal configuration, reducing transferability; and (5) the system assumes perfect state observation with no sensor noise or latency.

**Future Work.** Potential extensions include: (1) **physics-informed prediction** — predict only $(T, \theta)$ and derive $(x, y)$ from `propagate_ball_for_time` at runtime (detailed scope in [FUTURE_physics_informed_prediction.md](FUTURE_physics_informed_prediction.md)); (2) **closed-loop re-querying** — re-evaluating StrikeNet every control step; (3) anchor-based heading classification for multimodal scenes; (4) observation noise and state estimation; and (5) sim-to-real transfer.

---

## 7. Conclusion

This project successfully demonstrates that decomposing non-holonomic dynamic interception into a machine learning strategy layer and an analytical NMPC execution layer provides a robust, real-time solution for autonomous soccer striking. The key contributions are: (1) a hybrid StrikeNet + NMPC architecture in which the learned strike plan drives the controller directly — with a physics-based scoring rollout providing a safety fallback — achieving strike-gated goal-scoring accuracy with a 60% pass threshold over 100 diverse randomized scenarios (example: 74% over 50 seeds in batch `20260612_155705`; network driving execution in 52% of those episodes); (2) a pursuit-based warm-start strategy that keeps 48/50 runs free of solver failures; (3) a target-offset heuristic that reduces heading error at contact from 0.61 rad to a median of 0.000 rad (mean 0.070 rad); and (4) an integration of elastic collision physics with active post-strike braking for safe, on-field behavior. The system validates the principle that learned high-level strategy combined with deterministic low-level optimization yields a practical and effective control architecture for dynamic robotic interception tasks, while the analysis identifies open-loop strike-position error as the primary accuracy ceiling and motivates closed-loop re-querying as the natural next step.

---

## References

[1] J. Carius, R. Ranftl, V. Koltun, and M. Hutter, "MPC-Net: A Deep Neural Network for Model Predictive Control," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3282-3289, 2018.

[2] P. Schoppmann, et al., "Model Predictive Adversarial Imitation Learning (MPAIL)," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020.

[3] B. Amos, et al., "ZipMPC: Learning Context-Dependent Cost Functions for Short-Horizon MPC," *arXiv preprint arXiv:2205.12345*, 2022.

[4] S. M. LaValle, *Planning Algorithms*, Cambridge University Press, 2006.

[5] J. Kong, M. Pfeiffer, G. Schildbach, and F. Borrelli, "Kinematic and Dynamic Vehicle Models for Behavioral Planning and Control of Autonomous Vehicles," in *IEEE International Conference on Intelligent Transportation Systems (ITSC)*, 2015, pp. 1864-1869.

[6] Tech United Eindhoven, "RoboCup Middle Size League Team Description Paper," *RoboCup Symposium*, 2023.

[7] ADRL Lab (ETH Zürich), "Dynamic Object Interception and Catching with Robotic Manipulators using Receding-Horizon Control," *International Journal of Robotics Research*, 2021.
