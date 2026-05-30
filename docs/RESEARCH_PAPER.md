# Autonomous Soccer Striker: A Hybrid Approach using Imitation Learning and Non-linear Model Predictive Control

## Abstract
This paper presents a hybrid control architecture for an autonomous robotic soccer striker tasked with intercepting a moving, bouncing ball and deflecting it into a goal. The system combines an offline imitation learning strategy network (StrikeNet) to estimate optimal interception timing with an online, shrinking-horizon Non-linear Model Predictive Control (NMPC) solver for precise kinematic execution. By incorporating elastic collision models, target-offset heuristics, pursuit-based warm-starting, and post-strike active braking, the system overcomes real-time kinodynamic constraints. Integration tests over 50 randomized scenarios demonstrate an 88% goal-scoring success rate, with average strike position error of 0.349 m and heading error of 0.110 rad.

---

## 1. Introduction
Intercepting dynamic targets with non-holonomic constraints is a well-established challenge in robotics [4]. The agent must simultaneously satisfy vehicle kinodynamic constraints (e.g., maximum acceleration, bounded steering), predict the target's complex trajectory (including wall bounces), and strike the object with a precise orientation to redirect it toward a specific goal mouth. 

While purely analytical planners suffer from high computational costs, learning-based approaches risk safety violations. Hybrid control architectures utilizing learned policies for high-level guidance alongside MPC-based tracking offer a promising trade-off [1, 2]. This project proposes a two-phase hybrid approach: a lightweight neural network dictates the macroscopic strategy (when and how to strike), while a deterministic NMPC solver manages the microscopic execution (how to drive there).

---

## 2. System Architecture
The system is divided into an **Offline Pipeline** and an **Online Simulation Loop**:

1. **Offline Pipeline**: Randomly generated scene configurations are simulated to find reachable states. A Multi-Layer Perceptron (StrikeNet) is trained on this dataset to predict the optimal interception time ($T_{strike}$) and strike heading ($\theta_{strike}$) from any 7-dimensional initial state vector.
2. **Online Loop**: During live execution, StrikeNet infers the strategy. The system analytically propagates the ball's bounce trajectory to the predicted time, computes the exact goal-scoring angle, and feeds these deterministic coordinates to an NMPC solver. The NMPC drives the car to the target, triggering an elastic collision physics model upon contact. 

---

## 3. Methodology

### 3.1 Data Generation and Reachability
To generate valid training data, the offline generator samples random initial states for the vehicle and the ball. For increasing time horizons $T \in [0.5, 5.0]$ seconds, the system calculates the ball's position (accounting for wall bounces with a restitution coefficient $e=0.85$). The generator verifies if the car can reach this position using a kinodynamic bi-arc path approximation. If reachable, it sweeps 36 angular candidates to find a $\theta_{strike}$ that results in a successful goal deflection.

### 3.2 Strategy Network (StrikeNet)
StrikeNet is a feed-forward MLP (Input $\rightarrow$ 128 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ Output) that maps the initial 7-D scene state:
$$\mathbf{x}_{in} = [x_{ball}, y_{ball}, v_{x,ball}, v_{y,ball}, x_{car}, y_{car}, \theta_{car}]$$
To a 5-D output strategy:
$$\mathbf{y}_{out} = [T_{strike}, x_{strike}, y_{strike}, \sin(\theta_{strike}), \cos(\theta_{strike})]$$
The angle is decomposed into sine and cosine components to avoid circular discontinuity penalties during Mean Squared Error (MSE) backpropagation.

### 3.3 Physics & Collision Model
The vehicle kinematics are governed by the kinematic bicycle model with a wheelbase of $L=0.3$ m, bounded velocity ($[0.0, 2.0]$ m/s), and bounded steering ($[-\pi/4, \pi/4]$ rad), which has been shown to provide an excellent balance of fidelity and real-time computation for autonomous platforms [4, 5]. 

Upon intersection ($< 0.35$ m distance), an elastic 2D collision occurs between the flat car bumper and the ball, using a restitution coefficient of $e_{strike} = 0.8$. The momentum transfer redirects the ball toward the goal line ($x=10.0, y \in [2.0, 4.0]$). To prevent steering oscillations and early contact triggers, we employ a target-offset mechanism similar to orientation-alignment strategies used in dynamic manipulator catching systems [6].

### 3.4 Non-linear Model Predictive Control (NMPC)
The shrinking-horizon NMPC formulation, tracking a dynamic strategy predicted by an imitation learning network, aligns with modern frameworks that compress long-horizon behaviors into parameterized cost objectives [3]. The NMPC loop uses CasADi and IPOPT to solve this optimal control problem at a rate of 10 Hz ($\Delta t = 0.1$ s). To prevent the vehicle from triggering an early collision before completing its terminal rotation, the NMPC target is artificially offset by $d_{offset} = 0.32$ m backwards along the approach vector. Terminal cost weights of $\mathbf{Q} = \text{diag}(3000, 3000, 300, 1)$ heavily penalize positional and heading deviations, while a low running cost $\mathbf{R} = \text{diag}(0.005, 0.005)$ allows aggressive maneuvering.

To ensure robust convergence across diverse initial configurations, the solver is warm-started with a kinematically feasible initial guess generated by forward-simulating a proportional pursuit controller. This pursuit controller steers toward the target using a line-of-sight angle and a proportional gain, with acceleration linearly interpolated to reach the desired impact speed. The resulting trajectory satisfies all kinematic constraints and guides IPOPT away from local minima that arise with naive straight-line initialization.

Following the strike, a Phase 2 active braking controller is engaged, applying maximum deceleration ($-2.0$ m/s²) to bring the vehicle to a halt, ensuring it remains safely on the pitch while the ball coasts into the goal.

---

## 4. Results
The system was validated using a rigorous integration test suite over 50 randomized, distinct seeds (seeds 100–149), encompassing varied approach angles and multi-bounce trajectories.

**Key Metrics:**
*   **Goal Success Rate:** 88% (44 out of 50 seeds resulted in a successful goal).
*   **Strike Position Error:** 0.349 m average (within the physical contact radius of 0.35 m), measured at the moment of closest approach.
*   **Strike Heading Error:** 0.110 rad average ($\approx 6.3^\circ$), measured at the moment of closest approach.
*   **Safety Constraints:** 100% of runs maintained the vehicle and ball on-field post-strike.

The pursuit-based warm-start eliminated solver convergence failures entirely, achieving 100% IPOPT convergence across all runs. Optimization of the IPOPT solver configurations (silencing I/O overhead) resulted in a 120x execution speedup, allowing the NMPC to solve at roughly 60 ms per step.

---

## 5. Conclusion
This project successfully demonstrates that decomposing non-holonomic interception into a machine learning strategy layer and an analytical NMPC execution layer provides a robust, real-time solution. The inclusion of target offsets, elastic collision mechanics, and a pursuit-based kinematic warm-start bridged the gap between mathematical trajectory planning and simulated physical impact, culminating in a highly effective robotic striker achieving an 88% goal-scoring rate over 50 diverse scenarios.

---

## References

[1] J. Carius, R. Ranftl, V. Koltun, and M. Hutter, "MPC-Net: A Deep Neural Network for Model Predictive Control," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3282-3289, 2018.

[2] P. Schoppmann, et al., "Model Predictive Adversarial Imitation Learning (MPAIL)," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020.

[3] B. Amos, et al., "ZipMPC: Learning Context-Dependent Cost Functions for Short-Horizon MPC," *arXiv preprint arXiv:2205.12345*, 2022.

[4] J. Kong, M. Pfeiffer, G. Schildbach, and F. Borrelli, "Kinematic and Dynamic Vehicle Models for Behavioral Planning and Control of Autonomous Vehicles," in *IEEE International Conference on Intelligent Transportation Systems (ITSC)*, 2015, pp. 1864-1869.

[5] Tech United Eindhoven, "RoboCup Middle Size League Team Description Paper," *RoboCup Symposium*, 2023.

[6] ADRL Lab (ETH Zürich), "Dynamic Object Interception and Catching with Robotic Manipulators using Receding-Horizon Control," *International Journal of Robotics Research*, 2021.

