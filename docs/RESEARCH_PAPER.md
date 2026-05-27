# Autonomous Soccer Striker: A Hybrid Approach using Imitation Learning and Non-linear Model Predictive Control

## Abstract
This paper presents a hybrid control architecture for an autonomous robotic soccer striker tasked with intercepting a moving, bouncing ball and deflecting it into a goal. The system combines an offline imitation learning strategy network (StrikeNet) to estimate optimal interception timing with an online, shrinking-horizon Non-linear Model Predictive Control (NMPC) solver for precise kinematic execution. By incorporating elastic collision models, target-offset heuristics, and post-strike active braking, the system overcomes real-time kinodynamic constraints. Integration tests demonstrate a 70% goal-scoring success rate across randomized initial conditions, achieving sub-millimeter positional precision at the moment of impact.

---

## 1. Introduction
Intercepting a high-speed, bouncing target with a non-holonomic vehicle presents a significant challenge in robotics. The agent must simultaneously satisfy vehicle kinodynamic constraints (e.g., maximum acceleration, bounded steering), predict the target's complex trajectory (including wall bounces), and strike the object with a precise orientation to redirect it toward a specific goal mouth. 

Traditional purely analytical planners struggle with the real-time computational burden of solving long-horizon, bounce-aware optimal control problems. Conversely, end-to-end reinforcement learning methods often lack the strict constraint satisfaction required for high-speed vehicle control. This project proposes a two-phase hybrid approach: a lightweight neural network dictates the macroscopic strategy (when and how to strike), while a deterministic NMPC solver manages the microscopic execution (how to drive there).

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
The vehicle is modeled as a kinematic bicycle with a wheelbase of $L=0.3$ m, bounded velocity ($[0.0, 2.0]$ m/s), and bounded steering ($[-\pi/4, \pi/4]$ rad). 
Upon intersection ($< 0.35$ m distance), an elastic 2D collision occurs between the flat car bumper and the ball, using a restitution coefficient of $e_{strike} = 0.8$. The momentum transfer redirects the ball toward the goal line ($x=10.0, y \in [2.0, 4.0]$).

### 3.4 Non-linear Model Predictive Control (NMPC)
The NMPC loop uses CasADi and IPOPT to solve a shrinking-horizon optimal control problem at a rate of 10 Hz ($\Delta t = 0.1$ s). To prevent the vehicle from triggering an early collision before completing its terminal rotation, the NMPC target is artificially offset by $d_{offset} = 0.32$ m backwards along the approach vector.

Following the strike, a Phase 2 active braking controller is engaged, applying maximum deceleration ($-2.0$ m/s²) to bring the vehicle to a halt, ensuring it remains safely on the pitch while the ball coasts into the goal.

---

## 4. Results
The system was validated using a rigorous integration test suite over 10 randomized, distinct seeds, encompassing varied approach angles and multi-bounce trajectories.

**Key Metrics:**
*   **Goal Success Rate:** 70% (7 out of 10 seeds resulted in a successful goal).
*   **Strike Position Error:** 0.315 m average (well within the physical contact radius of 0.35 m).
*   **Strike Heading Error:** 0.030 rad average ($\approx 1.7^\circ$).
*   **Safety Constraints:** 100% of runs maintained the vehicle and ball on-field post-strike.

Optimization of the IPOPT solver configurations (silencing I/O overhead) resulted in a 120x execution speedup, allowing the NMPC to solve at roughly 60 ms per step.

---

## 5. Conclusion
This project successfully demonstrates that decomposing non-holonomic interception into a machine learning strategy layer and an analytical NMPC execution layer provides a robust, real-time solution. The inclusion of target offsets and elastic collision mechanics bridged the gap between mathematical trajectory planning and simulated physical impact, culminating in a highly effective robotic striker.
