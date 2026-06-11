"""
nmpc_solver.py — Shrinking-Horizon NMPC using CasADi.

Implements the InterceptionMPC class that, given the current car state,
a target strike state, and remaining horizon N, solves for the optimal
control sequence via multiple shooting and returns u_0.
"""

import casadi as ca
import numpy as np


class InterceptionMPC:
    """
    Non-linear Model Predictive Controller for kinodynamic interception.

    Uses CasADi's Opti() interface with IPOPT and the RK4-discretised
    kinematic bicycle model (multiple-shooting formulation).
    """

    def __init__(
        self,
        dt: float = 0.1,
        L: float = 0.3,
        v_min: float = 0.0,
        v_max: float = 2.0,
        delta_max: float = np.pi / 4,
        a_max: float = 2.0,
        Q_terminal: np.ndarray = None,
        R: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        dt         : time step (must match World.dt)
        L          : wheelbase
        v_min/max  : speed bounds
        delta_max  : max steering angle (symmetric)
        a_max      : max acceleration (symmetric)
        Q_terminal : 4×4 terminal cost weight  (default: diag([3000, 3000, 300, 1]))
        R          : 2×2 running cost weight    (default: diag([0.005, 0.005]))
        """
        self.dt = dt
        self.L = L
        self.v_min = v_min
        self.v_max = v_max
        self.delta_max = delta_max
        self.a_max = a_max

        if Q_terminal is None:
            Q_terminal = np.diag([3000.0, 3000.0, 300.0, 1.0])
        if R is None:
            R = np.diag([0.005, 0.005])

        self.Q_terminal = Q_terminal
        self.R = R

        # Solver options (built once, reused every solve call)
        self._p_opts = {"expand": True, "print_time": False, "verbose": False}
        self._s_opts = {
            "print_level": 0,
            "max_iter": 300,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "sb": "yes",  # suppress IPOPT banner
        }

        # Build the symbolic RK4 integrator once
        self._build_integrator()

    # ------------------------------------------------------------------
    # Symbolic RK4 integrator (built once, reused every solve)
    # ------------------------------------------------------------------
    def _build_integrator(self):
        """Create a CasADi Function for one RK4 step."""
        x = ca.MX.sym("x", 4)   # [x, y, theta, v]
        u = ca.MX.sym("u", 2)   # [a, delta]

        def dynamics(state, ctrl):
            """Continuous-time bicycle model in CasADi symbolics."""
            theta = state[2]
            v = state[3]
            a = ctrl[0]
            delta = ctrl[1]
            return ca.vertcat(
                v * ca.cos(theta),
                v * ca.sin(theta),
                (v / self.L) * ca.tan(delta),
                a,
            )

        dt = self.dt
        k1 = dynamics(x, u)
        k2 = dynamics(x + 0.5 * dt * k1, u)
        k3 = dynamics(x + 0.5 * dt * k2, u)
        k4 = dynamics(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.F = ca.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(
        self,
        current_state: np.ndarray,
        target_strike_state: np.ndarray,
        N_steps: int,
    ) -> np.ndarray:
        """
        Solve the NMPC problem and return the first control action u_0.

        Parameters
        ----------
        current_state       : array [x, y, theta, v]
        target_strike_state : array [x, y, theta, v]  — desired terminal state
        N_steps             : prediction horizon (number of control intervals)

        Returns
        -------
        u0 : array [a, delta]   — first optimal control action
        """
        if N_steps <= 0:
            # No time left — return zero control
            return np.array([0.0, 0.0])

        opti = ca.Opti()

        # ---- Decision variables ----
        X = opti.variable(4, N_steps + 1)   # states
        U = opti.variable(2, N_steps)       # controls

        # ---- Parameters (set at solve time) ----
        p_x0 = opti.parameter(4)            # initial state
        p_xT = opti.parameter(4)            # target terminal state

        # ---- Initial condition ----
        opti.subject_to(X[:, 0] == p_x0)

        # ---- Dynamics constraints (multiple shooting) ----
        for k in range(N_steps):
            x_next = self.F(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)

        # ---- Bound constraints ----
        for k in range(N_steps + 1):
            # Speed bounds (all N+1 states)
            opti.subject_to(opti.bounded(self.v_min, X[3, k], self.v_max))
            if k < N_steps:
                # Acceleration and steering bounds (N controls)
                opti.subject_to(opti.bounded(-self.a_max, U[0, k], self.a_max))
                opti.subject_to(opti.bounded(-self.delta_max, U[1, k], self.delta_max))

        # ---- Objective ----
        # Terminal cost with angle-wrap-safe heading penalty
        dx = X[0, N_steps] - p_xT[0]
        dy = X[1, N_steps] - p_xT[1]
        dv = X[3, N_steps] - p_xT[3]
        # Use 1 - cos(dtheta) instead of dtheta^2 to avoid wrapping issues
        dtheta_cos = 1.0 - ca.cos(X[2, N_steps] - p_xT[2])

        terminal_cost = (
            self.Q_terminal[0, 0] * dx ** 2
            + self.Q_terminal[1, 1] * dy ** 2
            + self.Q_terminal[2, 2] * dtheta_cos
            + self.Q_terminal[3, 3] * dv ** 2
        )

        # Running cost (control effort)
        control_cost = 0.0
        for k in range(N_steps):
            control_cost += (
                self.R[0, 0] * U[0, k] ** 2
                + self.R[1, 1] * U[1, k] ** 2
            )

        opti.minimize(terminal_cost + control_cost)

        opti.solver("ipopt", self._p_opts, self._s_opts)

        # ---- Set parameter values ----
        opti.set_value(p_x0, current_state)
        opti.set_value(p_xT, target_strike_state)

        # ---- Warm-start: kinematically feasible pursuit-based initial guess ----
        X_guess = np.zeros((4, N_steps + 1))
        U_guess = np.zeros((2, N_steps))
        X_guess[:, 0] = current_state

        x_curr = current_state.copy()
        for k in range(N_steps):
            # Compute line-of-sight angle to target position
            dx = target_strike_state[0] - x_curr[0]
            dy = target_strike_state[1] - x_curr[1]
            theta_los = np.arctan2(dy, dx)
            
            # Simple proportional pursuit steering angle
            dtheta = np.arctan2(np.sin(theta_los - x_curr[2]), np.cos(theta_los - x_curr[2]))
            delta = np.clip(1.5 * dtheta, -self.delta_max, self.delta_max)
            
            # Linear acceleration to reach target speed
            steps_left = N_steps - k
            a = np.clip((target_strike_state[3] - x_curr[3]) / (steps_left * self.dt), -self.a_max, self.a_max)
            
            u_cand = np.array([a, delta])
            U_guess[:, k] = u_cand
            
            # Propagate forward using symbolic RK4 integrator evaluated numerically
            x_next = np.array(self.F(x_curr, u_cand)).flatten()
            X_guess[:, k + 1] = x_next
            x_curr = x_next

        for k in range(N_steps + 1):
            opti.set_initial(X[:, k], X_guess[:, k])
        for k in range(N_steps):
            opti.set_initial(U[:, k], U_guess[:, k])

        # ---- Solve ----
        try:
            sol = opti.solve()
            u0 = sol.value(U[:, 0])
        except RuntimeError:
            # If IPOPT fails, fall back to the debug values
            print("[NMPC] WARNING: Solver failed, returning zero control.")
            u0 = np.array([0.0, 0.0])

        return np.array(u0).flatten()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mpc = InterceptionMPC()
    x0 = np.array([1.0, 3.0, 0.0, 0.0])
    x_target = np.array([5.0, 3.0, 0.0, 1.0])
    u0 = mpc.solve(x0, x_target, N_steps=30)
    print(f"Initial state : {x0}")
    print(f"Target state  : {x_target}")
    print(f"First control : a={u0[0]:.4f}, delta={u0[1]:.4f}")
