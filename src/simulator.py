"""
simulator.py — The "World" for the Robot Soccer Striker project.

Handles:
  - Car state update via RK4 integration of the kinematic bicycle model.
  - Ball state update via constant-velocity propagation.
  - Matplotlib-based 2-D rendering of the field, car, ball, and goal.

No planning or optimisation logic lives here.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


# ---------------------------------------------------------------------------
# Default physical constants
# ---------------------------------------------------------------------------
DEFAULT_DT = 0.1          # s
DEFAULT_WHEELBASE = 0.3   # m
DEFAULT_FIELD_W = 10.0    # m
DEFAULT_FIELD_H = 6.0     # m
CAR_LENGTH = 0.4          # m  (for drawing only)
CAR_WIDTH = 0.2           # m  (for drawing only)
BALL_RADIUS = 0.1         # m  (for drawing only)


class World:
    """Minimal physics world: one car, one ball, one goal."""

    def __init__(
        self,
        car_state: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        goal_pos: np.ndarray,
        dt: float = DEFAULT_DT,
        L: float = DEFAULT_WHEELBASE,
        field_size: tuple = (DEFAULT_FIELD_W, DEFAULT_FIELD_H),
    ):
        """
        Parameters
        ----------
        car_state : array [x, y, theta, v]
        ball_pos  : array [x, y]
        ball_vel  : array [vx, vy]
        goal_pos  : array [x, y]
        dt        : simulation time-step (s)
        L         : wheelbase (m)
        field_size: (width, height) of the pitch (m)
        """
        self.car_state = np.array(car_state, dtype=float)
        self.ball_pos = np.array(ball_pos, dtype=float)
        self.ball_vel = np.array(ball_vel, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.dt = dt
        self.L = L
        self.field_size = field_size

        # For rendering
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        car_state: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        goal_pos: np.ndarray,
    ) -> None:
        """Re-initialise the world for a new episode."""
        self.car_state = np.array(car_state, dtype=float)
        self.ball_pos = np.array(ball_pos, dtype=float)
        self.ball_vel = np.array(ball_vel, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _car_dynamics(state: np.ndarray, u: np.ndarray, L: float) -> np.ndarray:
        """
        Continuous-time derivatives of the kinematic bicycle model.

            q  = [x, y, theta, v]
            u  = [a, delta]

        Returns dq/dt.
        """
        _, _, theta, v = state
        a, delta = u
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            (v / L) * np.tan(delta),
            a,
        ])

    def _rk4_step(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Advance *state* by one dt using RK4 with controls *u* held constant."""
        dt = self.dt
        L = self.L
        k1 = self._car_dynamics(state, u, L)
        k2 = self._car_dynamics(state + 0.5 * dt * k1, u, L)
        k3 = self._car_dynamics(state + 0.5 * dt * k2, u, L)
        k4 = self._car_dynamics(state + dt * k3, u, L)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, u: np.ndarray) -> None:
        """
        Advance the world by one time-step.

        Parameters
        ----------
        u : array [a, delta]   (acceleration, steering angle)
        """
        u = np.asarray(u, dtype=float)

        # 1. Update car (RK4)
        self.car_state = self._rk4_step(self.car_state, u)
        # Normalise heading to [-pi, pi]
        self.car_state[2] = np.arctan2(
            np.sin(self.car_state[2]), np.cos(self.car_state[2])
        )

        # 2. Update ball (constant velocity)
        self.ball_pos = self.ball_pos + self.ball_vel * self.dt

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, ax=None, title: str = "") -> None:
        """
        Draw the current state of the world.

        If *ax* is None the method creates / reuses its own figure.
        """
        if ax is None:
            if self._fig is None or not plt.fignum_exists(self._fig.number):
                self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 6))
            ax = self._ax

        ax.cla()
        W, H = self.field_size

        # --- Field (green background) ---
        field = patches.Rectangle(
            (0, 0), W, H,
            linewidth=2, edgecolor="white", facecolor="#2e7d32",
        )
        ax.add_patch(field)

        # --- Goal marker ---
        gx, gy = self.goal_pos
        goal_marker = patches.Circle(
            (gx, gy), 0.25,
            linewidth=2, edgecolor="gold", facecolor="none",
        )
        ax.add_patch(goal_marker)
        ax.plot(gx, gy, marker="x", color="gold", markersize=10, markeredgewidth=2)

        # --- Ball (red) ---
        bx, by = self.ball_pos
        ball = patches.Circle(
            (bx, by), BALL_RADIUS,
            linewidth=1, edgecolor="black", facecolor="red",
        )
        ax.add_patch(ball)

        # --- Car (blue rectangle + heading line) ---
        cx, cy, ctheta, _ = self.car_state
        car_rect = patches.FancyBboxPatch(
            (-CAR_LENGTH / 2, -CAR_WIDTH / 2),
            CAR_LENGTH, CAR_WIDTH,
            boxstyle="round,pad=0.02",
            linewidth=1, edgecolor="black", facecolor="#1565c0",
        )
        t = (
            Affine2D().rotate(ctheta).translate(cx, cy) + ax.transData
        )
        car_rect.set_transform(t)
        ax.add_patch(car_rect)

        # Heading arrow
        arrow_len = CAR_LENGTH * 0.7
        ax.arrow(
            cx, cy,
            arrow_len * np.cos(ctheta),
            arrow_len * np.sin(ctheta),
            head_width=0.08, head_length=0.05,
            fc="yellow", ec="yellow",
        )

        # --- Axes ---
        margin = 0.5
        ax.set_xlim(-margin, W + margin)
        ax.set_ylim(-margin, H + margin)
        ax.set_aspect("equal")
        ax.set_facecolor("#1b5e20")
        ax.set_title(title, fontsize=12, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        plt.tight_layout()
        if matplotlib.get_backend().lower() != "agg":
            plt.pause(0.001)


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    world = World(
        car_state=[1.0, 3.0, 0.0, 0.0],
        ball_pos=[5.0, 3.0],
        ball_vel=[0.0, 0.0],
        goal_pos=[9.5, 3.0],
    )
    plt.ion()
    for i in range(30):
        world.step([0.5, 0.1])          # gentle acceleration + slight steer
        world.render(title=f"Step {i}")
    plt.ioff()
    plt.show()
