"""
goal.py — Defines the Goal geometry and scoring checks.
"""

import numpy as np

class Goal:
    def __init__(self, x: float = 10.0, y_min: float = 2.0, y_max: float = 4.0):
        self.x = float(x)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, (self.y_min + self.y_max) / 2.0])

    def check_score(self, pos_old: np.ndarray, pos_new: np.ndarray) -> bool:
        """
        Checks if the ball's trajectory crossed the goal line within the goal mouth.
        Since the goal is on the right wall (x = W), we check if the path crosses x = self.x
        from left to right, and the intersection point's y-coordinate is within [y_min, y_max].
        """
        x_old, y_old = pos_old[0], pos_old[1]
        x_new, y_new = pos_new[0], pos_new[1]

        # Check if we crossed the x line (either direction, but usually left-to-right)
        # We check if self.x lies between x_old and x_new
        if (x_old <= self.x <= x_new) or (x_new <= self.x <= x_old):
            if abs(x_new - x_old) < 1e-12:
                # Vertical movement exactly on the line
                if self.y_min <= y_new <= self.y_max:
                    return True
                return False
            # Linear interpolation to find y at x = self.x
            t = (self.x - x_old) / (x_new - x_old)
            y_cross = y_old + t * (y_new - y_old)
            if self.y_min <= y_cross <= self.y_max:
                return True
        return False
