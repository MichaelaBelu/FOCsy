from __future__ import annotations

from typing import Tuple

import numpy as np

def T_matrix(theta: float) -> np.ndarray:
    """Park-like transform matrix mapping abc -> dq0 in a frame rotated by theta"""
    angle = 2.0 * np.pi / 3.0
    coeff = 2.0 / 3.0
    row1 = [np.cos(theta), np.cos(theta - angle), np.cos(theta + angle)]
    row2 = [-np.sin(theta), -np.sin(theta - angle), -np.sin(theta + angle)]
    row3 = [0.5, 0.5, 0.5]
    return coeff * np.array([row1, row2, row3])


def T_inv(T: np.ndarray) -> np.ndarray:
    """Inverse mapping dq0 -> abc for this T: here T_inv = T^T (scaled inside T)."""
    # For our chosen T_matrix form, using T.T gives abc <- dq0 mapping
    return T.T


def dq0_to_abc(x_dq0: np.ndarray, theta: float) -> np.ndarray:
    return T_inv(T_matrix(theta)) @ x_dq0


def dq_voltage_limit(v_d: float, v_q: float, Vmax: float) -> Tuple[float, float]:
    '''
    This is called vector magnitude saturation.
    This is an inverter saturation model in dq = ACTUATOR LIMIT
    If the controller asks for too much dq voltage, the real inverter clips it
    Without modeling this, your sim will be unrealistically perfect and
      anti-windup won’t make sense.
    '''
    # it clips a 2-D vector to a circle of radius V_max without
    # changing its direction.
    # mag = magnitude (length) of the voltage vector
    mag = np.sqrt(v_d * v_d + v_q * v_q)
    if mag <= Vmax or mag < 1e-12:
        return v_d, v_q
    # This scales the voltage vector so that:
    # - Its direction stays the same
    # - Its magnitude becomes exactly V_max
    # normalised vector = 1/mag * [v_d; v_q] (same direction)
    # limited vector = Vmax * normalised = Vmax/mag * [v_d; v_q]
    return (Vmax / mag) * v_d, (Vmax / mag) * v_q

