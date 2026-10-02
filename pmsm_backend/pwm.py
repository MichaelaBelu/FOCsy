from __future__ import annotations

from typing import Tuple

import numpy as np

def abc_zero_sequence_offset(v_a_ref: float, v_b_ref: float, v_c_ref: float) -> float:
    """
    Offset-voltage SVPWM:
        v_offset = -(Vmax + Vmin)/2
    where Vmax and Vmin are the max/min among the three PHASE references.

    Pole-voltage refs:
        v_an* = v_a_ref + v_offset
        v_bn* = v_b_ref + v_offset
        v_cn* = v_c_ref + v_offset
    """
    v_max = max(v_a_ref, v_b_ref, v_c_ref)
    v_min = min(v_a_ref, v_b_ref, v_c_ref)
    v_offset = -0.5 * (v_max + v_min)
    return v_offset


def phase_refs_to_svpwm_pole_refs(v_abc_ref: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Input:
        phase voltage references [v_a*, v_b*, v_c*] with zero average
    Output:
        pole voltage references [v_an*, v_bn*, v_cn*]
        using offset-voltage SVPWM
    """
    v_a_ref, v_b_ref, v_c_ref = v_abc_ref
    v_offset = abc_zero_sequence_offset(v_a_ref, v_b_ref, v_c_ref)
    v_pole_ref = np.array([
        v_a_ref + v_offset,
        v_b_ref + v_offset,
        v_c_ref + v_offset
    ])
    return v_pole_ref, v_offset


def pole_refs_to_duties(v_pole_ref: np.ndarray, Vdc: float) -> np.ndarray:
    """
    For a 2-level inverter:
        average pole voltage = (d - 0.5)*Vdc
    therefore:
        d = 0.5 + v_pole_ref / Vdc

    eq. 7.59 and 7.60

    This matches center-aligned PWM average timing:
        T_on = Ts/2 + (v_pole_ref/Vdc)*Ts
    """
    d = 0.5 + v_pole_ref / Vdc
    return np.clip(d, 0.0, 1.0)


# TODO: learn this if it is actually true !!!!!!!!!!!!!!
def apply_deadtime_to_duties(d_abc_cmd: np.ndarray, i_abc: np.ndarray, deadtime: float, T_pwm: float) -> np.ndarray:
    """
    Average dead-time model.

    A common average approximation:
        Delta d = -(deadtime / T_pwm) * sign(i_phase)

    Positive current:
        upper switch effective duty is reduced
    Negative current:
        upper switch effective duty is increased

    This captures the average voltage error caused by dead-time.
    """
    if deadtime <= 0.0:
        return np.clip(d_abc_cmd, 0.0, 1.0)

    dt_ratio = deadtime / T_pwm

    sgn = np.zeros(3)
    for k in range(3):
        I_eps = 0.1
        sgn[k] = np.tanh(i_abc[k] / I_eps)
        # if i_abc[k] > 1e-9:
        #     sgn[k] = 1.0
        # elif i_abc[k] < -1e-9:
        #     sgn[k] = -1.0
        # else:
        #     sgn[k] = 0.0

    d_eff = d_abc_cmd - dt_ratio * sgn
    return np.clip(d_eff, 0.0, 1.0)


def duties_to_average_pole_voltages(d_abc: np.ndarray, Vdc: float) -> np.ndarray:
    """
    v_pole_avg = d * (Vdc / 2) + (1 - d) * (-Vdc / 2)

    average pole voltage for each leg:
        v_pole_avg = (d - 0.5)*Vdc
    """
    return (d_abc - 0.5) * Vdc


def pole_to_phase_voltages(v_pole_abc: np.ndarray) -> np.ndarray:
    """
    Phase voltages are pole voltages with the common-mode removed.

    Convert pole voltages to phase voltages by removing common-mode:
        v_phase = v_pole - mean(v_pole)

    Now the three phase voltages sum to zero, which is what the motor model expects.
    Because the motor does not care about a voltage that is added equally to all three phases.
    The motor phase voltages are the voltages across the windings relative to the floating motor neutral.
    """
    v_cm = np.mean(v_pole_abc)
    return v_pole_abc - v_cm


def triangular_carrier_from_time(t_local: np.ndarray, T_pwm: float) -> np.ndarray:
    """
    Symmetric triangular carrier in [-1, 1].

    keeps only the fractional part so x is always [0,1)

    abs(x - 0.5)
    - at x = 0.5 → value is 0
    - at x = 0 or x = 1 → value is 0.5
    - range is [0,0.5]

    returns:
        -1 at the center (x=0.5)
        +1 at the edges (x=0 and x=1)
    """
    x = (t_local / T_pwm) % 1.0
    return 4.0 * np.abs(x - 0.5) - 1.0


# def make_center_aligned_gate_from_duty(t_local: float, duty: float, T_pwm: float) -> float:
#     """
#     Center-aligned upper-switch command over one PWM period.
#     duty in [0,1]
#     """
#     t_in = t_local % T_pwm
#     t_on = 0.5 * (1.0 - duty) * T_pwm
#     t_off = T_pwm - t_on
#     return 1.0 if (t_in >= t_on and t_in <= t_off) else 0.0


def make_center_aligned_gates_with_deadtime(t_zoom: np.ndarray, duty_cmd: np.ndarray, T_pwm: float, deadtime: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate upper/lower gate signals from duty command with dead-time.
    This is for plotting, not for the ODE.
    - upper gate: delay turn-on by deadtime, turn-off immediately
    - lower gate: turn on only after upper has turned off + deadtime
    """
    g_up = np.zeros_like(t_zoom)
    g_lo = np.zeros_like(t_zoom)

    for k, tk in enumerate(t_zoom):
        t_in = tk % T_pwm
        d = np.clip(duty_cmd[k], 0.0, 1.0)

        # commanded symmetric pulse
        t_on_cmd = 0.5 * (1.0 - d) * T_pwm
        t_off_cmd = T_pwm - t_on_cmd

        # upper gate: delay turn-on by deadtime, turn-off immediately
        up_on = t_on_cmd + deadtime
        up_off = t_off_cmd

        # lower gate: turn on only after upper has turned off + deadtime
        lo_on_1 = 0.0
        lo_off_1 = max(t_on_cmd - deadtime, 0.0)

        lo_on_2 = min(t_off_cmd + deadtime, T_pwm)
        lo_off_2 = T_pwm

        upper = 1.0 if (t_in >= up_on and t_in <= up_off and up_on < up_off) else 0.0
        lower = 1.0 if ((t_in >= lo_on_1 and t_in <= lo_off_1 and lo_on_1 < lo_off_1) or
                        (t_in >= lo_on_2 and t_in <= lo_off_2 and lo_on_2 < lo_off_2)) else 0.0

        g_up[k] = upper
        g_lo[k] = lower

    return g_up, g_lo