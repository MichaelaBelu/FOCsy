from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def rad_s_to_rpm(omega: float) -> float:
    return omega * 60.0 / (2.0 * np.pi)


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * (2.0 * np.pi) / 60.0


# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "spmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",
#     "tfinal": 1.0,
#     "Imax": 3.0,
#     "P_rated": 0.0,
#     "poles": 10,
#     "Vdc": 60,  # 650.0,
#     "phi_f_ref": 0.00798,
#     "R_s_ref": 2.015,
#     "J_motor": 0.0000044347,
#     # "mass_default": 1600.0,
#     "wheel_radius": 0.04, # 0.3
#     "gear_ratio": 15.0,  # 9
#     "f_pwm": 20000.0,
#     # advanced
#     "f": 50.0,
#     "V_LL": 24.0,  # 400.0,
#     "L_ls": 0.0023,
#     "L_A": 0.0,
#     "L_B": 0.0,
#     "I_rated": 1.5,
#     "Ld_inf_factor": 0.4,
#     "Lq_inf_factor": 0.6,
#     "I_Ld_sat_factor": 1.0,
#     "I_Lq_sat_factor": 1.5,
#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,
#     "alpha_phi": 0.0010,
#     "C_th_s": 200.0,
#     "C_th_m": 400.0,
#     "R_th_sa": 0.4,
#     "R_th_sm": 0.8,
#     "R_th_ma": 1.2,
#     "T_amb": 25.0,
#     "T_warn_stator": 120.0,
#     "T_fault_stator": 160.0,
#     "T_warn_magnet": 100.0,
#     "T_fault_magnet": 140.0,
#     "T_demag": 120.0,
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 0.0,
#     "core_volume_m3": 0.0,
#     "B_iron_rated": 0.0,
#     "rho_air": 1.225,
#     "drivetrain_eff": 0.9,
#     "Cd": 0.6,  # 0.29
#     "A_front": 0.08,  # 2.2
#     "b_vehicle": 0.2,  # 8.0
#     # "slope_default_deg": 0.0,
#     # "Crr_default": 0.012,
#     "use_svpwm": True,
#     "include_deadtime": True,
#     "deadtime": 2.0e-6,
#     "plot_pwm_zoom_start": 0.295,
#     "plot_pwm_num_periods": 100,
#     "pwm_plot_samples": 30000,
#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",
#     # schedules
#     "t_steps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
#     # "omega_steps": [0.0, 50.0, 10.0],
#     "omega_steps": [rpm_to_rad_s(0.0), rpm_to_rad_s(1500.0),
#                     rpm_to_rad_s(1000.0), rpm_to_rad_s(1600.0),
#                     rpm_to_rad_s(1600.0), rpm_to_rad_s(3200.0),
#                     rpm_to_rad_s(2200.0), rpm_to_rad_s(3200.0),
#                     rpm_to_rad_s(0.0)],
#     "mass_step_times": [0.0],
#     "mass_step_values": [5.0],
#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [0.0],
#     "crr_step_times": [0.0],
#     "crr_step_values": [0.015],
#     "use_mtpa": False,
#     "use_field_weakening": False,
# }
# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "spmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     "tfinal": 0.1,           # Figure 3(a),(b) are 0–0.05 s
#     "Imax": 40.0,            # to allow up to about 15 Nm torque reference
#     "P_rated": 0.0,          # disable power clamp unless you really want it
#     "poles": 10,             # p = 5 pole pairs => 10 poles
#     "Vdc": 300.0,            # +150 to -150 dc link => 300 V total
#     "phi_f_ref": 0.05028,    # Vs = Wb
#     "R_s_ref": 0.43,         # ohm
#     "J_motor": 0.0006329,    # kg m^2

#     # make vehicle model as neutral as possible
#     "wheel_radius": 1.0,
#     "gear_ratio": 1.0,
#     "f_pwm": 10000.0,        # paper says max switching frequency 10 kHz

#     # advanced
#     "f": 50.0,               # not critical in closed-loop rotor FOC
#     "V_LL": 300.0 / np.sqrt(2),   # optional placeholder; not dominant in FOC mode
#     "L_ls": 0.00172,         # 1.72 mH
#     "L_A": 0.0,
#     "L_B": 0.0,
#     "I_rated": 12.7,         # approx from rated torque and flux
#     "Ld_inf_factor": 1.0,    # for SPMSM keep saturation model effectively flat
#     "Lq_inf_factor": 1.0,
#     "I_Ld_sat_factor": 1.0,
#     "I_Lq_sat_factor": 1.0,

#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,
#     "alpha_phi": 0.0010,
#     "C_th_s": 200.0,
#     "C_th_m": 400.0,
#     "R_th_sa": 0.4,
#     "R_th_sm": 0.8,
#     "R_th_ma": 1.2,
#     "T_amb": 25.0,
#     "T_warn_stator": 120.0,
#     "T_fault_stator": 160.0,
#     "T_warn_magnet": 100.0,
#     "T_fault_magnet": 140.0,
#     "T_demag": 120.0,
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 0.0,
#     "core_volume_m3": 0.0,
#     "B_iron_rated": 0.0,

#     # neutralize vehicle-road model
#     "rho_air": 1.225,
#     "drivetrain_eff": 1.0,
#     "Cd": 0.0,
#     "A_front": 0.0,
#     "b_vehicle": 0.0,

#     "use_svpwm": True,
#     "include_deadtime": True,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 0.002,
#     "plot_pwm_num_periods": 30,
#     "pwm_plot_samples": 10000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     # schedules
#     "t_steps": [0.0],
#     "omega_steps": [rpm_to_rad_s(1500.0)],   # paper case: 1500 r/min
#     "mass_step_times": [0.0],
#     "mass_step_values": [0.0],
#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [0.0],
#     "crr_step_times": [0.0],
#     "crr_step_values": [0.0],

#     "use_mtpa": False,
#     "use_field_weakening": False,
# }

# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "spmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     "tfinal": 1.0,
#     "Imax": 4.0,                 # paper says startup current reaches 4 A
#     "P_rated": 0.0,              # disable power clamp unless you know rated power
#     "poles": 8,                  # 4 pole pairs
#     "Vdc": 48.0,                 # not given in screenshot; choose a modest lab DC bus
#     "phi_f_ref": 0.00838,        # from table
#     "R_s_ref": 0.89,             # from table
#     "J_motor": 1.0e-5,           # not given; small lab motor guess

#     # make load close to no-load
#     "wheel_radius": 1.0,
#     "gear_ratio": 1.0,
#     "f_pwm": 20000.0,

#     # advanced
#     "f": 50.0,
#     "V_LL": 24.0,                # not given; placeholder only
#     "L_ls": 0.00062,             # from table
#     "L_A": 0.0,
#     "L_B": 0.0,
#     "I_rated": 2.5,              # steady current amplitude mentioned in text
#     "Ld_inf_factor": 1.0,        # keep SPMSM inductance flat unless you know saturation
#     "Lq_inf_factor": 1.0,
#     "I_Ld_sat_factor": 1.0,
#     "I_Lq_sat_factor": 1.0,

#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,
#     "alpha_phi": 0.0010,
#     "C_th_s": 200.0,
#     "C_th_m": 400.0,
#     "R_th_sa": 0.4,
#     "R_th_sm": 0.8,
#     "R_th_ma": 1.2,
#     "T_amb": 25.0,
#     "T_warn_stator": 120.0,
#     "T_fault_stator": 160.0,
#     "T_warn_magnet": 100.0,
#     "T_fault_magnet": 140.0,
#     "T_demag": 120.0,
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 0.0,
#     "core_volume_m3": 0.0,
#     "B_iron_rated": 0.0,

#     # near no-load operation
#     "rho_air": 1.225,
#     "drivetrain_eff": 1.0,
#     "Cd": 0.0,
#     "A_front": 0.0,
#     "b_vehicle": 0.0,

#     "use_svpwm": False,
#     "include_deadtime": False,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 0.02,
#     "plot_pwm_num_periods": 40,
#     "pwm_plot_samples": 12000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     # schedules
#     "t_steps": [0.0, 0.02],
#     "omega_steps": [rpm_to_rad_s(0.0), rpm_to_rad_s(1200.0)],

#     "mass_step_times": [0.0],
#     "mass_step_values": [0.0],

#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [0.0],

#     "crr_step_times": [0.0],
#     "crr_step_values": [0.0],

#     "use_mtpa": False,
#     "use_field_weakening": False,
# }

# # simulation of article https://www.sciencedirect.com/science/article/pii/S2215098616311119
# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "ipmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     "tfinal": 20.0,
#     "Imax": 20.0,
#     "P_rated": 2500.0,
#     "poles": 4,                  # 2 pole pairs
#     "Vdc": 400.0,
#     "phi_f_ref": 0.175,
#     "R_s_ref": 0.2,
#     "J_motor": 0.089,

#     # load-side approximation of the paper's 100% load case
#     "wheel_radius": 0.3,
#     "gear_ratio": 1.0,
#     "f_pwm": 10000.0,

#     # advanced
#     "f": 60.0,
#     "V_LL": 250.0,
#     "L_ls": 0.0085,
#     "L_A": 0.0005,
#     "L_B": 0.0005,
#     "I_rated": 20.0,
#     "Ld_inf_factor": 0.8,
#     "Lq_inf_factor": 0.8,
#     "I_Ld_sat_factor": 1.0,
#     "I_Lq_sat_factor": 1.0,

#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,
#     "alpha_phi": 0.0010,
#     "C_th_s": 200.0,
#     "C_th_m": 400.0,
#     "R_th_sa": 0.4,
#     "R_th_sm": 0.8,
#     "R_th_ma": 1.2,
#     "T_amb": 25.0,
#     "T_warn_stator": 120.0,
#     "T_fault_stator": 160.0,
#     "T_warn_magnet": 100.0,
#     "T_fault_magnet": 140.0,
#     "T_demag": 120.0,
#     "phi_f_min_factor": 0.20,

#     # Iron (core) loss model — Steinmetz-style, from Mi/Slemon/Bonert
#     # (IEEE Trans. Ind. Appl., 2003), p_iron = k_h*B^beta*omega_s + k_e*B^2*omega_s^2
#     "include_iron_loss": False,
#     "k_h_iron": 48.0,          # hysteresis coefficient, typical range 40-55
#     "k_e_iron": 0.055,         # eddy-current coefficient, typical range 0.04-0.07
#     "beta_iron": 2.0,          # Steinmetz exponent, typical range 1.8-2.2
#     "core_volume_m3": 0.002,   # lumped stator core volume used to scale W/m3 -> W
#     "B_iron_rated": 1.3,       # core flux density [T] corresponding to phi_f_ref

#     # neutral vehicle-side losses so the run is dominated by the chosen load proxy
#     "rho_air": 1.225,
#     "drivetrain_eff": 1.0,
#     "Cd": 0.0,
#     "A_front": 0.0,
#     "b_vehicle": 0.0,

#     "use_svpwm": False,
#     "include_deadtime": False,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 9.5,
#     "plot_pwm_num_periods": 100,
#     "pwm_plot_samples": 30000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     # schedules
#     "t_steps": [0.0],
#     "omega_steps": [rpm_to_rad_s(1800.0)],

#     "mass_step_times": [0.0, 10.0],
#     # 100%
#     "mass_step_values": [0.0, 40.0],
#     # # 75%
#     # "mass_step_values": [0.0, 30.0]
#     # # 50%
#     # "mass_step_values": [0.0, 20.0]
#     # # 25%
#     # "mass_step_values": [0.0, 10.0]

#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [1.0],

#     "crr_step_times": [0.0],
#     "crr_step_values": [0.012],

#     "use_mtpa": True,
#     "use_field_weakening": False,
# }

# simulation on pages 245-246
# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "ipmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     # screenshot time axis goes to 0.2 s
#     "tfinal": 0.2,
#     "Imax": 6.0,                 # inferred from q-current peak ~5 A
#     "P_rated": 0.0,              # disable power clamp; not shown in screenshot
#     "poles": 4,                  # 2 pole pairs (fits screenshot text and typical IPMSM example)
#     "Vdc": 300.0,                # assumed; high enough to avoid artificial voltage starvation
#     "phi_f_ref": 0.38,           # assumed to get ~6 Nm with iq~5 A and id<0
#     "R_s_ref": 0.2,              # assumed
#     "J_motor": 1.0e-3,           # assumed to get acceleration to ~1500 rpm in ~0.08 s

#     # make it a pure motor test, not a vehicle test
#     "wheel_radius": 1.0,
#     "gear_ratio": 1.0,
#     "f_pwm": 20000.0,

#     # advanced
#     "f": 60.0,                   # assumed
#     "V_LL": 220.0,               # assumed
#     # choose Ld ≈ 8.5 mH and Lq ≈ 10 mH
#     "L_ls": 0.0085,
#     "L_A": 0.0003333333333,
#     "L_B": 0.0003333333333,
#     "I_rated": 5.0,              # inferred from current plots
#     "Ld_inf_factor": 1.0,        # keep inductance nearly constant unless you want saturation
#     "Lq_inf_factor": 1.0,
#     "I_Ld_sat_factor": 1.0,
#     "I_Lq_sat_factor": 1.0,

#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,
#     "alpha_phi": 0.0010,
#     "C_th_s": 200.0,
#     "C_th_m": 400.0,
#     "R_th_sa": 0.4,
#     "R_th_sm": 0.8,
#     "R_th_ma": 1.2,
#     "T_amb": 25.0,
#     "T_warn_stator": 120.0,
#     "T_fault_stator": 160.0,
#     "T_warn_magnet": 100.0,
#     "T_fault_magnet": 140.0,
#     "T_demag": 120.0,
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 0.0,
#     "core_volume_m3": 0.0,
#     "B_iron_rated": 0.0,

#     # no external vehicle/load effects in the screenshot
#     "rho_air": 1.225,
#     "drivetrain_eff": 1.0,
#     "Cd": 0.0,
#     "A_front": 0.0,
#     "b_vehicle": 0.0,

#     # screenshot says overall simulation block not including inverter
#     "use_svpwm": False,
#     "include_deadtime": False,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 0.05,
#     "plot_pwm_num_periods": 40,
#     "pwm_plot_samples": 12000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     # speed command appears to step around 0.05 s
#     "t_steps": [0.0, 0.05],
#     "omega_steps": [rpm_to_rad_s(0.0), rpm_to_rad_s(1500.0)],

#     # no mechanical load visible in the screenshot
#     "mass_step_times": [0.0],
#     "mass_step_values": [0.0],

#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [0.0],

#     "crr_step_times": [0.0],
#     "crr_step_values": [0.0],

#     # screenshot includes MTPA block
#     "use_mtpa": True,
#     "use_field_weakening": False,
# }

# spmsm - 1200rpm speed step
# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "spmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     "tfinal": 0.2,
#     "Imax": 130.0,               # inferred from left current plot (~125 A q-current)
#     "P_rated": 0.0,
#     "poles": 4,                  # assumed
#     "Vdc": 300.0,
#     "phi_f_ref": 0.015,          # assumed to get ~5–6 Nm with large iq
#     "R_s_ref": 0.05,             # assumed low for high-current case
#     "J_motor": 1.0e-3,           # assumed

#     "wheel_radius": 1.0,
#     "gear_ratio": 1.0,
#     "f_pwm": 20000.0,

#     "f": 60.0,
#     "V_LL": 220.0,
#     "L_ls": 0.00062,             # chosen small, matches high-current SPMSM-like behavior
#     "L_A": 0.0,
#     "L_B": 0.0,
#     "I_rated": 125.0,
#     "Ld_inf_factor": 1.0,
#     "Lq_inf_factor": 1.0,
#     "I_Ld_sat_factor": 1.0,
#     "I_Lq_sat_factor": 1.0,

#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,
#     "alpha_phi": 0.0010,
#     "C_th_s": 200.0,
#     "C_th_m": 400.0,
#     "R_th_sa": 0.4,
#     "R_th_sm": 0.8,
#     "R_th_ma": 1.2,
#     "T_amb": 25.0,
#     "T_warn_stator": 120.0,
#     "T_fault_stator": 160.0,
#     "T_warn_magnet": 100.0,
#     "T_fault_magnet": 140.0,
#     "T_demag": 120.0,
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 0.0,
#     "core_volume_m3": 0.0,
#     "B_iron_rated": 0.0,

#     "rho_air": 1.225,
#     "drivetrain_eff": 1.0,
#     "Cd": 0.0,
#     "A_front": 0.0,
#     "b_vehicle": 0.0,

#     "use_svpwm": False,
#     "include_deadtime": False,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 0.05,
#     "plot_pwm_num_periods": 40,
#     "pwm_plot_samples": 12000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     "t_steps": [0.0, 0.05],
#     "omega_steps": [rpm_to_rad_s(0.0), rpm_to_rad_s(1200.0)],

#     "mass_step_times": [0.0],
#     "mass_step_values": [0.0],

#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [0.0],

#     "crr_step_times": [0.0],
#     "crr_step_values": [0.0],

#     "use_mtpa": False,
#     "use_field_weakening": False,
# }

# real EV
DEFAULTS: Dict[str, Any] = {
    # main
    "machine_type": "ipmsm",
    "drive_mode": "closed_loop_foc",
    "frame_mode": "rotor_foc",

    "tfinal": 20.0,
    "Imax": 150.0,               # peak inverter current limit
    "P_rated": 15000.0,          # 15 kW rated mechanical power
    "poles": 8,                  # 4 pole pairs, typical for EV traction
    "Vdc": 400.0,                # 400 V DC bus, standard EV voltage
    "phi_f_ref": 0.08,           # flux linkage, typical for 15 kW IPMSM
    "R_s_ref": 0.02,             # low stator resistance for traction motor
    "J_motor": 0.02,             # rotor inertia kg*m^2

    # drivetrain
    "wheel_radius": 0.32,        # m, typical passenger car wheel
    "gear_ratio": 8.0,           # single-speed reduction
    "f_pwm": 10000.0,            # 10 kHz switching frequency

    # advanced
    "f": 50.0,
    "V_LL": 230.0,
    "L_ls": 0.0002,              # H, leakage inductance
    "L_A": 0.00015,              # H, saliency term
    "L_B": 0.00008,              # H, saliency term (Lq > Ld for IPMSM)
    "I_rated": 100.0,            # A, continuous rated current
    "Ld_inf_factor": 0.6,        # Ld drops to 60% at high current
    "Lq_inf_factor": 0.5,        # Lq drops to 50% at high current
    "I_Ld_sat_factor": 0.8,      # saturation starts at 0.8 * I_rated
    "I_Lq_sat_factor": 0.6,      # Lq saturates earlier

    # thermal - realistic traction motor
    "T_ref": 25.0,
    "alpha_cu": 0.00393,         # copper temperature coefficient
    "alpha_phi": 0.0011,         # NdFeB magnet flux temperature coefficient
    "C_th_s": 800.0,             # J/K stator thermal capacitance
    "C_th_m": 400.0,             # J/K magnet thermal capacitance
    "R_th_sa": 0.15,             # K/W stator to ambient
    "R_th_sm": 0.25,             # K/W stator to magnet
    "R_th_ma": 0.40,             # K/W magnet to ambient
    "T_amb": 40.0,               # degC, warm ambient (engine bay)
    "T_warn_stator": 130.0,      # degC
    "T_fault_stator": 180.0,     # degC
    "T_warn_magnet": 120.0,      # degC
    "T_fault_magnet": 150.0,     # degC
    "T_demag": 140.0,            # degC, NdFeB irreversible demagnetization
    "phi_f_min_factor": 0.20,
    "include_iron_loss": False,
    # "k_h_iron": 0.0,
    # "k_e_iron": 0.0,
    # "beta_iron": 0.0,
    # "core_volume_m3": 0.0,
    # "B_iron_rated": 0.0,
    "k_h_iron": 48.0,
    "k_e_iron": 0.055,
    "beta_iron": 2.0,
    "core_volume_m3": 0.002,
    "B_iron_rated": 1.3,

    # vehicle - small passenger car
    "rho_air": 1.225,
    "drivetrain_eff": 0.95,
    "Cd": 0.30,                  # drag coefficient, typical hatchback
    "A_front": 2.2,              # m^2, frontal area
    "b_vehicle": 5.0,            # Ns/m, viscous damping

    "use_svpwm": True,
    "include_deadtime": True,
    "deadtime": 2.0e-6,

    "plot_pwm_zoom_start": 0.05,
    "plot_pwm_num_periods": 40,
    "pwm_plot_samples": 12000,

    "rtol": 1e-6,
    "atol": 1e-8,
    "solver_method": "Radau",

    # schedules - city drive cycle approximation
    # start stationary, accelerate to 50 km/h, hold, then 80 km/h
    "t_steps": [0.0, 1.0, 12.0],
    "omega_steps": [
        rpm_to_rad_s(0.0),
        rpm_to_rad_s(1060.0),    # ~50 km/h at wheel with gear ratio 8
        rpm_to_rad_s(1697.0),    # ~80 km/h at wheel with gear ratio 8
    ],

    # vehicle mass schedule - starts with driver only, passenger added
    "mass_step_times": [0.0, 10.0],
    "mass_step_values": [1400.0, 1550.0],  # kg, typical small car + passenger

    # road slope schedule - flat then uphill
    "slope_step_times": [0.0, 8.0],
    "slope_step_values_deg": [0.0, 5.0],   # deg, 5 degree incline

    # rolling resistance schedule - tarmac then rough road
    "crr_step_times": [0.0, 14.0],
    "crr_step_values": [0.012, 0.018],     # tarmac then worn road surface

    "use_mtpa": True,
    "use_field_weakening": True,
}

# inductance effect - current dependant:
# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "ipmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     "tfinal": 10.0,
#     "Imax": 150.0,               # peak inverter current limit
#     "P_rated": 15000.0,          # 15 kW rated mechanical power
#     "poles": 8,                  # 4 pole pairs, typical for EV traction
#     "Vdc": 400.0,                # 400 V DC bus, standard EV voltage
#     "phi_f_ref": 0.08,           # flux linkage, typical for 15 kW IPMSM
#     "R_s_ref": 0.02,             # low stator resistance for traction motor
#     "J_motor": 0.02,             # rotor inertia kg*m^2

#     # drivetrain
#     "wheel_radius": 0.32,        # m, typical passenger car wheel
#     "gear_ratio": 8.0,           # single-speed reduction
#     "f_pwm": 10000.0,            # 10 kHz switching frequency

#     # advanced
#     "f": 50.0,
#     "V_LL": 230.0,
#     # "L_ls": 0.0002,              # H, leakage inductance
#     # "L_A": 0.00015,              # H, saliency term
#     # "L_B": 0.00008,              # H, saliency term (Lq > Ld for IPMSM)
#     "L_ls": 0.0002,                
#     "L_A": 0.00005,                
#     "L_B": 0.0001167,  

#     "I_rated": 100.0,            # A, continuous rated current
#     # "Ld_inf_factor": 0.6,        # Ld drops to 60% at high current
#     # "Lq_inf_factor": 0.5,        # Lq drops to 50% at high current
#     # "I_Ld_sat_factor": 0.8,      # saturation starts at 0.8 * I_rated
#     # "I_Lq_sat_factor": 0.6,      # Lq saturates earlier
#     "Ld_inf_factor": 0.55,      # d-axis floor at 55% of Ld0 -- visible K_pd/gain effect
#     "Lq_inf_factor": 0.30,      # q-axis floor at 30% of Lq0 -- strong reluctance-torque effect (keep close to your original 0.35)
#     "I_Ld_sat_factor": 0.8,     # I_Ld_sat = 80 A -- knee now below Imax
#     "I_Lq_sat_factor": 0.5,     # I_Lq_sat = 50 A -- knee now below Imax

#     # thermal - realistic traction motor
#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,         # copper temperature coefficient
#     "alpha_phi": 0.0011,         # NdFeB magnet flux temperature coefficient
#     "C_th_s": 800.0,             # J/K stator thermal capacitance
#     "C_th_m": 400.0,             # J/K magnet thermal capacitance
#     "R_th_sa": 0.15,             # K/W stator to ambient
#     "R_th_sm": 0.25,             # K/W stator to magnet
#     "R_th_ma": 0.40,             # K/W magnet to ambient
#     "T_amb": 40.0,               # degC, warm ambient (engine bay)
#     "T_warn_stator": 130.0,      # degC
#     "T_fault_stator": 180.0,     # degC
#     "T_warn_magnet": 120.0,      # degC
#     "T_fault_magnet": 150.0,     # degC
#     "T_demag": 140.0,            # degC, NdFeB irreversible demagnetization
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 2.0,
#     "core_volume_m3": 0.001,
#     "B_iron_rated": 1.0,

#     # vehicle - small passenger car
#     "rho_air": 1.225,
#     "drivetrain_eff": 0.95,
#     "Cd": 0.30,                  # drag coefficient, typical hatchback
#     "A_front": 2.2,              # m^2, frontal area
#     "b_vehicle": 5.0,            # Ns/m, viscous damping

#     "use_svpwm": True,
#     "include_deadtime": True,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 0.05,
#     "plot_pwm_num_periods": 40,
#     "pwm_plot_samples": 12000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     "t_steps": [0.0],
#     "omega_steps": [rpm_to_rad_s(1500.0)],   # pick so steady-state |i_s| sits near Imax

#     "mass_step_times": [0.0],
#     "mass_step_values": [1500.0],            # single value, no mid-run change

#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [5.0],          # fixed grade -- keeps steady torque demand high
#                                              # so the current stays saturated, not just spikes briefly

#     "crr_step_times": [0.0],
#     "crr_step_values": [0.015],              # single value, no mid-run change

#     "use_mtpa": True,
#     "use_field_weakening": True,
# }

# inductance effect - constant:
# DEFAULTS: Dict[str, Any] = {
#     # main
#     "machine_type": "ipmsm",
#     "drive_mode": "closed_loop_foc",
#     "frame_mode": "rotor_foc",

#     "tfinal": 10.0,
#     "Imax": 150.0,               # peak inverter current limit
#     "P_rated": 15000.0,          # 15 kW rated mechanical power
#     "poles": 8,                  # 4 pole pairs, typical for EV traction
#     "Vdc": 400.0,                # 400 V DC bus, standard EV voltage
#     "phi_f_ref": 0.08,           # flux linkage, typical for 15 kW IPMSM
#     "R_s_ref": 0.02,             # low stator resistance for traction motor
#     "J_motor": 0.02,             # rotor inertia kg*m^2

#     # drivetrain
#     "wheel_radius": 0.32,        # m, typical passenger car wheel
#     "gear_ratio": 8.0,           # single-speed reduction
#     "f_pwm": 10000.0,            # 10 kHz switching frequency

#     # advanced
#     "f": 50.0,
#     "V_LL": 230.0,
#     # "L_ls": 0.0002,              # H, leakage inductance
#     # "L_A": 0.00015,              # H, saliency term
#     # "L_B": 0.00008,              # H, saliency term (Lq > Ld for IPMSM)
#     "L_ls": 0.0002,                
#     "L_A": 0.00005,                
#     "L_B": 0.0001167,  

#     "I_rated": 100.0,            # A, continuous rated current
#     # "Ld_inf_factor": 0.6,        # Ld drops to 60% at high current
#     # "Lq_inf_factor": 0.5,        # Lq drops to 50% at high current
#     # "I_Ld_sat_factor": 0.8,      # saturation starts at 0.8 * I_rated
#     # "I_Lq_sat_factor": 0.6,      # Lq saturates earlier
#     # "Ld_inf_factor": 1.0,          # d-axis barely saturates -- floor is 90% of Ld0
#     # "Lq_inf_factor": 1.0,          # q-axis saturates hard -- floor is 35% of Lq0
#     # "I_Ld_sat_factor": 3.0,        # d-axis saturation develops very gradually (I_Ld_sat = 300 A)
#     # "I_Lq_sat_factor": 2.4,        # q-axis saturation develops faster (I_Lq_sat = 240 A)
#     "Ld_inf_factor": 1.0,      # d-axis floor at 55% of Ld0 -- visible K_pd/gain effect
#     "Lq_inf_factor": 1.0,      # q-axis floor at 30% of Lq0 -- strong reluctance-torque effect (keep close to your original 0.35)
#     "I_Ld_sat_factor": 0.8,     # I_Ld_sat = 80 A -- knee now below Imax
#     "I_Lq_sat_factor": 0.5,     # I_Lq_sat = 50 A -- knee now below Imax

#     # thermal - realistic traction motor
#     "T_ref": 25.0,
#     "alpha_cu": 0.00393,         # copper temperature coefficient
#     "alpha_phi": 0.0011,         # NdFeB magnet flux temperature coefficient
#     "C_th_s": 800.0,             # J/K stator thermal capacitance
#     "C_th_m": 400.0,             # J/K magnet thermal capacitance
#     "R_th_sa": 0.15,             # K/W stator to ambient
#     "R_th_sm": 0.25,             # K/W stator to magnet
#     "R_th_ma": 0.40,             # K/W magnet to ambient
#     "T_amb": 40.0,               # degC, warm ambient (engine bay)
#     "T_warn_stator": 130.0,      # degC
#     "T_fault_stator": 180.0,     # degC
#     "T_warn_magnet": 120.0,      # degC
#     "T_fault_magnet": 150.0,     # degC
#     "T_demag": 140.0,            # degC, NdFeB irreversible demagnetization
#     "phi_f_min_factor": 0.20,
#     "include_iron_loss": False,
#     "k_h_iron": 0.0,
#     "k_e_iron": 0.0,
#     "beta_iron": 2.0,
#     "core_volume_m3": 0.001,
#     "B_iron_rated": 1.0,

#     # vehicle - small passenger car
#     "rho_air": 1.225,
#     "drivetrain_eff": 0.95,
#     "Cd": 0.30,                  # drag coefficient, typical hatchback
#     "A_front": 2.2,              # m^2, frontal area
#     "b_vehicle": 5.0,            # Ns/m, viscous damping

#     "use_svpwm": True,
#     "include_deadtime": True,
#     "deadtime": 2.0e-6,

#     "plot_pwm_zoom_start": 0.05,
#     "plot_pwm_num_periods": 40,
#     "pwm_plot_samples": 12000,

#     "rtol": 1e-6,
#     "atol": 1e-8,
#     "solver_method": "Radau",

#     "t_steps": [0.0],
#     "omega_steps": [rpm_to_rad_s(1500.0)],   # pick so steady-state |i_s| sits near Imax

#     "mass_step_times": [0.0],
#     "mass_step_values": [1500.0],            # single value, no mid-run change

#     "slope_step_times": [0.0],
#     "slope_step_values_deg": [5.0],          # fixed grade -- keeps steady torque demand high
#                                             # so the current stays saturated, not just spikes briefly

#     "crr_step_times": [0.0],
#     "crr_step_values": [0.015],              # single value, no mid-run change

#     "use_mtpa": True,
#     "use_field_weakening": True,
# }

@dataclass
class FieldSpec:
    key: str
    label: str
    kind: str
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    allow_negative: bool = True
    options: Optional[List[str]] = None
    help_text: str = ""
    readonly: bool = False


MAIN_FIELDS: List[FieldSpec] = [
    FieldSpec("use_svpwm", "Use SVPWM", "bool", DEFAULTS["use_svpwm"]),
    FieldSpec("include_deadtime", "Include dead-time", "bool", DEFAULTS["include_deadtime"]),
    FieldSpec("machine_type", "Machine type", "enum", DEFAULTS["machine_type"], options=["ipmsm", "spmsm"]),
    FieldSpec("drive_mode", "Drive mode", "enum", DEFAULTS["drive_mode"], options=["closed_loop_foc", "open_loop_abc"]),
    FieldSpec("frame_mode", "Frame mode", "enum", DEFAULTS["frame_mode"], options=["rotor_foc", "stator"]),
    # FieldSpec("region", "Region", "enum", DEFAULTS["region"], options=["EU", "US"]),
    FieldSpec("tfinal", "Simulation end time [s]", "float", DEFAULTS["tfinal"], minimum=0.001, allow_negative=False),
    FieldSpec("Imax", "Max stator current [A]", "float", DEFAULTS["Imax"], minimum=0.001, allow_negative=False),
    FieldSpec("poles", "Total poles", "int", DEFAULTS["poles"], minimum=2, allow_negative=False),
    # FieldSpec("Vdc", "DC bus voltage [V]", "float", DEFAULTS["Vdc"], minimum=1.0, allow_negative=False),
    # FieldSpec("phi_f_ref", "PM flux linkage at T_ref [Wb]", "float", DEFAULTS["phi_f_ref"], minimum=1e-9, allow_negative=False),
    # FieldSpec("R_s_ref", "Stator resistance at T_ref [Ω]", "float", DEFAULTS["R_s_ref"], minimum=1e-9, allow_negative=False),
    # FieldSpec("J_motor", "Motor inertia [kg·m²]", "float", DEFAULTS["J_motor"], minimum=1e-9, allow_negative=False),
    # FieldSpec("mass_default", "Default vehicle mass [kg]", "float", DEFAULTS["mass_default"], minimum=0.0, allow_negative=False),
    # FieldSpec("wheel_radius", "Wheel radius [m]", "float", DEFAULTS["wheel_radius"], minimum=1e-6, allow_negative=False),
    # FieldSpec("gear_ratio", "Gear ratio", "float", DEFAULTS["gear_ratio"], minimum=1e-9, allow_negative=False),
    FieldSpec("f_pwm", "PWM frequency [Hz]", "float", DEFAULTS["f_pwm"], minimum=1.0, allow_negative=False),
    FieldSpec("use_mtpa", "Use MTPA", "bool", DEFAULTS["use_mtpa"]),
    FieldSpec("use_field_weakening", "Use field weakening", "bool", DEFAULTS["use_field_weakening"]),
]

# advanced
ADV_FIELDS: List[FieldSpec] = [
    # FieldSpec("region", "Region", "enum", DEFAULTS["region"], options=["EU", "US"]),
    FieldSpec("Vdc", "DC bus voltage [V]", "float", DEFAULTS["Vdc"], minimum=1.0, allow_negative=False),
    FieldSpec("phi_f_ref", "PM flux linkage at T_ref [Wb]", "float", DEFAULTS["phi_f_ref"], minimum=1e-9, allow_negative=False),
    FieldSpec("R_s_ref", "Stator resistance at T_ref [Ω]", "float", DEFAULTS["R_s_ref"], minimum=1e-9, allow_negative=False),
    FieldSpec("J_motor", "Motor inertia [kg·m²]", "float", DEFAULTS["J_motor"], minimum=1e-9, allow_negative=False),
    # FieldSpec("mass_default", "Default vehicle mass [kg]", "float", DEFAULTS["mass_default"], minimum=0.0, allow_negative=False),
    FieldSpec("wheel_radius", "Wheel radius [m]", "float", DEFAULTS["wheel_radius"], minimum=1e-6, allow_negative=False),
    FieldSpec("gear_ratio", "Gear ratio", "float", DEFAULTS["gear_ratio"], minimum=1e-9, allow_negative=False),
    FieldSpec("f", "Electrical frequency [Hz]", "float", DEFAULTS["f"], minimum=0.0, allow_negative=False),
    FieldSpec("V_LL", "Line-to-line RMS voltage [V]", "float", DEFAULTS["V_LL"], minimum=0.0, allow_negative=False),
    FieldSpec("L_ls", "Stator leakage inductance [H]", "float", DEFAULTS["L_ls"], minimum=1e-9, allow_negative=False),
    FieldSpec("L_A", "L_A [H]", "float", DEFAULTS["L_A"]),
    FieldSpec("L_B", "L_B [H]", "float", DEFAULTS["L_B"]),
    FieldSpec("I_rated", "Rated phase current [A]", "float", DEFAULTS["I_rated"], minimum=1e-9, allow_negative=False), # usually the current the motor can carry continuously without overheating under its rated conditions
    FieldSpec("P_rated", "Rated motor power [W]", "float", DEFAULTS["P_rated"], minimum=0.0, allow_negative=False),
    FieldSpec("Ld_inf_factor", "Ld_inf / Ld0", "float", DEFAULTS["Ld_inf_factor"], minimum=0.0, maximum=1.0, allow_negative=False),
    FieldSpec("Lq_inf_factor", "Lq_inf / Lq0", "float", DEFAULTS["Lq_inf_factor"], minimum=0.0, maximum=1.0, allow_negative=False),
    FieldSpec("I_Ld_sat_factor", "I_Ld_sat / I_rated", "float", DEFAULTS["I_Ld_sat_factor"], minimum=1e-9, allow_negative=False),
    FieldSpec("I_Lq_sat_factor", "I_Lq_sat / I_rated", "float", DEFAULTS["I_Lq_sat_factor"], minimum=1e-9, allow_negative=False),
    # temp ignored
    FieldSpec("T_ref", "Reference temperature [°C]", "float", DEFAULTS["T_ref"]),
    FieldSpec("alpha_cu", "Copper temp coefficient [1/°C]", "float", DEFAULTS["alpha_cu"], minimum=0.0, allow_negative=False),
    FieldSpec("alpha_phi", "PM flux temp coefficient [1/°C]", "float", DEFAULTS["alpha_phi"], minimum=0.0, allow_negative=False),
    FieldSpec("C_th_s", "Stator thermal capacitance [J/K]", "float", DEFAULTS["C_th_s"], minimum=1e-9, allow_negative=False),
    FieldSpec("C_th_m", "Magnet thermal capacitance [J/K]", "float", DEFAULTS["C_th_m"], minimum=1e-9, allow_negative=False),
    FieldSpec("R_th_sa", "Stator→ambient thermal resistance [K/W]", "float", DEFAULTS["R_th_sa"], minimum=1e-9, allow_negative=False),
    FieldSpec("R_th_sm", "Stator→magnet thermal resistance [K/W]", "float", DEFAULTS["R_th_sm"], minimum=1e-9, allow_negative=False),
    FieldSpec("R_th_ma", "Magnet→ambient thermal resistance [K/W]", "float", DEFAULTS["R_th_ma"], minimum=1e-9, allow_negative=False),
    FieldSpec("T_amb", "Ambient temperature [°C]", "float", DEFAULTS["T_amb"]),
    FieldSpec("T_warn_stator", "Stator warning temperature [°C]", "float", DEFAULTS["T_warn_stator"]),
    FieldSpec("T_fault_stator", "Stator fault temperature [°C]", "float", DEFAULTS["T_fault_stator"]),
    FieldSpec("T_warn_magnet", "Magnet warning temperature [°C]", "float", DEFAULTS["T_warn_magnet"]),
    FieldSpec("T_fault_magnet", "Magnet fault temperature [°C]", "float", DEFAULTS["T_fault_magnet"]),
    FieldSpec("T_demag", "Demag-risk temperature [°C]", "float", DEFAULTS["T_demag"]),
    FieldSpec("phi_f_min_factor", "phi_f_min / phi_f_ref", "float", DEFAULTS["phi_f_min_factor"], minimum=0.0, maximum=1.0, allow_negative=False),
    FieldSpec("include_iron_loss", "Include iron (core) losses", "bool", DEFAULTS["include_iron_loss"]),
    FieldSpec("k_h_iron", "Hysteresis loss coefficient k_h", "float", DEFAULTS["k_h_iron"], minimum=0.0, allow_negative=False),
    FieldSpec("k_e_iron", "Eddy-current loss coefficient k_e", "float", DEFAULTS["k_e_iron"], minimum=0.0, allow_negative=False),
    FieldSpec("beta_iron", "Steinmetz exponent β", "float", DEFAULTS["beta_iron"], minimum=1.0, maximum=3.0, allow_negative=False),
    FieldSpec("core_volume_m3", "Stator core volume [m³]", "float", DEFAULTS["core_volume_m3"], minimum=1e-9, allow_negative=False),
    FieldSpec("B_iron_rated", "Core flux density at rated flux [T]", "float", DEFAULTS["B_iron_rated"], minimum=1e-6, maximum=2.5, allow_negative=False),
    FieldSpec("rho_air", "Air density [kg/m³]", "float", DEFAULTS["rho_air"], minimum=0.0, allow_negative=False),
    FieldSpec("drivetrain_eff", "Drivetrain efficiency", "float", DEFAULTS["drivetrain_eff"], minimum=1e-9, maximum=1.0, allow_negative=False),
    FieldSpec("Cd", "Drag coefficient", "float", DEFAULTS["Cd"], minimum=0.0, allow_negative=False),
    FieldSpec("A_front", "Frontal area [m²]", "float", DEFAULTS["A_front"], minimum=0.0, allow_negative=False),
    FieldSpec("b_vehicle", "Viscous road coefficient [N/(m/s)]", "float", DEFAULTS["b_vehicle"], minimum=0.0, allow_negative=False),
    # FieldSpec("slope_default_deg", "Default slope [deg]", "float", DEFAULTS["slope_default_deg"]),
    # FieldSpec("Crr_default", "Default rolling resistance coefficient", "float", DEFAULTS["Crr_default"], minimum=0.0, allow_negative=False),
    # FieldSpec("use_svpwm", "Use SVPWM", "bool", DEFAULTS["use_svpwm"]),
    # FieldSpec("include_deadtime", "Include dead-time", "bool", DEFAULTS["include_deadtime"]),
    FieldSpec("deadtime", "Dead-time [s]", "float", DEFAULTS["deadtime"], minimum=0.0, allow_negative=False),
    FieldSpec("plot_pwm_num_periods", "PWM periods in zoom", "int", DEFAULTS["plot_pwm_num_periods"], minimum=1, allow_negative=False),
    FieldSpec("pwm_plot_samples", "PWM plot samples", "int", DEFAULTS["pwm_plot_samples"], minimum=100, allow_negative=False),
    FieldSpec("rtol", "Relative tolerance", "float", DEFAULTS["rtol"], minimum=1e-12, allow_negative=False),
    FieldSpec("atol", "Absolute tolerance", "float", DEFAULTS["atol"], minimum=1e-12, allow_negative=False),
    FieldSpec("solver_method", "solve_ivp method", "enum", DEFAULTS["solver_method"], options=["Radau"]),
]


CRR_GUIDE = [
    ("Smooth steel rail", 0.001),
    ("Good asphalt / concrete", 0.008),
    ("Typical passenger car on asphalt", 0.010),
    ("Rough asphalt", 0.015),
    ("Wet grass", 0.020),
    ("Gravel road", 0.025),
    ("Compacted dirt", 0.030),
    ("Cobblestones / stones", 0.035),
    ("Loose gravel", 0.050),
    ("Sand", 0.100),
]