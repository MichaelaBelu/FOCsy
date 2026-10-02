from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class PMSMConfig:
    machine_type: str
    drive_mode: str
    frame_mode: str

    use_mtpa: bool
    use_field_weakening: bool

    # region: str
    f: float
    V_LL: float
    Vdc: float

    L_ls: float
    L_A: float
    L_B: float

    I_rated: float
    P_rated: float
    Ld_inf_factor: float
    Lq_inf_factor: float
    I_Ld_sat_factor: float
    I_Lq_sat_factor: float

    T_ref: float
    alpha_cu: float
    alpha_phi: float
    C_th_s: float
    C_th_m: float
    R_th_sa: float
    R_th_sm: float
    R_th_ma: float
    T_amb: float
    R_s_ref: float
    phi_f_ref: float

    T_warn_stator: float
    T_fault_stator: float
    T_warn_magnet: float
    T_fault_magnet: float
    T_demag: float
    phi_f_min_factor: float

    include_iron_loss: bool
    k_h_iron: float
    k_e_iron: float
    beta_iron: float
    core_volume_m3: float
    B_iron_rated: float

    poles: int

    rho_air: float
    gear_ratio: float
    drivetrain_eff: float
    wheel_radius: float
    Cd: float
    A_front: float
    b_vehicle: float
    # mass_default: float
    # slope_default_deg: float
    # Crr_default: float

    use_svpwm: bool
    include_deadtime: bool
    f_pwm: float
    deadtime: float
    plot_pwm_zoom_start: float
    plot_pwm_num_periods: int
    pwm_plot_samples: int

    tfinal: float
    rtol: float
    atol: float
    solver_method: str

    J_motor: float
    Imax: float

    t_steps: np.ndarray
    omega_steps: np.ndarray
    mass_step_times: np.ndarray
    mass_step_values: np.ndarray
    slope_step_times: np.ndarray
    slope_step_values_deg: np.ndarray
    crr_step_times: np.ndarray
    crr_step_values: np.ndarray


def config_from_params(params: Dict[str, Any]) -> PMSMConfig:
    return PMSMConfig(
        machine_type=str(params["machine_type"]),
        drive_mode=str(params["drive_mode"]),
        frame_mode=str(params["frame_mode"]),
        use_mtpa=bool(params["use_mtpa"]),
        use_field_weakening=bool(params["use_field_weakening"]),
        # region=str(params["region"]),
        f=float(params["f"]),
        V_LL=float(params["V_LL"]),
        Vdc=float(params["Vdc"]),
        L_ls=float(params["L_ls"]),
        L_A=float(params["L_A"]),
        L_B=float(params["L_B"]),
        I_rated=float(params["I_rated"]),
        P_rated=float(params["P_rated"]),
        Ld_inf_factor=float(params["Ld_inf_factor"]),
        Lq_inf_factor=float(params["Lq_inf_factor"]),
        I_Ld_sat_factor=float(params["I_Ld_sat_factor"]),
        I_Lq_sat_factor=float(params["I_Lq_sat_factor"]),
        T_ref=float(params["T_ref"]),
        alpha_cu=float(params["alpha_cu"]),
        alpha_phi=float(params["alpha_phi"]),
        C_th_s=float(params["C_th_s"]),
        C_th_m=float(params["C_th_m"]),
        R_th_sa=float(params["R_th_sa"]),
        R_th_sm=float(params["R_th_sm"]),
        R_th_ma=float(params["R_th_ma"]),
        T_amb=float(params["T_amb"]),
        R_s_ref=float(params["R_s_ref"]),
        phi_f_ref=float(params["phi_f_ref"]),
        T_warn_stator=float(params["T_warn_stator"]),
        T_fault_stator=float(params["T_fault_stator"]),
        T_warn_magnet=float(params["T_warn_magnet"]),
        T_fault_magnet=float(params["T_fault_magnet"]),
        T_demag=float(params["T_demag"]),
        phi_f_min_factor=float(params["phi_f_min_factor"]),
        include_iron_loss=bool(params["include_iron_loss"]),
        k_h_iron=float(params["k_h_iron"]),
        k_e_iron=float(params["k_e_iron"]),
        beta_iron=float(params["beta_iron"]),
        core_volume_m3=float(params["core_volume_m3"]),
        B_iron_rated=float(params["B_iron_rated"]),
        poles=int(params["poles"]),
        rho_air=float(params["rho_air"]),
        gear_ratio=float(params["gear_ratio"]),
        drivetrain_eff=float(params["drivetrain_eff"]),
        wheel_radius=float(params["wheel_radius"]),
        Cd=float(params["Cd"]),
        A_front=float(params["A_front"]),
        b_vehicle=float(params["b_vehicle"]),
        # mass_default=float(params["mass_default"]),
        # slope_default_deg=float(params["slope_default_deg"]),
        # Crr_default=float(params["Crr_default"]),
        use_svpwm=bool(params["use_svpwm"]),
        include_deadtime=bool(params["include_deadtime"]),
        f_pwm=float(params["f_pwm"]),
        deadtime=float(params["deadtime"]),
        plot_pwm_zoom_start=float(params["plot_pwm_zoom_start"]),
        plot_pwm_num_periods=int(params["plot_pwm_num_periods"]),
        pwm_plot_samples=int(params["pwm_plot_samples"]),
        tfinal=float(params["tfinal"]),
        rtol=float(params["rtol"]),
        atol=float(params["atol"]),
        solver_method=str(params["solver_method"]),
        J_motor=float(params["J_motor"]),
        Imax=float(params["Imax"]),
        t_steps=np.array(params["t_steps"], dtype=float),
        omega_steps=np.array(params["omega_steps"], dtype=float),
        mass_step_times=np.array(params["mass_step_times"], dtype=float),
        mass_step_values=np.array(params["mass_step_values"], dtype=float),
        slope_step_times=np.array(params["slope_step_times"], dtype=float),
        slope_step_values_deg=np.array(params["slope_step_values_deg"], dtype=float),
        crr_step_times=np.array(params["crr_step_times"], dtype=float),
        crr_step_values=np.array(params["crr_step_values"], dtype=float),
    )
