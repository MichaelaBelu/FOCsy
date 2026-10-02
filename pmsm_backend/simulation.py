from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# numpy >= 2.0 renamed trapz to trapezoid; keep this working on either version.
_trapz = getattr(np, "trapezoid", None) or np.trapz

from pmsm_backend.config import PMSMConfig
from pmsm_backend.mtpa import MtpaLut
from pmsm_backend.fw_lut import FwLut
from pmsm_backend.transforms import (
    T_matrix,
    dq0_to_abc,
    dq_voltage_limit,
)
from pmsm_backend.pwm import (
    phase_refs_to_svpwm_pole_refs,
    pole_refs_to_duties,
    apply_deadtime_to_duties,
    duties_to_average_pole_voltages,
    pole_to_phase_voltages,
    triangular_carrier_from_time,
    make_center_aligned_gates_with_deadtime,
)

_MTPA_CACHE: Dict[tuple, Optional[MtpaLut]] = {}
_FW_CACHE: Dict[tuple, Optional[FwLut]] = {}

class PMSMSimulation:
    def get_mtpa_cache_key(self) -> tuple:
        """
        Returns a cache key containing only the parameters that materially affect the MTPA LUT.
        Rounded values avoid pointless rebuilds from tiny floating-point differences.
        """
        return (
            self.machine_type,
            int(self.polepairs),
            round(float(self.Imax), 8),
            round(float(self.L_ds), 10),
            round(float(self.L_qs), 10),
            round(float(self.Ld_inf), 10),
            round(float(self.Lq_inf), 10),
            round(float(self.I_Ld_sat), 8),
            round(float(self.I_Lq_sat), 8),
            round(float(self.phi_f_ref), 10),
        )

    def get_fw_cache_key(self) -> tuple:
        """
        Same idea as get_mtpa_cache_key(), plus Vmax -- unlike the MTPA
        point (torque-only, voltage-independent), the Regime-B FW table
        genuinely depends on the voltage limit, so a change in Vdc (and
        therefore Vmax) must invalidate the cached table too.
        """
        return self.get_mtpa_cache_key() + (round(float(self.Vmax), 8),)

    def __init__(self, cfg: PMSMConfig):
        self.cfg = cfg

        self.machine_type = cfg.machine_type  # "ipmsm" or "spmsm"
        # self.use_mtpa = cfg.use_mtpa
        # self.use_field_weakening = cfg.use_field_weakening
        self.use_mtpa = bool(cfg.use_mtpa)
        self.use_field_weakening = bool(cfg.use_field_weakening)

        # SPMSM: disable field weakening in this model
        if self.machine_type == "spmsm":
            self.use_field_weakening = False

        # steps with leaner flow:
        # self.tau_mass = 1.0      # s
        # self.tau_slope = 0.5     # s
        # self.tau_crr = 0.3       # s
        # Smooth schedule transitions.
        # Smaller tau = faster transition.
            # tau = 0.50 s  -> slow smooth transition
            # tau = 0.15 s  -> noticeably faster but still smooth
            # tau = 0.05 s  -> very fast smooth transition
            # tau = 0.01 s  -> almost step-like
        self.tau_mass = 0.15     # s
        self.tau_slope = 0.05    # s
        self.tau_crr = 0.05      # s

        self.P_rated = cfg.P_rated

        # self.tau_omega_ref = 0.1 # s
        # self.omega_ref_rate_up = 200.0   # rad/s^2
        # self.omega_ref_rate_down = 300.0 # rad/s^2
        self.tau_omega_ref = 0.005  # 0.02
        self.omega_ref_rate_up = 1e4  # 1000.0
        self.omega_ref_rate_down = 1e4  #1200.0

        # --------------------------- User settings --------------------------------
        # self.machine_type = cfg.machine_type  # "ipmsm" or "spmsm"
        # TODO: synchronous frame for IPMSM needs to use FOC

        # IMPORTANT:
        #  frame_mode="rotor_foc" for drive_mode="closed_loop_foc"
        #  frame_mode="stator" only for open-loop demonstrations

        # Choose what is applied at the stator:
        #  - "closed_loop_foc": apply dq voltages from PI current controllers (recommended)
        #  - "open_loop_abc"  : apply imposed sinusoidal abc voltages (fundamentally unstable for PMSM)
        self.drive_mode = cfg.drive_mode

        # Choose dq reference frame USED FOR CONTROL + MODEL EQUATIONS BELOW:
        #  - "rotor_foc" : theta_frame = theta_r_e (rotor electrical angle)  <-- this is what you want for FOC
        #  - "stator"    : theta_frame = omega_e * t (for demonstrations; not phase-locked)
        self.frame_mode = cfg.frame_mode

        if self.drive_mode == "closed_loop_foc" and self.frame_mode != "rotor_foc":
            raise ValueError("closed_loop_foc must use frame_mode='rotor_foc'")

        # ------------------- ELECTRICAL + MECHANICAL PARAMETERS (EDIT) -----------------
        # # DC bus / inverter limit (simple SVPWM dq magnitude limit)
        # Vdc = 300.0
        # Vmax = Vdc / np.sqrt(3)

        # For open-loop abc voltages:
        '''
        V_LL = phase to phase
             = line to line inverter's usual limit, voltmeter across 2 phases
             = This is what the inverter directly creates at its terminals
        V_phase = This is the voltage that actually appears across one stator winding
                = Exists only in star (Y) connection
        RMS is the effective DC-equivalent value of an AC signal in terms of power and heating.
        '''
        # self.region = cfg.region  # "EU" or "US"

        # if self.region == "EU":
        #     # 400 V-class industrial drive
        #     self.f = 50.0 if abs(cfg.f - 50.0) < 1e-15 else cfg.f
        #     self.V_LL = 400.0 if abs(cfg.V_LL - 400.0) < 1e-15 else cfg.V_LL
        #     self.Vdc = cfg.Vdc  # inverter DC bus (typical for 400 V drives)
        # elif self.region == "US":
        #     # 208 V-class low-voltage drive
        #     self.f = 60.0 if abs(cfg.f - 50.0) < 1e-15 else cfg.f
        #     self.V_LL = 208.0 if abs(cfg.V_LL - 400.0) < 1e-15 else cfg.V_LL
        #     self.Vdc = cfg.Vdc  # inverter DC bus (typical for 208 V drives)
        # else:
        self.f = cfg.f
        self.V_LL = cfg.V_LL
        self.Vdc = cfg.Vdc

        # inverter voltage capability (SVPWM)
        # also called base speed (omega_base)
        self.Vmax = self.Vdc / np.sqrt(3)  # max dq voltage magnitude (peak)

        # motor phase voltages
        self.V_phase = self.V_LL / np.sqrt(3)
        self.V_m = np.sqrt(2) * self.V_phase  # peak phase voltage

        # electrical frequency
        self.omega_e = 2.0 * np.pi * self.f

        # leakage + saliency “construction”
        self.L_ls = cfg.L_ls
        self.L_A = cfg.L_A
        self.L_B = cfg.L_B

        # compute L_d and L_q as for IPMSM stator speed:
        # between eq. 4.109 and 4.110 page 199  
        '''
        L_ls: stator leakage inductance
            : some stator-produced flux goes through the main magnetic path and contributes to torque/energy conversion
            : some flux “leaks” locally around slots, teeth, end windings, etc.
            : that leakage part behaves like an extra inductance in series with the stator winding
        L_A, L_B: terms describing the rotor-position-dependent saliency > after transforming the abc 
          inductance matrix into dq coordinates, those combinations collapse into constant dq inductances
        L_A: how much useful main flux the stator can establish through the air gap and rotor iron
             the general strength of magnetic coupling in the machine
        L_B: how much the rotor geometry causes the inductance to change as the rotor turns
        '''
        self.L_ds = self.L_ls + 1.5 * (self.L_A - self.L_B)
        self.L_qs = self.L_ls + 1.5 * (self.L_A + self.L_B)

        # SPMSM special case: cylindrical rotor => Ld = Lq
        if self.machine_type == "spmsm":
            self.L_qs = self.L_ds

        ######################################################################################################################################################
        # MTPA LUT
        # They are the physical parameters of the inductance-vs-current
        # approximation and must come from startup motor data,
        # FEM, measurement, or datasheet fitting.
        #
        # Low-current values already come from your L_ds and L_qs.
        # High-current values are the inductances near the current limit.
        self.Ld0 = self.L_ds
        self.Lq0 = self.L_qs

        self.I_rated = cfg.I_rated  # motor rated current (typically the rated phase current, in amps)

        self.Ld_inf = cfg.Ld_inf_factor * self.Ld0  # strong d-axis saturation inductance; typical range is about 30% to 70% of Ld0
        self.Lq_inf = cfg.Lq_inf_factor * self.Lq0  # strong q-axis saturation inductance; typical range is about 40% to 80% of Lq0

        self.I_Ld_sat = cfg.I_Ld_sat_factor * self.I_rated  # d-axis saturation current scale; typical range is about 0.5 to 1.5 × rated phase current
        self.I_Lq_sat = cfg.I_Lq_sat_factor * self.I_rated  # q-axis saturation current scale; typical range is about 0.5 to 2.0 × rated phase current
        # If Ld_inf or Lq_inf is set too low, the model may make the motor appear
        # unrealistically saturated at high current. If I_L*_sat is set too small,
        # the inductance drops too early; if set too large, the inductance curve
        # stays too flat.
        ######################################################################################################################################################

        # IMPORTANT:
        #  The magnetic flux varies considerably with the operating temperature.
        # phi_f = 0.9  # Wb (example) # turning it into temperature dependant

        # Stator resistance
        # R_s = 2.0  # turning it into temperature dependant

        # temperature dependant variables:
        self.T_ref = cfg.T_ref  # degC
        self.alpha_cu = cfg.alpha_cu  # 1/degC
        self.alpha_phi = cfg.alpha_phi  # 1/degC  (NdFeB approx from your note)

        # TODO: C_th_s and C_th_m??
        self.C_th_s = cfg.C_th_s
        self.C_th_m = cfg.C_th_m
        self.R_th_sa = cfg.R_th_sa
        self.R_th_sm = cfg.R_th_sm
        self.R_th_ma = cfg.R_th_ma
        self.T_amb = cfg.T_amb

        # Iron (core) loss model — Steinmetz-style, from Mi/Slemon/Bonert
        # (IEEE Trans. Ind. Appl., 2003):
        #   p_iron = k_h*B^beta*omega_s + k_e*B^2*omega_s^2   [W/m^3]
        # Modeled as an extra heat source feeding the stator thermal node.
        self.include_iron_loss = cfg.include_iron_loss
        self.k_h_iron = cfg.k_h_iron
        self.k_e_iron = cfg.k_e_iron
        self.beta_iron = cfg.beta_iron
        self.core_volume_m3 = cfg.core_volume_m3
        self.B_iron_rated = cfg.B_iron_rated

        # ---- Iron-loss electrical branch (core-loss / "R_fe" current) -----------------
        # P_iron_arr above (Steinmetz) only heats the stator thermally -- it never draws
        # any extra current from the supply, so E_elec_in (computed purely from the
        # actually-solved v_d,i_d,v_q,i_q) has nothing in it to pay for that loss. That
        # makes the energy-balance residual go strongly negative whenever iron loss is
        # on (see energy-balance section below).
        #
        # The physically correct fix is the classical iron-loss equivalent-circuit model:
        # a core-loss resistance R_fe in parallel with the back-EMF branch, so the
        # terminal current actually drawn is i_d + i_Fe_d, i_q + i_Fe_q, where i_Fe is
        # the current that flows into R_fe (Ohm's law: i_Fe = E_backemf / R_fe).
        #
        # R_fe is calibrated ONCE here from the rated operating point (B_iron_rated at
        # the rated electrical frequency cfg.f), using the same Steinmetz formula, then
        # held FIXED for the whole run. This is deliberate: if R_fe were instead
        # recomputed every step to force-match the instantaneous Steinmetz P_iron, the
        # resulting i_Fe would just be P_iron/E_backemf by construction, and the
        # energy-balance residual would become tautologically ~0% -- true by algebra,
        # not because anything was actually verified. With a FIXED R_fe, the resulting
        # electrical draw (E_iron_electrical, see energy-balance section) generally will
        # NOT exactly equal the Steinmetz E_iron away from rated conditions -- the two
        # stay independent estimates of the same physical loss, so the residual is still
        # a meaningful check, not a guaranteed zero.
        #
        # IMPORTANT CAVEAT -- this is computed in POSTPROCESSING, after solve_ivp has
        # already finished solving for i_d(t), i_q(t) (see the postprocessing loop
        # below, where i_Fe_d_arr/i_Fe_q_arr are filled in). i_Fe never feeds back into
        # the ODE, the current-loop PI controller, di_d/dt, di_q/dt, or torque -- those
        # are all computed exactly as if iron loss did not exist. i_Fe only gets added
        # onto the terminal current afterward, purely to make E_elec_in (the energy-
        # balance report) more honest. This is a one-way/decoupled approximation, not a
        # full resimulation of a machine with iron loss -- worth being explicit about,
        # since it differs from how this plays out on real hardware:
        #
        # In a real drive, the phase current SENSOR physically measures the TOTAL
        # current in the winding -- there is no way to separate "the part going to
        # torque" from "the part going to core loss" with a single current sensor;
        # they are the same physical current. So the current-loop feedback automatically
        # sees and reacts to whatever extra current a real iron-loss branch adds, purely
        # as a side effect of ordinary closed-loop control -- the integrator drives
        # MEASURED current to the commanded reference regardless of what's contributing
        # to that measured current. No special iron-loss term needs to be designed into
        # the PI gains for this to work; the loop's normal disturbance-rejection handles
        # it, the same way it implicitly handles other things not present in the
        # standard Kp=L*omega_cc / Ki=R*omega_cc gain formulas either (temperature drift
        # in R_s, saturation-dependent L, manufacturing tolerance, etc.).
        #
        # That's exactly what's different here: i_Fe_d_arr/i_Fe_q_arr are invisible to
        # everything in this simulation, including K_pd_now/K_pq_now and the current-
        # loop error terms (err_d, err_q) below -- nothing "sees" i_Fe, so nothing
        # reacts to it, unlike a real current sensor which would.
        #
        # Where iron loss DOES get explicitly modeled in real control design is one
        # level up from the PI gains: efficiency-focused drives (common in EV traction)
        # use "loss-minimizing control" (a.k.a. maximum-efficiency control), where the
        # id_ref/iq_ref REFERENCE split is computed to minimize total loss at a given
        # torque and speed -- and that optimization does explicitly use a core-loss-
        # resistance model much like R_fe_iron here. But that changes what current is
        # COMMANDED, not the PI gains themselves -- the low-level tracking loop
        # underneath still just uses plain R_s/L-based tuning, exactly as this
        # simulation's current-loop PI does today.
        if self.include_iron_loss:
            _lambda_rated_for_Rfe = float(np.hypot(cfg.phi_f_ref, self.Lq0 * cfg.I_rated))
            _omega_s_rated = 2.0 * np.pi * cfg.f
            _E_rated = _omega_s_rated * _lambda_rated_for_Rfe
            _P_iron_rated = self.core_volume_m3 * (
                self.k_h_iron * (self.B_iron_rated ** self.beta_iron) * _omega_s_rated
                + self.k_e_iron * (self.B_iron_rated ** 2) * (_omega_s_rated ** 2)
            )
            self.R_fe_iron = 1.5 * _E_rated ** 2 / max(_P_iron_rated, 1e-9)
        else:
            self.R_fe_iron = float("inf")
        # NOTE:
        #  So in model:
        #  current → copper loss → stator heats up
        #  stator heats magnets
        #  magnet temperature changes 𝜙_f(T_m)

        self.R_s_ref = cfg.R_s_ref
        self.phi_f_ref = cfg.phi_f_ref

        # ------------------- thermal protection / fault thresholds -------------------
        # NOTE:
        #  These are example values only. In real work, use datasheet values.
        self.T_warn_stator = cfg.T_warn_stator
        self.T_fault_stator = cfg.T_fault_stator

        self.T_warn_magnet = cfg.T_warn_magnet
        self.T_fault_magnet = cfg.T_fault_magnet

        self.T_demag = cfg.T_demag
        self.phi_f_min = cfg.phi_f_min_factor * self.phi_f_ref  # do not let flux become negative/unphysical

        # mechanical / poles
        # TODO: to be option for user
        self.poles = cfg.poles
        self.polepairs = self.poles // 2
        # J = 0.01
        # B_visc = 0.0005
        # load_torque = 1.0

        # ==============================================================================================================================================
        # ------------------- VEHICLE / LOAD PARAMETERS -------------------
        self.g = 9.81
        self.rho_air = cfg.rho_air  # kg/m^3

        # Drivetrain
        self.gear_ratio = cfg.gear_ratio  # motor speed / wheel speed
        self.drivetrain_eff = cfg.drivetrain_eff
        self.wheel_radius = cfg.wheel_radius  # m

        # Vehicle / road
        self.Cd = cfg.Cd
        self.A_front = cfg.A_front
        self.b_vehicle = cfg.b_vehicle

        # Default values (can change by time using schedules below)
        # self.mass_default = cfg.mass_default
        # self.slope_default_deg = cfg.slope_default_deg
        # self.Crr_default = cfg.Crr_default

        # =============================================================================
        # INVERTER / PWM SETTINGS
        # =============================================================================
        self.use_svpwm = cfg.use_svpwm
        self.include_deadtime = cfg.include_deadtime

        self.f_pwm = cfg.f_pwm
        self.T_pwm = 1.0 / self.f_pwm

        self.deadtime = cfg.deadtime
        # TODO: input on from where to show the 20 PWM switches
        self.plot_pwm_zoom_start = cfg.plot_pwm_zoom_start
        self.plot_pwm_num_periods = cfg.plot_pwm_num_periods
        self.plot_pwm_zoom_span = self.plot_pwm_num_periods * self.T_pwm
        self.plot_pwm_zoom_end = self.plot_pwm_zoom_start + self.plot_pwm_zoom_span
        self.pwm_plot_samples = cfg.pwm_plot_samples

        # ------------------- STEP SCHEDULES -------------------
        # simulation time
        self.t0 = 0.0
        self.tfinal = cfg.tfinal
        # TODO: SET ALSO THE NUM MAYBE
        # self.t_eval = np.linspace(self.t0, self.tfinal, int(self.tfinal * 10000))
        n_eval = max(2, int(self.tfinal * 5000))
        self.t_eval = np.linspace(self.t0, self.tfinal, n_eval)

        # Speed reference schedule (mechanical rad/s)
        self.t_steps = cfg.t_steps
        self.omega_steps = cfg.omega_steps

        self.mass_step_times = cfg.mass_step_times
        self.mass_step_values = cfg.mass_step_values

        self.slope_step_times = cfg.slope_step_times
        self.slope_step_values_deg = cfg.slope_step_values_deg

        self.crr_step_times = cfg.crr_step_times
        self.crr_step_values = cfg.crr_step_values

        # NOTE:
        #  start on flat asphalt
        #  then add passengers/cargo
        #  then go uphill onto rough/sandy surface

        self.J_motor = cfg.J_motor
        # TODO: choose nominal or worst-case design mass
        # if J_total gets larger:
        # acceleration gets smaller
        # speed response becomes slower
        # same torque produces less speed change
        # So if your controller was designed for a smaller J, the real
        # system may respond more slowly than expected.
        # self.J_ctrl = self.J_motor + self.mass_default * self.wheel_radius ** 2 / self.gear_ratio ** 2

        # self._seg_mass = self.mass_default
        # self._seg_slope_deg = self.slope_default_deg
        # self._seg_crr = self.Crr_default
        # self._seg_omega_ref = 0.0
        # self._seg_J_total = self.J_motor + self.mass_default * self.wheel_radius ** 2 / (self.gear_ratio ** 2)
        # =============================================================================
        # CONTROL BANDWIDTH + GAINS
        # =============================================================================
        # Current loop bandwidth selection (rule-of-thumb)
        # page 71 - 2.6.1.1 Selection of the bandwidth for current control
        # IMPORTANT:
        #  If the current is sampled twice every switching period, as a rule of
        #  thumb, the maximum available bandwidth can be up to 1/10 of the
        #  switching frequency. On the other hand, if it is sampled once every
        #  switching period, the maximum available bandwidth can be up to 1/20 of
        #  switching frequency. For this case, it is desirable to restrict
        #  the bandwidth to 1/25 of the sampling frequency
        self.omega_cc = 2.0 * np.pi * (self.f_pwm / 25.0)

        # IMPORTANT:
        #  controllers:
        #  - current control bandwidth ωcc should be at least five times
        #    wider than the speed control bandwidth ωcs
        # Speed loop bandwidth
        # omega_cs = omega_cc / 5.0
        self.omega_cs = self.omega_cc / 5.0

        # PI current regulator (eq. 6.28/6.29):
        #   K_pd = L_d*ω_c,  K_pq = L_q*ω_c
        #   K_i  = R_s*ω_c
        self.K_pd = self.L_ds * self.omega_cc
        self.K_pq = self.L_qs * self.omega_cc
        self.K_i = self.R_s_ref * self.omega_cc

        # Anti-windup gain (note: K_a = 1/K_p)
        self.K_aw_d = self.K_i / max(self.K_pd, 1e-12)
        self.K_aw_q = self.K_i / max(self.K_pq, 1e-12)

        # IMPORTANT:
        #  page 73 - 2.6.2 ANTI-WINDUP CONTROLLER
        #  When we saturate the commanded voltage, the PI “thinks” it can apply
        #  more voltage than the inverter allows. The integrator would keep
        #  accumulating error (“wind up”) and then, when saturation clears, the
        #  integrator is huge → overshoot.
        #  u = Kp*e + x_int (Kp*e is proportional term, u is controller output
        #                    and x_int is the accumulated integral action)
        #                    It stores past error and provides steady-state accuracy
        #  xint_dot = K_i*e + K_aw(u_sat−u)
        #   >> the delta(u) = measures how much the controller is asking for more
        #                     than the actuator can deliver
        #                   = delta is negative
        #   >> any excess output during saturation is strongly influenced by 𝐾_𝑝.
        #      To “undo” that excess at a comparable rate: The integrator
        #      correction should scale inversely with 𝐾_𝑝.

        # Speed PI from “DC motor-like” procedure:
        # For PMSM a common torque constant near id≈0 is:
        # K_T ≈ 1.5*polepairs*phi_f
        # self.K_T = 1.5 * self.polepairs * self.phi_f_ref
        # self.K_ps = self.J_ctrl * self.omega_cs / max(self.K_T, 1e-12)
        # self.K_is = self.K_ps * self.omega_cs / 5.0
        # # K_aw_s = 1.0 / max(K_ps, 1e-12)
        # self.K_aw_s = self.K_is / max(self.K_ps, 1e-12)
        # TODO: maybe implement combination of PI and IP control (pages 78-81)

        # IMPORTANT:
        #  Current limits for speed loop output (iq_ref clamp)
        #  Given on the motor datasheet as rated current or continuous current
        #    - What sets it physically
        #    - Copper loss I^2*R
        #    - Thermal path from windings → stator → housing → air
        #  If you exceed this current continuously, the motor will overheat
        #  Inverter current limit → hard electrical ceiling
        #  - Maximum current the inverter can physically deliver
        #  Limited by:
        #    - Semiconductor peak current rating
        #    - DC bus voltage
        #    - Gate driver protection
        #    - Current sensor range
        #  examples:
        #    - Small servo (400 W) >> 2.5 A
        #    - Industrial servo (3 kW) >> 7 A
        #    - Large PMSM (20 kW) >> 35 A
        #  there can be 2 limits > long term and short term
        self.Imax = cfg.Imax

        #########################################################################################################################################################
        # MTPA LUT - beg
        #########################################################################################################################################################
        self.mtpa_model_params = {
            "Ld_inf": self.Ld_inf,  # measured / FEM / identified at high current
            "Lq_inf": self.Lq_inf,
            "I_Ld_sat": self.I_Ld_sat,
            "I_Lq_sat": self.I_Lq_sat,
        }

        self.mtpa_lut = self.build_mtpa_object(
            machine_type=self.machine_type,
            polepairs=self.polepairs,
            Imax=self.Imax,
            L_ds=self.L_ds,
            L_qs=self.L_qs,
            mtpa_model_params=self.mtpa_model_params,
            phi_f_initial=self.phi_f_ref,
        )
        #########################################################################################################################################################
        # MTPA LUT - end
        #########################################################################################################################################################

        #########################################################################################################################################################
        # FIELD-WEAKENING "REGIME B" LUT - beg
        # (voltage-limited-only FW: hits torque_cmd exactly, using LESS
        #  than Imax of current. See pmsm_backend/fw_lut.py -- FwLut's
        #  class docstring -- for the full derivation, a worked numeric
        #  example, and the table structure. Only meaningful for IPMSM
        #  with an already-built mtpa_lut; None for SPMSM (which never
        #  field-weakens at all -- see the comment above
        #  current_references_with_fw()).
        #########################################################################################################################################################
        self.fw_lut = self.build_fw_object(
            machine_type=self.machine_type,
            polepairs=self.polepairs,
            Imax=self.Imax,
            Vmax=self.Vmax,
            mtpa_lut=self.mtpa_lut,
            phi_f_initial=self.phi_f_ref,
        )
        #########################################################################################################################################################
        # FIELD-WEAKENING "REGIME B" LUT - end
        #########################################################################################################################################################

        self.FW_EPS = 1e-12
        # It’s a tiny number used as a safety threshold to avoid:
        # division by zero
        # square root of negative numbers (due to floating-point noise)
        # unstable behavior at very low speeds
        self.MAX_SPEED_REACHED_TOL = 0.995

        self.rtol = cfg.rtol
        self.atol = cfg.atol
        self.solver_method = cfg.solver_method

        # Storage used by thermal event functions
        # self._fault_events_info: List[str] = []

    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================
    # def set_segment_constants(self, t_segment_start: float) -> None:
    #     """
    #     Freeze all schedule-driven values for the current solver segment.
    #     Since solve_piecewise splits exactly at discontinuities, these values
    #     remain constant inside the segment.
    #     """
    #     self._seg_mass = self.piecewise_step(
    #         t_segment_start, self.mass_step_times, self.mass_step_values
    #     )
    #     self._seg_slope_deg = self.piecewise_step(
    #         t_segment_start, self.slope_step_times, self.slope_step_values_deg
    #     )
    #     self._seg_crr = self.piecewise_step(
    #         t_segment_start, self.crr_step_times, self.crr_step_values
    #     )
    #     self._seg_omega_ref = self.omega_ref(t_segment_start)

    #     self._seg_J_total = (
    #         self.J_motor
    #         + self._seg_mass * self.wheel_radius ** 2 / (self.gear_ratio ** 2)
    #     )
    
    # def total_resisting_torque_segment(self, omega_m: float) -> float:
    #     """
    #     Resisting torque referred to the motor shaft for the current segment.
    #     Uses frozen schedule values, so no schedule lookup is done inside the ODE.
    #     """
    #     slope_rad = np.deg2rad(self._seg_slope_deg)
    #     v = (omega_m / self.gear_ratio) * self.wheel_radius

    #     s_v = np.tanh(v / 0.05)

    #     F_grade = self._seg_mass * self.g * np.sin(slope_rad)
    #     F_roll = 
    #         self._seg_crr * self._seg_mass * self.g * np.cos(slope_rad) * np.sign(s_v)
    #         if abs(s_v) > 1e-9 else 0.0
    #     )
    #     F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v * abs(v)
    #     F_visc = self.b_vehicle * v

    #     F_total = F_grade + F_roll + F_aero + F_visc
    #     T_wheel = F_total * self.wheel_radius
    #     T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)

    #     return T_motor_load

    # def get_schedule_breakpoints(self) -> np.ndarray:
    #     """
    #     Return sorted unique time breakpoints where any schedule changes.
    #     Includes t0 and tfinal.
    #     """
    #     all_breaks = (
    #         [self.t0, self.tfinal]
    #         + list(np.asarray(self.t_steps, dtype=float))
    #         + list(np.asarray(self.mass_step_times, dtype=float))
    #         + list(np.asarray(self.slope_step_times, dtype=float))
    #         + list(np.asarray(self.crr_step_times, dtype=float))
    #     )

    #     # keep only times inside [t0, tfinal]
    #     all_breaks = [tb for tb in all_breaks if self.t0 <= tb <= self.tfinal]

    #     # sorted unique
    #     return np.array(sorted(set(all_breaks)), dtype=float)
    
    # def solve_piecewise(self, x0: np.ndarray):
    #     """
    #     Solve the ODE piecewise across all schedule discontinuities.
    #     Returns:
    #         t_all, y_all, segment_solutions, stop_info
    #     """
    #     breaks = self.get_schedule_breakpoints()

    #     t_parts = []
    #     y_parts = []
    #     segment_solutions = []

    #     x_init = x0.copy()

    #     stop_info = {
    #         "success": True,
    #         "status": 0,
    #         "message": "Completed successfully.",
    #         "failed_segment": None,
    #         "failed_time": None,
    #         "t_events": [[], []],
    #     }

    #     for k in range(len(breaks) - 1):
    #         t_start = float(breaks[k])
    #         t_end = float(breaks[k + 1])

    #         if t_end <= t_start:
    #             continue
            
    #         # self.set_segment_constants(t_start)

    #         # restrict requested output points to this segment
    #         seg_mask = (self.t_eval >= t_start) & (self.t_eval <= t_end)
    #         t_eval_seg = self.t_eval[seg_mask]

    #         # make sure segment end points are included
    #         if t_eval_seg.size == 0 or abs(t_eval_seg[0] - t_start) > 1e-15:
    #             t_eval_seg = np.insert(t_eval_seg, 0, t_start)
    #         if abs(t_eval_seg[-1] - t_end) > 1e-15:
    #             t_eval_seg = np.append(t_eval_seg, t_end)

    #         def stator_fault_event(t, x):
    #             return self.T_fault_stator - x[7]
    #         stator_fault_event.terminal = True
    #         stator_fault_event.direction = -1

    #         def magnet_fault_event(t, x):
    #             return self.T_fault_magnet - x[8]
    #         magnet_fault_event.terminal = True
    #         magnet_fault_event.direction = -1

    #         sol_seg = solve_ivp(
    #             self.motor_ode_pmsm_controlled,
    #             (t_start, t_end),
    #             x_init,
    #             method=self.solver_method,
    #             t_eval=t_eval_seg,
    #             rtol=self.rtol,
    #             atol=self.atol,
    #             events=[stator_fault_event, magnet_fault_event],
    #         )

    #         segment_solutions.append(sol_seg)

    #         # collect event times
    #         for i_ev in range(2):
    #             if len(sol_seg.t_events[i_ev]) > 0:
    #                 stop_info["t_events"][i_ev].extend(sol_seg.t_events[i_ev].tolist())

    #         # append segment data, avoiding duplicate boundary sample
    #         if len(t_parts) == 0:
    #             t_parts.append(sol_seg.t)
    #             y_parts.append(sol_seg.y)
    #         else:
    #             # skip first point because it is the same as previous segment end
    #             t_parts.append(sol_seg.t[1:])
    #             y_parts.append(sol_seg.y[:, 1:])

    #         if sol_seg.status == 1:
    #             stop_info["success"] = False
    #             stop_info["status"] = 1
    #             stop_info["message"] = "Terminated by event."
    #             stop_info["failed_segment"] = (t_start, t_end)
    #             stop_info["failed_time"] = float(sol_seg.t[-1]) if sol_seg.t.size else t_start
    #             break

    #         if not sol_seg.success:
    #             stop_info["success"] = False
    #             stop_info["status"] = sol_seg.status
    #             stop_info["message"] = sol_seg.message
    #             stop_info["failed_segment"] = (t_start, t_end)
    #             stop_info["failed_time"] = float(sol_seg.t[-1]) if sol_seg.t.size else t_start
    #             break

    #         # use final state of this segment as initial state of next one
    #         x_init = sol_seg.y[:, -1].copy()

    #     if len(t_parts) == 0:
    #         raise RuntimeError("Piecewise solver produced no output.")

    #     t_all = np.concatenate(t_parts)
    #     y_all = np.concatenate(y_parts, axis=1)

    #     return t_all, y_all, segment_solutions, stop_info
#--------------------------------------------------------------------------------
    def first_order_tracking(self, x: float, x_target: float, tau: float) -> float:
        """
        Compute the first-order tracking rate that moves x toward x_target.

        A smaller tau makes the response faster, while a larger tau makes it slower.
        A tiny lower bound is applied to tau to avoid division by zero.

        Args:
            x: Current value.
            x_target: Desired target value.
            tau: Time constant controlling tracking speed.

        Returns:
            The rate of change needed for x to approach x_target.
        """
        tau_eff = max(tau, 1e-9)
        return (x_target - x) / tau_eff
    
    def rate_limited_tracking(
        self,
        x: float,
        x_target: float,
        tau: float,
        rate_up: float,
        rate_down: float
    ) -> float:
        tau_eff = max(tau, 1e-9)
        dx = (x_target - x) / tau_eff
        return np.clip(dx, -abs(rate_down), abs(rate_up))
    
    # def omega_ref_target(self, t: float) -> float:
    #     return self.piecewise_step(t, self.t_steps, self.omega_steps)

    def mass_target(self, t: float) -> float:
        return self.piecewise_step(t, self.mass_step_times, self.mass_step_values)

    def slope_target(self, t: float) -> float:
        return self.piecewise_step(t, self.slope_step_times, self.slope_step_values_deg)

    def crr_target(self, t: float) -> float:
        return self.piecewise_step(t, self.crr_step_times, self.crr_step_values)
#--------------------------------------------------------------------------------
    def smooth_sign(self, v: float, v_eps: float = 0.05) -> float:
        """
        Smooth approximation of sign(v).
        v_eps sets the transition width around zero speed.
        """
        return np.tanh(v / v_eps)

    def piecewise_step(self, t: float, times: np.ndarray, values: np.ndarray) -> float:
        if len(times) != len(values):
            raise ValueError("times and values must have same length")
        idx = np.searchsorted(times, t, side="right") - 1
        if idx < 0:
            return float(values[0])
        return float(values[idx])

    def vehicle_speed_from_motor_speed(self, omega_m: float) -> float:
        '''
        Force (N)
           ↓ × wheel radius
        Wheel torque (Nm)
           ↓ ÷ gear ratio
        Motor torque (Nm)
        | Quantity | Symbol | Units      | Where it acts     |
        | -------- | ------ | ---------- | ----------------- |
        | Force    | (F)    | Newton (N) | linear motion     |
        | Torque   | (T)    | Nm         | rotational motion |
        '''
        omega_wheel = omega_m / self.gear_ratio
        v = omega_wheel * self.wheel_radius
        return v

    # def total_resisting_torque(self, t: float, omega_m: float) -> float:
    #     """
    #     Return resisting torque referred to the MOTOR shaft [Nm].
    #     Inputs are intuitive vehicle parameters (kg, slope, road coefficient).
    #     """
    #     # Scheduled inputs
    #     mass = self.piecewise_step(t, self.mass_step_times, self.mass_step_values)  # kg
    #     slope_deg = self.piecewise_step(t, self.slope_step_times, self.slope_step_values_deg)  # deg
    #     Crr = self.piecewise_step(t, self.crr_step_times, self.crr_step_values)  # -

    #     slope_rad = np.deg2rad(slope_deg)
    #     v = self.vehicle_speed_from_motor_speed(omega_m)  # m/s

    #     s_v = self.smooth_sign(v, v_eps=0.05)

    #     '''
    #     Aerodynamic drag should always oppose velocity
    #     Viscous loss should always oppose velocity
    #     Grade force depends on road slope, not on velocity direction
    #     '''

    #     # Resistive forces at vehicle
    #     F_grade = mass * self.g * np.sin(slope_rad)

    #     # Rolling resistance usually opposes motion
    #     # F_roll = Crr * mass * g * np.cos(slope_rad)
    #     # F_roll = Crr * mass * self.g * np.cos(slope_rad) * np.sign(s_v) if abs(s_v) > 1e-9 else 0.0
    #     F_roll = Crr * mass * self.g * np.cos(slope_rad) * s_v

    #     # Aero drag opposes motion and should not help reverse motion
    #     F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v * abs(v)

    #     # Lumped viscous term
    #     F_visc = self.b_vehicle * v

    #     # Total road/load force
    #     F_total = F_grade + F_roll + F_aero + F_visc

    #     # Convert force -> wheel torque
    #     T_wheel = F_total * self.wheel_radius

    #     # Convert wheel torque -> motor shaft torque
    #     # TODO: input
    #     T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)

    #     return T_motor_load

    # eq. 1.69
    # def equivalent_vehicle_inertia(self, t: float) -> float:
    #     mass = self.piecewise_step(t, self.mass_step_times, self.mass_step_values)
    #     return mass * self.wheel_radius ** 2 / (self.gear_ratio ** 2)

    # def total_inertia(self, t: float) -> float:
    #     return self.J_motor + self.equivalent_vehicle_inertia(t)

    # source: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://eprints.whiterose.ac.uk/id/eprint/164680/1/System%20Identification%20AG_SX_finalV.pdf
    # page 3 eq. (2)
    def R_s_of_T(self, Ts: float) -> float:
        '''
        :param Ts: temperature of the stator
        :return:
        '''
        return self.R_s_ref * (1.0 + self.alpha_cu * (Ts - self.T_ref))

    # source?
    def phi_f_of_T(self, Tm: np.ndarray | float) -> np.ndarray | float:
        '''
        :param Tm: PM temperature
        :return:
        '''
        phi = self.phi_f_ref * (1.0 - self.alpha_phi * (Tm - self.T_ref))
        return np.maximum(phi, self.phi_f_min)

    # =============================================================================
    # REFERENCES (speed + currents)
    # =============================================================================
    def omega_ref(self, t: float) -> float:
        """
        Piecewise-constant (step) speed reference.

        t_steps[k]      : time at which step k becomes active
        omega_steps[k]  : speed reference after that time

        Arrays must have the same length.
        """
        if len(self.t_steps) != len(self.omega_steps):
            raise ValueError(
                f"omega_ref error: t_steps (len={len(self.t_steps)}) "
                f"and omega_steps (len={len(self.omega_steps)}) must have the same length."
            )

        idx = np.searchsorted(self.t_steps, t, side="right") - 1

        if idx < 0:
            return float(self.omega_steps[0])

        return float(self.omega_steps[idx])

    #########################################################################################################################################################
    # MTPA LUT - beg
    #########################################################################################################################################################

    def build_mtpa_object(
        self,
        machine_type: str,
        polepairs: int,
        Imax: float,
        L_ds: float,
        L_qs: float,
        mtpa_model_params: Dict[str, float],
        phi_f_initial: float,
    ) -> Optional[MtpaLut]:
        if machine_type == "spmsm":
            return None

        key = self.get_mtpa_cache_key()

        if key in _MTPA_CACHE:
            return _MTPA_CACHE[key]

        # mtpa = MtpaLut(
        #     polepairs=polepairs,
        #     i_max=Imax,
        #     n_points=121,
        #     id_samples=200,
        # )
        mtpa = MtpaLut(
            polepairs=polepairs,
            i_max=Imax,

            # None means automatic sizing
            n_points=None,
            id_samples=None,

            # Tuning:
            # 4 points/Nm  -> about 0.25 Nm torque spacing
            # 10 points/A  -> about 0.1 A id spacing
            torque_points_per_nm=4.0,
            id_points_per_amp=10.0,

            # Safety limits
            min_torque_points=121,
            max_torque_points=1201,
            min_id_samples=200,
            max_id_samples=2000,
        )
        
        mtpa.configure_inductance_model(
            Ld0=L_ds,
            Lq0=L_qs,
            Ld_inf=mtpa_model_params["Ld_inf"],
            Lq_inf=mtpa_model_params["Lq_inf"],
            I_Ld_sat=mtpa_model_params["I_Ld_sat"],
            I_Lq_sat=mtpa_model_params["I_Lq_sat"],
        )
        mtpa.build(phi_f_initial)

        _MTPA_CACHE[key] = mtpa
        return mtpa

    #########################################################################################################################################################
    # FIELD-WEAKENING "REGIME B" LUT - beg
    #########################################################################################################################################################

    def build_fw_object(
        self,
        machine_type: str,
        polepairs: int,
        Imax: float,
        Vmax: float,
        mtpa_lut: Optional[MtpaLut],
        phi_f_initial: float,
    ) -> Optional[FwLut]:
        """
        Build (or fetch from cache) the Regime-B field-weakening LUT.
        Mirrors build_mtpa_object() above -- same caching strategy, same
        "None for SPMSM" guard (SPMSM never field-weakens at all, see
        the big comment above current_references_with_fw() explaining
        why), same "build once, reuse for the lifetime of the process"
        intent.

        Requires mtpa_lut to already be built (FwLut leans on it for
        Ld(id)/Lq(iq) and the MTPA boundary point -- see fw_lut.py's
        class docstring for why), so this must be called AFTER
        self.mtpa_lut is assigned in __init__, never before.
        """
        if machine_type == "spmsm" or mtpa_lut is None:
            return None

        key = self.get_fw_cache_key()

        if key in _FW_CACHE:
            return _FW_CACHE[key]

        fw = FwLut(
            polepairs=polepairs,
            i_max=Imax,
            v_max=Vmax,
            mtpa_lut=mtpa_lut,

            # Kept modest on purpose -- see FwLut's __init__ docstring
            # for the cost trade-off (this table is far more expensive
            # per grid point to build than the MTPA one).
            n_torque_points=31,
            n_speed_points=31,
            id_bracket_scan_points=60,
        )
        fw.build(phi_f_initial)

        _FW_CACHE[key] = fw
        return fw

    #########################################################################################################################################################
    # FIELD-WEAKENING "REGIME B" LUT - end
    #########################################################################################################################################################

    # def pmsm_torque_constant(self, phi_f: float) -> float:
    #     # SPMSM / id≈0 torque constant used only to convert the speed-loop
    #     # output to a torque request before the MTPA lookup.
    #     # Equation used:
    #     #   Te ≈ 1.5*p*phi_f*iq
    #     return 1.5 * self.polepairs * phi_f

    def clamp_current_circle(self, i_d_cmd: float, i_q_cmd: float, Imax: float) -> Tuple[float, float]:
        """
        Enforce current magnitude <= Imax.
        """
        mag = np.hypot(i_d_cmd, i_q_cmd)
        # Computes the magnitude (length) of the current vector:
        # |I_s| = sqrt(i_d^2 + i_q^2)
        # This is the Euclidean norm of the dq current vector.
        if mag <= Imax or mag < self.FW_EPS:
            return i_d_cmd, i_q_cmd
        scale = Imax / mag
        return scale * i_d_cmd, scale * i_q_cmd

    def pmsm_torque_to_current_refs(self, machine_type: str, torque_ref: float, phi_f: float, mtpa_lut: Optional[MtpaLut]) -> Tuple[float, float]:
        """
        Current references before field weakening.

        SPMSM:
          Eq. (5.69), (5.70)
            id*=0
            iq*=Te/(1.5*p*phi_f)

        IPMSM:
          use the prebuilt MTPA LUT.
        """
        if machine_type == "spmsm" or not self.use_mtpa:
            id_ref = 0.0
            iq_ref = torque_ref / max(1.5 * self.polepairs * phi_f, 1e-12)
            return self.clamp_current_circle(id_ref, iq_ref, self.Imax)

        if mtpa_lut is None:
            raise ValueError("IPMSM requires mtpa_lut.")
        
        id_ref, iq_ref = mtpa_lut.lookup(torque_ref)
        return self.clamp_current_circle(id_ref, iq_ref, self.Imax)
    #########################################################################################################################################################
    # MTPA LUT - end
    #########################################################################################################################################################

    # FIELD - WEAKENING BEGIN -------------------------------------------------------------------------------------------------------
    # def ipmsm_base_speed(self, phi_f: float, i_d_mtpa: float, i_q_mtpa: float) -> float:
    #     """
    #     Eq. (8.31):
    #         omega_r_base = V_s_max / sqrt((L_ds*i_ds + phi_f)^2 + (L_qs*i_qs)^2)
    #     """
    #     denom = np.sqrt((self.L_ds * i_d_mtpa + phi_f) ** 2 + (self.L_qs * i_q_mtpa) ** 2)
    #     if denom < self.FW_EPS:
    #         return np.inf
    #     return self.Vmax / denom

    def ipmsm_max_speed_finite(self, phi_f: float) -> Optional[float]:
        """
        Eq. (8.35), finite-speed case only:
            omega_r_max = V_s_max / (phi_f - L_ds*I_s_max)
        valid only if phi_f > L_ds*Imax
        """
        margin = phi_f - self.L_ds * self.Imax
        if margin <= self.FW_EPS:
            return None  # not finite-speed case
        return self.Vmax / margin

    def ipmsm_fw_currents_finite(self, omega_r_e: float, phi_f: float) -> Optional[Tuple[float, float]]:
        """
        Eqs. (8.36), (8.37) for IPMSM finite-speed FW.
        """
        # TODO: ASK - do we actually use I_s or I_max??
        denom = (self.L_qs ** 2 - self.L_ds ** 2)
        if abs(denom) < self.FW_EPS:
            return None

        if abs(omega_r_e) < self.FW_EPS:
            return 0.0, 0.0
        # <0 ==> reverse rotation

        rad = ((self.L_ds * phi_f) ** 2
               + denom * (phi_f ** 2 + (self.L_qs * self.Imax) ** 2 - (self.Vmax / abs(omega_r_e)) ** 2))

        rad = max(rad, 0.0)

        i_d_fw = (self.L_ds * phi_f - np.sqrt(rad)) / denom
        i_d_fw = np.clip(i_d_fw, -self.Imax, 0.0)

        i_q_fw_mag = np.sqrt(max(self.Imax ** 2 - i_d_fw ** 2, 0.0))
        i_q_fw = np.sign(omega_r_e if abs(omega_r_e) > self.FW_EPS else 1.0) * i_q_fw_mag
        # always produce torque in the same direction as rotation
        # TODO: Ask if we want to take regenerative braking into consideration
        return i_d_fw, i_q_fw

    def spmsm_base_speed(self, phi_f: float) -> float:
        """
        Base-speed condition for SPMSM at point M (i_d*=0, i_q*=I_smax),
        from the voltage limit with Eqs. (8.42), (8.43):

            V_s_max^2 = omega_r^2 * [ (L_s I_s_max)^2 + phi_f^2 ]

        therefore
            omega_r_base = V_s_max / sqrt(phi_f^2 + (L_s I_s_max)^2)
        """
        denom = np.sqrt(phi_f ** 2 + (self.L_ds * self.Imax) ** 2)
        if denom < self.FW_EPS:
            return np.inf
        return self.Vmax / denom

    def spmsm_max_speed_finite(self, phi_f: float) -> Optional[float]:
        """
        Finite-speed case for SPMSM:
            omega_r_max = V_s_max / (phi_f - L_s*I_s_max)
        valid only if phi_f > L_s*I_s_max
        """
        margin = phi_f - self.L_ds * self.Imax
        if margin <= self.FW_EPS:
            return None
        return self.Vmax / margin
    # TODO: combine with ipmsm >> it is the same

    def spmsm_fw_currents_finite(self, omega_r_e: float, phi_f: float) -> Tuple[float, float]:
        """
        Eqs. (8.46), (8.47) for SPMSM finite-speed FW.
        """
        if abs(omega_r_e) < self.FW_EPS:
            return 0.0, 0.0

        denom = 2.0 * self.L_ds * phi_f
        if abs(denom) < self.FW_EPS:
            return 0.0, 0.0

        i_d_fw = (((self.Vmax / abs(omega_r_e)) ** 2 - phi_f ** 2 - (self.L_ds * self.Imax) ** 2) / denom)
        i_d_fw = np.clip(i_d_fw, -self.Imax, 0.0)

        i_q_fw_mag = np.sqrt(max(self.Imax ** 2 - i_d_fw ** 2, 0.0))
        i_q_fw = np.sign(omega_r_e if abs(omega_r_e) > self.FW_EPS else 1.0) * i_q_fw_mag

        return i_d_fw, i_q_fw

    def motor_inductances(self, machine_type: str, i_d: float, i_q: float, mtpa_lut: Optional[MtpaLut]) -> Tuple[float, float]:
        """
        Current-dependent inductances used by controller and ODE.

        SPMSM:
            Ld = Lq = constant cylindrical-rotor value

        IPMSM:
            use the same saturation-law model as in the MTPA LUT
        """
        if machine_type == "spmsm" or mtpa_lut is None:
            return self.L_ds, self.L_qs

        return mtpa_lut.ld(i_d), mtpa_lut.lq(i_q)


    def current_references_with_fw(self, machine_type, omega_m, torque_cmd, phi_f, mtpa_lut):
        """
        Returns:
            id_ref, iq_ref, omega_base, omega_max, max_speed_reached_now

        Below base speed:
          - SPMSM: id*=0, iq*=Te/(1.5*p*phi_f)
          - IPMSM: MTPA LUT

        Above base speed:
          - use existing finite-speed field-weakening equations
        """
        omega_r_e = abs(self.polepairs * omega_m)

        # ---------- base current references ----------
        if machine_type == "spmsm":
            id_base = 0.0
            iq_base = torque_cmd / max(1.5 * self.polepairs * phi_f, 1e-12)
            id_base, iq_base = self.clamp_current_circle(id_base, iq_base, self.Imax)

            omega_base = self.spmsm_base_speed(phi_f)
            omega_max = self.spmsm_max_speed_finite(phi_f)
            if omega_max is None:
                omega_max = np.inf

        elif machine_type == "ipmsm":
            if self.use_mtpa and mtpa_lut is not None:
                id_base, iq_base = mtpa_lut.lookup(torque_cmd)
            else:
                id_base = 0.0
                iq_base = torque_cmd / max(1.5 * self.polepairs * phi_f, 1e-12)

            id_base, iq_base = self.clamp_current_circle(id_base, iq_base, self.Imax)

            """
            Eq. (8.31):
                omega_r_base = V_s_max / sqrt((L_ds*i_ds + phi_f)^2 + (L_qs*i_qs)^2)
            """
            Ld_base, Lq_base = self.motor_inductances(machine_type, id_base, iq_base, mtpa_lut)
            denom = np.sqrt((Ld_base * id_base + phi_f) ** 2 + (Lq_base * iq_base) ** 2)
            omega_base = np.inf if denom < self.FW_EPS else self.Vmax / denom

            omega_max = self.ipmsm_max_speed_finite(phi_f)
            if omega_max is None:
                omega_max = np.inf

        else:
            raise ValueError(f"Unknown machine_type: {machine_type}")

        # ---------- no field weakening ----------
        if (not self.use_field_weakening) or machine_type == "spmsm":
            return id_base, iq_base, omega_base, omega_max, False

        # ---------- field weakening enabled ----------
        if omega_r_e >= omega_max:
            return -self.Imax, 0.0, omega_base, omega_max, True

        if omega_r_e <= omega_base:
            return id_base, iq_base, omega_base, omega_max, False

        # ---- Regime B: voltage-limited-only FW -----------------------------
        # Try to hit torque_cmd EXACTLY using less than the full Imax, before
        # falling back to the Imax-pinned Regime-C formula below. This is the
        # branch that was previously missing entirely -- see fw_lut.py's
        # FwLut class docstring for the full derivation (torque equation +
        # voltage-ellipse equation, solved jointly for i_d, no Imax anywhere
        # in it) and a worked numeric example for this project's IPMSM
        # parameters.
        #
        # self.fw_lut is None only if it hasn't been built (shouldn't happen
        # for IPMSM once __init__ has run -- see build_fw_object()); guarded
        # here anyway so a missing table degrades to the old Regime-C-only
        # behavior instead of crashing.
        if machine_type == "ipmsm" and self.fw_lut is not None:
            fw_b = self.fw_lut.lookup(abs(torque_cmd), omega_r_e)
            if fw_b is not None:
                i_d_fw, i_q_fw_mag = fw_b
                i_q_fw = np.sign(torque_cmd) * abs(i_q_fw_mag)  # reaches correct sign
                i_d_fw, i_q_fw = self.clamp_current_circle(i_d_fw, i_q_fw, self.Imax)  # safety clamp
                return i_d_fw, i_q_fw, omega_base, omega_max, False
            # fw_b is None -> Regime B has no feasible point here (even the
            # full current isn't enough to hit torque_cmd at this speed).
            # Fall through to Regime C below, exactly as before this change.

        # ---- Regime C: current-AND-voltage-limited (unchanged) -------------
        # Correct as-is once BOTH constraints are genuinely active -- see the
        # "TODO: ASK - do we actually use I_s or I_max??" comment inside
        # ipmsm_fw_currents_finite() below: the answer is that Imax is
        # exactly right THERE, because this branch is now only reached once
        # Regime B has already ruled out any lower-current solution.
        if machine_type == "spmsm":
            i_d_fw, i_q_fw_mag = self.spmsm_fw_currents_finite(omega_r_e, phi_f)
        else:
            fw = self.ipmsm_fw_currents_finite(omega_r_e, phi_f)
            if fw is None:
                return id_base, iq_base, omega_base, omega_max, False
            i_d_fw, i_q_fw_mag = fw

        i_q_fw = np.sign(torque_cmd) * abs(i_q_fw_mag)
        i_d_fw, i_q_fw = self.clamp_current_circle(i_d_fw, i_q_fw, self.Imax)
        return i_d_fw, i_q_fw, omega_base, omega_max, False

    # def torque_from_iq_command(self, machine_type: str, iq_cmd: float, phi_f: float) -> float:
    #     """
    #     Converts the existing speed-loop q-axis current command into a torque command.

    #     Equation used:
    #         Te* ≈ 1.5 * p * phi_f * iq*
    #     This preserves the original speed-PI tuning.
    #     """
    #     return 1.5 * self.polepairs * phi_f * iq_cmd

    def torque_from_iq_command(self, machine_type: str, iq_cmd: float, phi_f: float) -> float:
        """
        Convert the speed-loop q-axis current command into a torque command.

        SPMSM:
            Te* = 1.5 * p * phi_f * iq*

        IPMSM:
            First form a provisional torque using the same expression above,
            then refine it using the MTPA LUT and the full saliency torque law:
                Te = 1.5 * p * (phi_f * iq + (Ld - Lq) * id * iq)

        This keeps the original speed-PI tuning nearly unchanged while giving a
        better torque command for IPMSM.
        """
        # Base conversion that is exact for SPMSM and still a good first guess for IPMSM
        torque_cmd = 1.5 * self.polepairs * phi_f * iq_cmd

        # SPMSM: no reluctance torque term
        if machine_type == "spmsm":
            return torque_cmd

        # IPMSM without MTPA/LUT: nothing better can be inferred from iq_cmd alone
        if not self.use_mtpa or self.mtpa_lut is None:
            '''
            if MTPA is disabled, or
            if the IPMSM MTPA lookup table does not exist,
            '''
            return torque_cmd

        # One-step refinement using the actual MTPA operating point
        id_est, iq_est = self.mtpa_lut.lookup(torque_cmd)
        Ld_est, Lq_est = self.motor_inductances(machine_type, id_est, iq_est, self.mtpa_lut)

        return 1.5 * self.polepairs * (
            phi_f * iq_est + (Ld_est - Lq_est) * id_est * iq_est
        )

    # IMPORTANT:
    #  Why do we not weaken the field in SPMSM:
    #  Because a surface-mounted PMSM (SPMSM) has very low inductance, so
    #  it can’t generate enough opposing magnetic field to safely or
    #  efficiently weaken the permanent-magnet flux at high speed.
    #  - SPMSMs have magnets on the surface → air-gap is large → inductance is small.
    #  - Small inductance = d-axis current produces very little flux reduction.
    #  - Demagnetization risk
    #    - Strong negative d-axis current can partially demagnetize the
    #      surface magnets, permanently reducing torque.
    #  Why IPMSMs are more resistant:
    #  - Magnets are buried inside the rotor
    #    - They are shielded by rotor iron, which reduces the demagnetizing
    #      field seen by the magnets.
    #  - Higher d-axis inductance (Ld)
    #    - You need much less neg. d-axis current to weaken the air-gap flux.
    #    - That means lower demagnetizing stress on the magnets.
    #  - Saliency helps
    #    - Torque can come from reluctance torque, not just magnet torque,
    #      so extreme negative d-axis current is unnecessary.

    def dq_cmd_to_inverter_output(self, v_d_cmd: float, v_q_cmd: float, theta_frame: float, i_d: float, i_q: float, Vdc: float, T_pwm: float, deadtime: float, use_deadtime: bool = True) -> Dict[str, np.ndarray | float]:
        """
        Full inverter chain:
            dq cmd -> abc phase refs
                   -> SVPWM offset-voltage pole refs
                   -> duty ratios
                   -> dead-time correction
                   -> average pole voltages
                   -> average phase voltages
                   -> back to dq applied voltage

        Returns a dictionary so you can inspect everything later if needed.
        """
        # dq -> abc phase refs (line-neutral references)
        v_abc_ref = dq0_to_abc(np.array([v_d_cmd, v_q_cmd, 0.0]), theta_frame)

        # offset-voltage SVPWM
        v_pole_ref, v_offset = phase_refs_to_svpwm_pole_refs(v_abc_ref)

        # duty commands
        d_cmd = pole_refs_to_duties(v_pole_ref, Vdc)

        # phase currents for dead-time sign
        i_abc = dq0_to_abc(np.array([i_d, i_q, 0.0]), theta_frame)

        if use_deadtime:
            d_eff = apply_deadtime_to_duties(d_cmd, i_abc, deadtime, T_pwm)
        else:
            d_eff = d_cmd.copy()

        # actual average voltages after dead-time
        v_pole_avg = duties_to_average_pole_voltages(d_eff, Vdc)
        v_abc_avg = pole_to_phase_voltages(v_pole_avg)

        # back to dq applied voltage
        v_dq0_applied = T_matrix(theta_frame) @ v_abc_avg
        v_d_applied, v_q_applied, _ = v_dq0_applied

        return {
            "v_abc_ref": v_abc_ref,
            "v_offset": v_offset,
            "v_pole_ref": v_pole_ref,
            "d_cmd": d_cmd,
            "i_abc": i_abc,
            "d_eff": d_eff,
            "v_pole_avg": v_pole_avg,
            "v_abc_avg": v_abc_avg,
            "v_d_applied": v_d_applied,
            "v_q_applied": v_q_applied,
        }

    # =============================================================================
    # EVENT FUNCTIONS
    # =============================================================================
    # event functions when temperature fault happens = demagnetisation = stop solver
    # def stator_fault_event(self, t: float, x: np.ndarray) -> float:
    #     T_s = x[7]
    #     return self.T_fault_stator - T_s

    # def magnet_fault_event(self, t: float, x: np.ndarray) -> float:
    #     T_m = x[8]
    #     return self.T_fault_magnet - T_m

    def total_resisting_torque_from_inputs(
        self,
        omega_m: float,
        mass: float,
        slope_deg: float,
        crr: float
    ) -> float:
        slope_rad = np.deg2rad(slope_deg)
        v = self.vehicle_speed_from_motor_speed(omega_m)
        s_v = self.smooth_sign(v, v_eps=0.05)

        # source: https://x-engineer.org/modeling-simulation-vehicle-automatic-transmission/5/ 
        # Grade or slope resistance (19)
        F_grade = mass * self.g * np.sin(slope_rad)

        # Rolling resistance (16)
        F_roll = crr * mass * self.g * np.cos(slope_rad) * s_v

        # Aerodynamic drag (15)
        F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v * abs(v)

        # lumped modeling approximation: a linear viscous-damping force proportional to speed, added to represent losses 
        # that grow roughly with velocity, such as driveline parasitics, bearing drag, and other unmodeled speed-proportional effects
        F_visc = self.b_vehicle * v

        F_total = F_grade + F_roll + F_aero + F_visc

        # Convert force at the tire to wheel torque 
        T_wheel = F_total * self.wheel_radius
        # T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)
        # because of downhill
        '''
        This handles the difference between driving and regenerative braking (downhill) in terms of drivetrain efficiency.
        When Twheel≥0
        Twheel​≥0 (driving, uphill, flat road):The motor is pushing the vehicle forward. Energy flows from motor to wheels, so the drivetrain efficiency 
        ηdt​ acts as a loss — the motor must produce more torque than what arrives at the wheel
        '''
        if T_wheel >= 0.0:
            T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)
        else:
            T_motor_load = T_wheel * self.drivetrain_eff / max(self.gear_ratio, 1e-12)
        return T_motor_load

    def apply_rated_power_limit(self, torque_cmd: float, omega_m: float) -> float:
        """
        Limit torque command so that |T * omega_m| <= P_rated.

        This is a motor rated mechanical power limit, not an inverter limit.
        It should act as a supervisory constraint on torque request.
        """
        if self.P_rated <= 0.0:
            return torque_cmd

        omega_abs = abs(omega_m)

        # Near zero speed, do not apply the power limit.
        # Current limit / torque limit should dominate there.
        if omega_abs < 1e-3:
            return torque_cmd

        torque_power_limit = self.P_rated / omega_abs
        return float(np.clip(torque_cmd, -torque_power_limit, torque_power_limit))

    # =============================================================================
    # ODE
    # =============================================================================
    def motor_ode_pmsm_controlled(self, t: float, x: np.ndarray) -> List[float]:
        """IPMSM ODE in chosen reference frame (dq axes)."""
        # i_d, i_q, omega_m, theta_m, xint_d, xint_q, xint_s = x
        (
            i_d, i_q, omega_m, theta_m,
            xint_d, xint_q, xint_s,
            T_s, T_m,
            omega_ref_f, mass_f, slope_f_deg, crr_f
        ) = x
        # temp ignored
        # (
        #     i_d, i_q, omega_m, theta_m,
        #     xint_d, xint_q, xint_s,
        #     omega_ref_f, mass_f, slope_f_deg, crr_f
        # ) = x
        R_s = self.R_s_of_T(T_s)
        phi_f = float(self.phi_f_of_T(T_m))

        # R_s = self.R_s_ref * (1.0 + self.alpha_cu * (T_s - self.T_ref))
        # phi_f = self.phi_f_ref * (1.0 - self.alpha_phi * (T_m - self.T_ref))
        # if phi_f < self.phi_f_min:
        #     phi_f = self.phi_f_min
        # temp ignored
        # R_s = self.R_s_ref
        # phi_f = self.phi_f_ref
        # xint = integral part is a state

        # rotor electrical angle: theta_r_e = p * theta_m (p=polepairs)
        # rotor electrical quantities
        omega_r_e = self.polepairs * omega_m
        theta_r_e = self.polepairs * theta_m

        # MTPA LUT - beg
        Ld_now, Lq_now = self.motor_inductances(self.machine_type, i_d, i_q, self.mtpa_lut)
        K_pd_now = Ld_now * self.omega_cc  # Eq. (6.28)
        K_pq_now = Lq_now * self.omega_cc  # Eq. (6.29)
        K_i_now = R_s * self.omega_cc
        K_aw_d_now = K_i_now / max(K_pd_now, 1e-12)
        K_aw_q_now = K_i_now / max(K_pq_now, 1e-12)
        # MTPA LUT - end

        # IMPORTANT:
        #  The electrical angle is faster than the mechanical angle.
        #  Reason: each mechanical revolution contains p electrical revolutions.
        #  Because the stator rotating field completes one electrical cycle for each pole pair.

        # ==========================================================
        # REFERENCE FRAME SECTION (what dq means right now)
        # ----------------------------------------------------------
        # frame_mode == "rotor_foc":
        #   theta_frame = theta_r_e
        #   omega_frame = omega_r_e
        #   -> classic rotor-aligned FOC frame for PMSM
        #
        # frame_mode == "stator":
        #   theta_frame = omega_e*t
        #   omega_frame = omega_e
        #   -> stator synchronous frame (needs phase locking to be stable)
        # ==========================================================

        # IMPORTANT:
        #  Phase locking in a PMSM means keeping the stator magnetic field
        #  perfectly synchronized with the rotor’s magnetic field so torque
        #  is produced smoothly and efficiently.
        #  - Phase locking ensures these two fields rotate at the same
        #    electrical speed and maintain the correct angle between them.
        #  - If they stay locked → constant torque
        #  - If they lose lock → torque drops, vibration, or the motor stalls

        # choose frame angle (where dq variables are referenced)
        if self.frame_mode == 'rotor_foc':
            theta_frame = theta_r_e
            omega_frame = omega_r_e
        else:
            theta_frame = self.omega_e * t
            omega_frame = self.omega_e

        omega_ref_tgt = self.omega_ref(t)
        mass_tgt = self.mass_target(t)
        slope_tgt_deg = self.slope_target(t)
        crr_tgt = self.crr_target(t)

        domega_ref_f_dt = self.rate_limited_tracking(
            omega_ref_f,   # current filtered speed reference
            omega_ref_tgt,  # desired target value
            self.tau_omega_ref,  # how quickly to track the target
            self.omega_ref_rate_up,  # maximum rate of increase
            self.omega_ref_rate_down  # maximum rate of decrease
        )    
        dmass_f_dt = self.first_order_tracking(mass_f, mass_tgt, self.tau_mass)
        dslope_f_dt = self.first_order_tracking(slope_f_deg, slope_tgt_deg, self.tau_slope)
        dcrr_f_dt = self.first_order_tracking(crr_f, crr_tgt, self.tau_crr)

        # ==========================================================
        # SPEED CONTROLLER (outer loop)  -> produces torque_ref -> iq_ref
        # ----------------------------------------------------------
        # w_ref = self.omega_ref(t)
        # w_ref = self._seg_omega_ref
        w_ref = omega_ref_f

        # Clamp commanded speed to finite max speed if applicable
        if self.machine_type == "ipmsm":
            omega_max_e_pre = self.ipmsm_max_speed_finite(phi_f)
        else:
            omega_max_e_pre = self.spmsm_max_speed_finite(phi_f)

        if omega_max_e_pre is not None:
            omega_max_m_pre = omega_max_e_pre / self.polepairs
            w_ref = np.clip(w_ref, -omega_max_m_pre, omega_max_m_pre)

        # err_omega = w_ref - omega_m

        # iq_ref_uns = self.K_ps * err_omega + xint_s
        # iq_ref_speed = np.clip(iq_ref_uns, -self.Imax, self.Imax)
        # xint_s_dot = self.K_is * err_omega + self.K_aw_s * (iq_ref_speed - iq_ref_uns)

        J_ctrl_now = self.J_motor + mass_f * self.wheel_radius**2 / (self.gear_ratio**2)
        K_T_now = 1.5 * self.polepairs * phi_f
        K_ps_now = J_ctrl_now * self.omega_cs / max(K_T_now, 1e-12)
        K_is_now = K_ps_now * self.omega_cs / 5.0
        K_aw_s_now = K_is_now / max(K_ps_now, 1e-12)

        err_omega = w_ref - omega_m

        iq_ref_uns = K_ps_now * err_omega + xint_s
        iq_ref_speed = np.clip(iq_ref_uns, -self.Imax, self.Imax)
        xint_s_dot = K_is_now * err_omega + K_aw_s_now * (iq_ref_speed - iq_ref_uns)


        # Current references with field-weakening
        torque_cmd = self.torque_from_iq_command(self.machine_type, iq_ref_speed, phi_f)
        torque_cmd = self.apply_rated_power_limit(torque_cmd, omega_m)

        id_ref, iq_ref, omega_base_e, omega_max_e, max_speed_reached_now = \
            self.current_references_with_fw(self.machine_type, omega_m, torque_cmd, phi_f, self.mtpa_lut)

        # ==========================================================
        # CURRENT CONTROLLERS (inner loop) -> produce v_d, v_q
        # ==========================================================
        err_d = id_ref - i_d
        err_q = iq_ref - i_q

        # PI (unsaturated) -> feedback control (PI)
        u_d_uns = K_pd_now * err_d + xint_d
        u_q_uns = K_pq_now * err_q + xint_q

        # Decoupling + back-EMF feedforward (rotor-aligned dq form)
        v_d_cmd_uns = u_d_uns - omega_frame * Lq_now * i_q
        v_q_cmd_uns = u_q_uns + omega_frame * (Ld_now * i_d + phi_f)

        # inverter dq vector magnitude limit
        v_d_cmd, v_q_cmd = dq_voltage_limit(v_d_cmd_uns, v_q_cmd_uns, self.Vmax)

        # anti-windup should compare the SATURATED controller command
        # against the unsaturated controller request
        xint_d_dot = K_i_now * err_d + K_aw_d_now * (v_d_cmd - v_d_cmd_uns)
        xint_q_dot = K_i_now * err_q + K_aw_q_now * (v_q_cmd - v_q_cmd_uns)

        # ==========================================================
        # DRIVE MODE SECTION (what voltages actually applied)
        # ==========================================================
        if self.drive_mode == "closed_loop_foc":
            if self.use_svpwm:
                inv = self.dq_cmd_to_inverter_output(
                    v_d_cmd=v_d_cmd,
                    v_q_cmd=v_q_cmd,
                    theta_frame=theta_frame,
                    i_d=i_d,
                    i_q=i_q,
                    Vdc=self.Vdc,
                    T_pwm=self.T_pwm,
                    deadtime=self.deadtime,
                    use_deadtime=self.include_deadtime
                )
                v_d = float(inv["v_d_applied"])
                v_q = float(inv["v_q_applied"])
            else:
                v_d, v_q = v_d_cmd, v_q_cmd
        else:
            # CRITICAL: slip speed (works for both frames)
            v_as = self.V_m * np.cos(self.omega_e * t)
            v_bs = self.V_m * np.cos(self.omega_e * t - 2.0 * np.pi / 3.0)
            v_cs = self.V_m * np.cos(self.omega_e * t - 4.0 * np.pi / 3.0)
            v_dq0 = T_matrix(theta_frame) @ np.array([v_as, v_bs, v_cs])
            v_d, v_q, _ = v_dq0
            v_d, v_q = dq_voltage_limit(v_d, v_q, self.Vmax)

        # ==========================================================
        # ELECTRICAL DYNAMICS (dq currents)
        # ==========================================================
        if self.frame_mode == "rotor_foc":
            di_d_dt = (v_d - R_s * i_d + omega_frame * Lq_now * i_q) / Ld_now
            di_q_dt = (v_q - R_s * i_q - omega_frame * (Ld_now * i_d + phi_f)) / Lq_now
        else:
            # ----------------------------------------------------------
            # STATOR SYNCHRONOUS dq frame (theta_frame = omega_e*t)
            # Saliency + PM flux rotate into this frame -> L(t), phi(t)
            # ----------------------------------------------------------
            theta_err = theta_r_e - theta_frame
            theta_err_dot = omega_r_e - omega_frame

            deltaL = (self.L_ds - self.L_qs)

            # time-varying inductances (your Eq. 4.110-style form)
            Ld_eff = 0.5 * (self.L_ds + self.L_qs) + 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
            Lq_eff = 0.5 * (self.L_ds + self.L_qs) - 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
            Ldq_eff = 0.5 * (self.L_ds - self.L_qs) * np.sin(2.0 * theta_err)

            # if SPMSM (deltaL == 0), explicitly zero out coupling to avoid numeric noise
            if abs(deltaL) < 1e-12:
                Ld_eff = self.L_ds
                Lq_eff = self.L_ds
                Ldq_eff = 0.0

            # PM flux projection into the stator synchronous dq frame
            phi_d = phi_f * np.cos(theta_err)
            phi_q = -phi_f * np.sin(theta_err)

            # flux linkages in this frame
            lambda_d = Ld_eff * i_d + Ldq_eff * i_q + phi_d
            lambda_q = Ldq_eff * i_d + Lq_eff * i_q + phi_q

            # voltage equations in rotating dq frame
            dlambda_d_dt = v_d - R_s * i_d + omega_frame * lambda_q
            dlambda_q_dt = v_q - R_s * i_q - omega_frame * lambda_d
            dlambda_dt = np.array([dlambda_d_dt, dlambda_q_dt])

            # time-derivatives of inductances (chain rule)
            if abs(deltaL) < 1e-12:
                dLd_dt = dLq_dt = dLdq_dt = 0.0
            else:
                dLd_dt = -deltaL * np.sin(2.0 * theta_err) * theta_err_dot
                dLq_dt = deltaL * np.sin(2.0 * theta_err) * theta_err_dot
                dLdq_dt = deltaL * np.cos(2.0 * theta_err) * theta_err_dot

            # time derivative of PM flux projections
            dphi_d_dt = -phi_f * np.sin(theta_err) * theta_err_dot
            dphi_q_dt = -phi_f * np.cos(theta_err) * theta_err_dot

            # assemble L matrix and its derivative effect
            Lmat = np.array([[Ld_eff, Ldq_eff],
                             [Ldq_eff, Lq_eff]])

            dL_times_i = np.array([
                dLd_dt * i_d + dLdq_dt * i_q,
                dLdq_dt * i_d + dLq_dt * i_q
            ])

            dphi_dt = np.array([dphi_d_dt, dphi_q_dt])

            rhs = dlambda_dt - dL_times_i - dphi_dt
            di_d_dt, di_q_dt = np.linalg.solve(Lmat, rhs)

        # ==========================================================
        # TORQUE
        # ==========================================================
        if self.frame_mode == "rotor_foc":
            T_e = 1.5 * self.polepairs * (phi_f * i_q + (Ld_now - Lq_now) * i_d * i_q)
        else:
            T_e = 1.5 * self.polepairs * (lambda_d * i_q - lambda_q * i_d)

        # Mechanical dynamics
        # eq. 1.59
        # J_total = self.total_inertia(t)
        # T_load = self.total_resisting_torque(t, omega_m)
        # J_total = self._seg_J_total
        # T_load = self.total_resisting_torque_segment(omega_m)
        # J_total = self.J_motor + mass_f * self.wheel_radius**2 / (self.gear_ratio**2)
        # T_load = self.total_resisting_torque_from_inputs(omega_m, mass_f, slope_f_deg, crr_f)
        # J_total = self.J_motor + mass_tgt * self.wheel_radius**2 / (self.gear_ratio**2)
        # T_load = self.total_resisting_torque_from_inputs(omega_m, mass_tgt, slope_tgt_deg, crr_tgt)
        J_total = self.J_motor + mass_f * self.wheel_radius**2 / (self.gear_ratio**2)
        T_load = self.total_resisting_torque_from_inputs(omega_m, mass_f, slope_f_deg, crr_f)
        domega_m_dt = (T_e - T_load) / J_total
        dtheta_m_dt = omega_m

        # Thermal model =================================================================================================================================
        # temp ignored
        # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://repositum.tuwien.at/bitstream/20.500.12708/196526/1/Fischer%20Andreas%20-%202024%20-%20Model-based%20thermal%20protection%20strategy%20for%20a%20permanent...pdf
            # eq. 2.7
        P_cu = 1.5 * R_s * (i_d ** 2 + i_q ** 2)  # Copper loss

        # Iron (core) loss — Steinmetz model, gated by include_iron_loss.
        # B_core is approximated from the dq flux-linkage magnitude, scaled so that
        # the flux-linkage magnitude at RATED operation (id=0, iq=I_rated) maps to
        # B_iron_rated. Using phi_f_ref alone (the no-load PM flux) as the
        # reference would badly under-scale it whenever armature-reaction flux
        # (Lq*iq) dominates, which is common for weak-magnet/high-leakage motors.
        # This model has no explicit tooth/yoke geometry, so this lumped scaling
        # stands in for the article's per-tooth/per-yoke flux-density waveform.
        if self.include_iron_loss:
            lambda_d = Ld_now * i_d + phi_f
            lambda_q = Lq_now * i_q
            psi_s_mag = float(np.hypot(lambda_d, lambda_q))
            lambda_rated = float(np.hypot(self.phi_f_ref, self.Lq0 * self.I_rated))
            B_core = self.B_iron_rated * psi_s_mag / max(lambda_rated, 1e-12)
            omega_s = abs(omega_r_e)
            P_iron = self.core_volume_m3 * (
                self.k_h_iron * (B_core ** self.beta_iron) * omega_s
                + self.k_e_iron * (B_core ** 2) * (omega_s ** 2)
            )
        else:
            P_iron = 0.0

        # # TODO: Simple rotor/magnet loss placeholder
        P_rotor = 0.0

        # # Thermal dynamics
        # # rate of change of stator temperature
        # # stator temperature rate =
            #     # (copper loss heat input
            #     # + iron loss heat input
            #     # - heat from stator to ambient
            #     # - heat from stator to magnet/rotor)
            #     # / stator thermal capacitance
        dT_s_dt = (P_cu + P_iron - (T_s - self.T_amb) / self.R_th_sa - (T_s - T_m) / self.R_th_sm) / self.C_th_s
        # # rate of change of rotor/magnet temperature
        # # magnet/rotor temperature rate =
            #     # (rotor loss heat input
            #     # + heat from stator to magnet/rotor
            #     # - heat from magnet/rotor to ambient)
            #     # / magnet/rotor thermal capacitance
        dT_m_dt = (P_rotor + (T_s - T_m) / self.R_th_sm - (T_m - self.T_amb) / self.R_th_ma) / self.C_th_m
        # ===============================================================================================================================================

        # return [di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt, xint_d_dot, xint_q_dot, xint_s_dot]
        # return [di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt, xint_d_dot, xint_q_dot, xint_s_dot, dT_s_dt, dT_m_dt]
        # temp ignored
        return [
            di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt,
            xint_d_dot, xint_q_dot, xint_s_dot,
            dT_s_dt, dT_m_dt,
            domega_ref_f_dt, dmass_f_dt, dslope_f_dt, dcrr_f_dt
        ]
        # return [
        #     di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt,
        #     xint_d_dot, xint_q_dot, xint_s_dot,
        #     domega_ref_f_dt, dmass_f_dt, dslope_f_dt, dcrr_f_dt
        # ]
    # =============================================================================
    # SOLVE + POSTPROCESS + FIGURES
    # =============================================================================
    def run(self) -> Dict[str, Any]:
        # =============================================================================
        # INITIAL CONDITIONS
        # =============================================================================
        i_d0, i_q0 = 0.0, 0.0
        omega_m0 = 0.0
        theta_m0 = 0.0
        xint_d0 = 0.0
        xint_q0 = 0.0
        xint_s0 = 0.0
        # temp ignored
        T_s0 = self.T_amb
        T_m0 = self.T_amb

        # steps
        omega_ref_f0 = self.omega_ref(self.t0)
        mass_f0 = self.mass_target(self.t0)
        slope_f0 = self.slope_target(self.t0)
        crr_f0 = self.crr_target(self.t0)

        # x0 = np.array([i_d0, i_q0, omega_m0, theta_m0, xint_d0, xint_q0, xint_s0, T_s0, T_m0])
        # temp ignored
        x0 = np.array([
            i_d0, i_q0, omega_m0, theta_m0,
            xint_d0, xint_q0, xint_s0,
            T_s0, T_m0,
            omega_ref_f0, mass_f0, slope_f0, crr_f0
        ])
        # x0 = np.array([
        #     i_d0, i_q0, omega_m0, theta_m0,
        #     xint_d0, xint_q0, xint_s0,
        #     omega_ref_f0, mass_f0, slope_f0, crr_f0
        # ])

        # =============================================================================
        # SOLVE
        # =============================================================================
        # TODO: solve_ivp research and mention in the thesis how it works
        '''
        - Numerically computes the state trajectory of a system starting from an initial state
        - So for each time t and state vector x, it returns the derivative dx/dt.
        - Explicit solver: directly estimates the next value from the current slope
        - Implicit solver: solves an equation involving the unknown next value itself
          - That means the solver must solve for it numerically.
          - This is more computationally expensive per step, but much more stable for stiff systems.
        - The solver is always estimating its own local error and adjusting step size.
        - allowed error≈atol+rtol⋅∣y∣
        '''
        # temp ignored
        def stator_fault_event(t, x):
            T_s = x[7]
            return self.T_fault_stator - T_s

        # stator_fault_event.terminal = True
        # stator_fault_event.direction = -1

        def magnet_fault_event(t, x):
            T_m = x[8]
            return self.T_fault_magnet - T_m

        # magnet_fault_event.terminal = True
        # magnet_fault_event.direction = -1

        print("BEFORE SOLVE:")
        print("tfinal =", self.tfinal)
        print("t_eval start/end =", self.t_eval[0], self.t_eval[-1])
        print("t_steps =", self.t_steps)
        print("omega_steps =", self.omega_steps)
        # print("breakpoints =", self.get_schedule_breakpoints())

        # t, y, segment_solutions, stop_info = self.solve_piecewise(x0)
        # temp ignored
        # sol = solve_ivp(
        #     self.motor_ode_pmsm_controlled,
        #     (self.t0, self.tfinal),
        #     x0,
        #     method=self.solver_method,
        #     t_eval=self.t_eval,
        #     rtol=self.rtol,
        #     atol=self.atol,
        #     events=[stator_fault_event, magnet_fault_event],
        # )
        sol = solve_ivp(
            self.motor_ode_pmsm_controlled,
            (self.t0, self.tfinal),
            x0,
            method=self.solver_method,
            t_eval=self.t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        t = sol.t
        y = sol.y

        stop_info = {
            "success": sol.success,
            "status": sol.status,
            "message": sol.message,
            "failed_segment": None,
            "failed_time": float(sol.t[-1]) if sol.t.size else None,
            # temp ignored
            # "t_events": [
            #     sol.t_events[0].tolist() if len(sol.t_events) > 0 else [],
            #     sol.t_events[1].tolist() if len(sol.t_events) > 1 else [],
            # ],
        }

        print("AFTER SOLVE:")
        print("final returned time =", t[-1])
        print("success =", stop_info["success"])
        print("status =", stop_info["status"])
        print("message =", stop_info["message"])
        print("failed segment =", stop_info["failed_segment"])
        print("failed time =", stop_info["failed_time"])

        print(
            "piecewise summary:",
            stop_info["success"],
            stop_info["status"],
            t[-1],
            stop_info["message"]
        )

        # ------------------- check event triggers -------------------
        # event_messages: List[str] = []

        # if sol.t_events[0].size > 0:
        #     event_messages.append(f"FAULT: stator temperature exceeded {self.T_fault_stator:.1f} °C at t = {sol.t_events[0][0]:.6f} s")

        # if sol.t_events[1].size > 0:
        #     event_messages.append(f"FAULT: magnet temperature exceeded {self.T_fault_magnet:.1f} °C at t = {sol.t_events[1][0]:.6f} s")

        # if sol.status == 1:
        #     event_messages.append("Simulation terminated due to thermal fault event.")

        # if sol.t_events[0].size > 0:
        #     event_messages.append(f"Stator fault at t = {sol.t_events[0][0]:.6f} s")

        # if sol.t_events[1].size > 0:
        #     event_messages.append(f"Magnet fault at t = {sol.t_events[1][0]:.6f} s")
        
        event_messages: List[str] = []
        # temp ignored
        # stator_events = stop_info["t_events"][0]
        # magnet_events = stop_info["t_events"][1]

        # if len(stator_events) > 0:
        #     event_messages.append(
        #         f"FAULT: stator temperature exceeded {self.T_fault_stator:.1f} °C at t = {stator_events[0]:.6f} s"
        #     )

        # if len(magnet_events) > 0:
        #     event_messages.append(
        #         f"FAULT: magnet temperature exceeded {self.T_fault_magnet:.1f} °C at t = {magnet_events[0]:.6f} s"
        #     )

        # if stop_info["status"] == 1:
        #     event_messages.append("Simulation terminated due to thermal fault event.")

        # if len(stator_events) > 0:
        #     event_messages.append(f"Stator fault at t = {stator_events[0]:.6f} s")

        # if len(magnet_events) > 0:
        #     event_messages.append(f"Magnet fault at t = {magnet_events[0]:.6f} s")
        # ------------------------------------------------------------

        # t = sol.t
        # i_d = sol.y[0]
        # i_q = sol.y[1]
        # omega_m = sol.y[2]
        # theta_m = sol.y[3]

        i_d = y[0]
        i_q = y[1]
        omega_m = y[2]
        theta_m = y[3]

        # =============================================================================
        # MAX-SPEED REACHED MESSAGE (finite-speed FW cases)
        # =============================================================================
        # phi_f_arr = np.asarray(self.phi_f_of_T(sol.y[8]), dtype=float)
        # temp ignored
        phi_f_arr = np.asarray(self.phi_f_of_T(y[8]), dtype=float)
        # phi_f_arr = np.full_like(t, self.phi_f_ref, dtype=float)

        omega_max_e_arr = np.zeros_like(t)
        for k in range(len(t)):
            if self.machine_type == "ipmsm":
                omg = self.ipmsm_max_speed_finite(phi_f_arr[k])
            else:
                omg = self.spmsm_max_speed_finite(phi_f_arr[k])

            omega_max_e_arr[k] = np.nan if omg is None else omg

        omega_max_m_arr = omega_max_e_arr / self.polepairs
        reached_mask = np.abs(omega_m) >= self.MAX_SPEED_REACHED_TOL * omega_max_m_arr

        max_speed_message = ""
        if np.any(reached_mask):
            k_hit = np.argmax(reached_mask)
            max_speed_message = (
                f"MAX SPEED REACHED: {self.machine_type.upper()} finite-speed limit reached at "
                f"t = {t[k_hit]:.6f} s | "
                f"omega_m = {omega_m[k_hit]:.3f} rad/s | "
                f"speed = {omega_m[k_hit] * 60.0 / (2.0 * np.pi):.1f} rpm"
            )
        else:
            max_speed_message = f"Max-speed status: {self.machine_type.upper()} finite-speed limit not reached."

        # # load info for plotting
        # T_load_arr = np.array([self.total_resisting_torque(tk, omega_m[k]) for k, tk in enumerate(t)])
        # v_vehicle = np.array([self.vehicle_speed_from_motor_speed(w) for w in omega_m])
        # mass_arr = np.array([self.piecewise_step(tk, self.mass_step_times, self.mass_step_values) for tk in t])
        # slope_arr = np.array([self.piecewise_step(tk, self.slope_step_times, self.slope_step_values_deg) for tk in t])
        # crr_arr = np.array([self.piecewise_step(tk, self.crr_step_times, self.crr_step_values) for tk in t])

        # Extract temperatures after the solver
        # T_s = sol.y[7]
        # T_m = sol.y[8]

        # temp ignored
        T_s = y[7]
        T_m = y[8]

        # ================================================================================================================================================
        # temperature warning and fault detection (demagnetisation)
        '''
        | Part              | Main risk              | Consequence           |
        | ----------------- | ---------------------- | --------------------- |
        | **Stator**        | insulation overheating | electrical failure    |
        | **Rotor magnets** | demagnetization        | permanent torque loss |
        Stator temp → protects windings
        Rotor temp → protects magnets
        '''

        warning_messages = []
        # temp ignored
        max_Ts = np.max(T_s)
        max_Tm = np.max(T_m)

        # if max_Ts >= self.T_warn_stator:
        #     t_warn_s = t[np.argmax(T_s >= self.T_warn_stator)]
        #     warning_messages.append(
        #         f"WARNING: stator temperature exceeded warning threshold "
        #         f"({self.T_warn_stator:.1f} °C) at t = {t_warn_s:.4f} s."
        #     )

        # if max_Ts >= self.T_fault_stator:
        #     t_fault_s = t[np.argmax(T_s >= self.T_fault_stator)]
        #     warning_messages.append(
        #         f"FAULT: stator temperature exceeded fault threshold "
        #         f"({self.T_fault_stator:.1f} °C) at t = {t_fault_s:.4f} s."
        #     )

        # if max_Tm >= self.T_warn_magnet:
        #     t_warn_m = t[np.argmax(T_m >= self.T_warn_magnet)]
        #     warning_messages.append(
        #         f"WARNING: magnet temperature exceeded warning threshold "
        #         f"({self.T_warn_magnet:.1f} °C) at t = {t_warn_m:.4f} s."
        #     )

        # if max_Tm >= self.T_fault_magnet:
        #     t_fault_m = t[np.argmax(T_m >= self.T_fault_magnet)]
        #     warning_messages.append(
        #         f"FAULT: magnet temperature exceeded fault threshold "
        #         f"({self.T_fault_magnet:.1f} °C) at t = {t_fault_m:.4f} s."
        #     )

        # if max_Tm >= self.T_demag:
        #     t_demag = t[np.argmax(T_m >= self.T_demag)]
        #     warning_messages.append(
        #         f"DEMAG RISK: magnet temperature exceeded irreversible-demagnetization "
        #         f"risk threshold ({self.T_demag:.1f} °C) at t = {t_demag:.4f} s."
        #     )

        # if not warning_messages:
        #     warning_messages.append("Thermal status: OK. No thermal thresholds exceeded.")
        # ================================================================================================================================================

        # Post torque
        # IMPORTANT:
        #  What is the difference between T_e and Torque?
        #  - T_e = instantaneous electromagnetic torque computed inside the ODE
        #          at that time step
        #        = scalar
        #  * Torque = an array of torque values computed after the solver for
        #             plotting
        #           = vector
        #  So they represent the same physical quantity; one is the current
        #    instant, one is the saved trajectory.

        if self.frame_mode == "rotor_foc":
            Torque = np.zeros_like(t)
            for k in range(len(t)):
                if self.machine_type == "spmsm" or self.mtpa_lut is None:
                    Ld_k, Lq_k = self.L_ds, self.L_qs
                else:
                    Ld_k, Lq_k = self.mtpa_lut.ld(i_d[k]), self.mtpa_lut.lq(i_q[k])

                Torque[k] = 1.5 * self.polepairs * (
                        phi_f_arr[k] * i_q[k] + (Ld_k - Lq_k) * i_d[k] * i_q[k]
                )
        else:
            Torque = np.zeros_like(t)

            for k, tk in enumerate(t):
                omega_mk = omega_m[k]
                theta_mk = theta_m[k]
                i_dk, i_qk = i_d[k], i_q[k]

                theta_r_e = self.polepairs * theta_mk
                theta_frame = self.omega_e * tk
                theta_err = theta_r_e - theta_frame

                Ld_eff = 0.5 * (self.L_ds + self.L_qs) + 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
                Lq_eff = 0.5 * (self.L_ds + self.L_qs) - 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
                Ldq_eff = 0.5 * (self.L_ds - self.L_qs) * np.sin(2.0 * theta_err)

                # temp ignored
                # phi_fk = float(self.phi_f_of_T(T_m[k]))
                phi_fk = self.phi_f_ref
                phi_d = phi_fk * np.cos(theta_err)
                phi_q = -phi_fk * np.sin(theta_err)

                lambda_d = Ld_eff * i_dk + Ldq_eff * i_qk + phi_d
                lambda_q = Ldq_eff * i_dk + Lq_eff * i_qk + phi_q

                Torque[k] = 1.5 * self.polepairs * (lambda_d * i_qk - lambda_q * i_dk)

        # Speed in rpm
        speed_rpm = omega_m * 60.0 / (2.0 * np.pi)

        # =============================================================================
        # BUILD PHASE CURRENTS FOR PLOTTING (dq -> abc in THE SAME theta_frame)
        # =============================================================================
        i_phase_a = np.zeros_like(t)
        i_phase_b = np.zeros_like(t)
        i_phase_c = np.zeros_like(t)

        for k, tk in enumerate(t):
            theta_r_e = self.polepairs * theta_m[k]
            if self.frame_mode == "rotor_foc":
                theta_frame = theta_r_e
            else:
                theta_frame = self.omega_e * tk
            i_abc = dq0_to_abc(np.array([i_d[k], i_q[k], 0.0]), theta_frame)
            i_phase_a[k], i_phase_b[k], i_phase_c[k] = i_abc

        # =============================================================================
        # PLOTS (ALL-IN-ONE SCREEN)
        # =============================================================================

        # ---------- Rebuild controller/reference signals for plotting ----------
        # xint_d = sol.y[4]
        # xint_q = sol.y[5]
        # xint_s = sol.y[6]

        xint_d = y[4]
        xint_q = y[5]
        xint_s = y[6]

        # steps
        # temp ignored
        omega_ref_f_arr = y[9]
        mass_f_arr = y[10]
        slope_f_arr = y[11]
        crr_f_arr = y[12]
        # omega_ref_f_arr = y[7]
        # mass_f_arr = y[8]
        # slope_f_arr = y[9]
        # crr_f_arr = y[10]

        # steps - beg
        # T_load_arr = np.array([
        #     self.total_resisting_torque_from_inputs(omega_m[k], mass_f_arr[k], slope_f_arr[k], crr_f_arr[k])
        #     for k in range(len(t))
        # ])

        # mass_arr = mass_f_arr.copy()
        # slope_arr = slope_f_arr.copy()
        # crr_arr = crr_f_arr.copy()
        # mass_arr = np.array([self.mass_target(tk) for tk in t])
        # slope_arr = np.array([self.slope_target(tk) for tk in t])
        # crr_arr = np.array([self.crr_target(tk) for tk in t])

        v_vehicle = np.array([self.vehicle_speed_from_motor_speed(w) for w in omega_m])

        # T_load_arr = np.array([
        #     self.total_resisting_torque_from_inputs(omega_m[k], mass_arr[k], slope_arr[k], crr_arr[k])
        #     for k in range(len(t))
        # ])

        mass_arr = mass_f_arr.copy()
        slope_arr = slope_f_arr.copy()
        crr_arr = crr_f_arr.copy()

        T_load_arr = np.array([
            self.total_resisting_torque_from_inputs(omega_m[k], mass_arr[k], slope_arr[k], crr_arr[k])
            for k in range(len(t))
        ])
        # steps - end


        # =============================================================================
        # INDUCTANCE-HELPER CURVES FOR FRONTEND PLOT
        # =============================================================================
        id_grid_help = np.linspace(-self.Imax, 0.0, 300)
        iq_grid_help = np.linspace(0.0, self.Imax, 300)

        if self.machine_type == "spmsm" or self.mtpa_lut is None:
            Ld_curve = np.full_like(id_grid_help, self.L_ds, dtype=float)
            Lq_curve = np.full_like(iq_grid_help, self.L_qs, dtype=float)
        else:
            Ld_curve = np.array([self.mtpa_lut.ld(v) for v in id_grid_help], dtype=float)
            Lq_curve = np.array([self.mtpa_lut.lq(v) for v in iq_grid_help], dtype=float)



        id_ref_arr = np.zeros_like(t)
        iq_ref_arr = np.zeros_like(t)  # clamped iq_ref used by current loop

        # NEW: comparison arrays
        iq_ref_pi_arr = np.zeros_like(t)  # q-current from speed PI before MTPA / FW
        id_mtpa_arr = np.zeros_like(t)  # d-current after MTPA (before FW overwrite)
        iq_mtpa_arr = np.zeros_like(t)  # q-current after MTPA (before FW overwrite)
        is_pi_arr = np.zeros_like(t)  # current magnitude from PI-only request
        is_mtpa_arr = np.zeros_like(t)  # current magnitude after MTPA
        torque_cmd_arr = np.zeros_like(t)  # torque command produced from PI current request

        # speed reference (rad/s and rpm for plotting)
        omega_ref_arr = np.zeros_like(t)
        speed_ref_rpm = np.zeros_like(t)

        u_d_uns_arr = np.zeros_like(t)  # PI output before decoupling
        u_q_uns_arr = np.zeros_like(t)

        vd_uns_arr = np.zeros_like(t)  # after decoupling, before saturation
        vq_uns_arr = np.zeros_like(t)

        vd_cmd_arr = np.zeros_like(t)  # after saturation
        vq_cmd_arr = np.zeros_like(t)

        iq_ref_uns_arr = np.zeros_like(t)  # speed PI output before clamp
        iq_ref_clamp_arr = np.zeros_like(t)  # speed PI output after clamp (same as iq_ref_arr)

        v_mag_uns_arr = np.zeros_like(t)  # ||v*|| before saturation (Volts)
        v_mag_cmd_arr = np.zeros_like(t)  # ||v||  after saturation (Volts)

        omega_base_arr = np.zeros_like(t)
        omega_max_arr = np.zeros_like(t)
        max_speed_reached_arr = np.zeros_like(t, dtype=bool)

        # SVPWM / inverter arrays for plotting ----
        va_ref_arr = np.zeros_like(t)
        vb_ref_arr = np.zeros_like(t)
        vc_ref_arr = np.zeros_like(t)

        v_offset_arr = np.zeros_like(t)

        van_ref_arr = np.zeros_like(t)
        vbn_ref_arr = np.zeros_like(t)
        vcn_ref_arr = np.zeros_like(t)

        da_cmd_arr = np.zeros_like(t)
        db_cmd_arr = np.zeros_like(t)
        dc_cmd_arr = np.zeros_like(t)

        da_eff_arr = np.zeros_like(t)
        db_eff_arr = np.zeros_like(t)
        dc_eff_arr = np.zeros_like(t)

        van_avg_arr = np.zeros_like(t)
        vbn_avg_arr = np.zeros_like(t)
        vcn_avg_arr = np.zeros_like(t)

        va_avg_arr = np.zeros_like(t)
        vb_avg_arr = np.zeros_like(t)
        vc_avg_arr = np.zeros_like(t)

        ed_cross_arr = np.zeros_like(t)
        eq_backemf_arr = np.zeros_like(t)
        epm_arr = np.zeros_like(t)
        Ld_arr = np.zeros_like(t)
        Lq_arr = np.zeros_like(t)

        # Current-loop PI gains actually used at each timestep. K_pd/K_pq are
        # NOT constant -- they scale with Ld_now/Lq_now, the current-dependent
        # (saturating) dq inductances, so they move whenever i_d/i_q move into
        # saturation. K_i is shared between both axes (it only depends on R_s,
        # which itself drifts with stator temperature T_s) -- see K_pd_now/
        # K_pq_now/K_i_now just above where they're computed each step.
        K_pd_arr = np.zeros_like(t)
        K_pq_arr = np.zeros_like(t)
        K_i_arr = np.zeros_like(t)

        # Speed-loop (outer) PI gains actually used at each timestep. Also not
        # constant: K_ps ("DC-motor-like" gain, see the speed-loop comment
        # below) scales with J_ctrl_now (effective control inertia, which
        # grows with vehicle mass_f_arr when a load schedule is active) and
        # with K_T_now = 1.5*polepairs*phi_f (the torque constant, which
        # moves whenever phi_f itself moves -- flux weakening, temperature).
        # K_is is tied to K_ps by a fixed ratio (omega_cs/5), so it moves in
        # lockstep -- see K_ps_now/K_is_now just below where they're computed.
        K_ps_arr = np.zeros_like(t)
        K_is_arr = np.zeros_like(t)

        P_cu_arr = np.zeros_like(t)
        P_stator_loss_arr = np.zeros_like(t)
        P_iron_arr = np.zeros_like(t)
        B_core_arr = np.zeros_like(t)
        # Core-loss (R_fe) branch current -- see __init__ for R_fe_iron calibration.
        # This is what actually gets added to the terminal current for E_elec_in in the
        # energy-balance section; it's an independent estimate from P_iron_arr above.
        i_Fe_d_arr = np.zeros_like(t)
        i_Fe_q_arr = np.zeros_like(t)

        # actually applied voltages after PWM with deadtime
        vd_applied_arr = np.zeros_like(t)
        vq_applied_arr = np.zeros_like(t)
        # PWM -------------------------------------

        for k, tk in enumerate(t):
            i_dk, i_qk = i_d[k], i_q[k]
            omega_mk = omega_m[k]

            omega_r_e = self.polepairs * omega_mk
            omega_frame = omega_r_e if self.frame_mode == "rotor_foc" else self.omega_e

            # ---- speed reference ----
            # w_ref = self.omega_ref(tk)
            w_ref = omega_ref_f_arr[k]
            omega_ref_arr[k] = w_ref
            speed_ref_rpm[k] = w_ref * 60.0 / (2.0 * np.pi)

            # ---- speed loop -> iq_ref ----
            err_omega = w_ref - omega_mk

            # temp ignored
            phi_fk = float(self.phi_f_of_T(T_m[k]))
            # phi_fk = self.phi_f_ref

            # ---------------------------------------------------------
            # 1) Raw q-axis current request from speed PI
            # ---------------------------------------------------------
            # iq_ref_uns = self.K_ps * err_omega + xint_s[k]
            J_ctrl_now = self.J_motor + mass_f_arr[k] * self.wheel_radius**2 / (self.gear_ratio**2)
            K_T_now = 1.5 * self.polepairs * phi_fk
            K_ps_now = J_ctrl_now * self.omega_cs / max(K_T_now, 1e-12)
            K_is_now = K_ps_now * self.omega_cs / 5.0
            K_aw_s_now = K_is_now / max(K_ps_now, 1e-12)

            K_ps_arr[k] = K_ps_now
            K_is_arr[k] = K_is_now

            iq_ref_uns = K_ps_now * err_omega + xint_s[k]
            iq_ref_k = np.clip(iq_ref_uns, -self.Imax, self.Imax)

            iq_ref_uns_arr[k] = iq_ref_uns
            iq_ref_clamp_arr[k] = iq_ref_k
            iq_ref_pi_arr[k] = iq_ref_k

            # PI-only current magnitude:
            # below FW this is equivalent to id=0, iq=iq_ref_k
            is_pi_arr[k] = np.abs(iq_ref_k)

            # Convert speed-loop q-current request to torque command
            torque_ref_k = self.torque_from_iq_command(self.machine_type, iq_ref_k, phi_fk)
            torque_ref_k = self.apply_rated_power_limit(torque_ref_k, omega_mk)
            torque_cmd_arr[k] = torque_ref_k

            # ---------------------------------------------------------
            # 2) MTPA-only current references (before FW logic)
            # ---------------------------------------------------------
            id_mtpa_k, iq_mtpa_k = self.pmsm_torque_to_current_refs(
                machine_type=self.machine_type,
                torque_ref=torque_ref_k,
                phi_f=phi_fk,
                mtpa_lut=self.mtpa_lut,
            )

            id_mtpa_arr[k] = id_mtpa_k
            iq_mtpa_arr[k] = iq_mtpa_k
            is_mtpa_arr[k] = np.hypot(id_mtpa_k, iq_mtpa_k)

            # ---------------------------------------------------------
            # 3) Final references after MTPA + field weakening
            # ---------------------------------------------------------
            id_ref_k, iq_ref_k_fw, omega_base_e_k, omega_max_e_k, max_speed_reached_k = \
                self.current_references_with_fw(self.machine_type, omega_mk, torque_ref_k, phi_fk, self.mtpa_lut)

            id_ref_arr[k] = id_ref_k
            iq_ref_arr[k] = iq_ref_k_fw

            omega_base_arr[k] = omega_base_e_k / self.polepairs
            omega_max_arr[k] = omega_max_e_k / self.polepairs
            max_speed_reached_arr[k] = max_speed_reached_k

            # ---- current-dependent inductances and gains ----
            # temp ignored
            # phi_fk = float(self.phi_f_of_T(T_m[k]))
            # phi_fk = self.phi_f_ref
            R_sk = self.R_s_of_T(T_s[k])



            Ld_now, Lq_now = self.motor_inductances(self.machine_type, i_dk, i_qk, self.mtpa_lut)

            Ld_arr[k] = Ld_now
            Lq_arr[k] = Lq_now

            omega_e_k = self.polepairs * omega_mk  # rotor electrical speed
            ed_cross_arr[k] = -omega_e_k * Lq_now * i_qk
            eq_backemf_arr[k] = omega_e_k * (Ld_now * i_dk + phi_fk)
            epm_arr[k] = omega_e_k * phi_fk

            P_cu_arr[k] = 1.5 * R_sk * (i_d[k] ** 2 + i_q[k] ** 2)

            if self.include_iron_loss:
                lambda_d_k = Ld_now * i_dk + phi_fk
                lambda_q_k = Lq_now * i_qk
                psi_s_mag_k = float(np.hypot(lambda_d_k, lambda_q_k))
                lambda_rated_k = float(np.hypot(self.phi_f_ref, self.Lq0 * self.I_rated))
                B_core_arr[k] = self.B_iron_rated * psi_s_mag_k / max(lambda_rated_k, 1e-12)
                omega_s_k = abs(omega_e_k)
                P_iron_arr[k] = self.core_volume_m3 * (
                    self.k_h_iron * (B_core_arr[k] ** self.beta_iron) * omega_s_k
                    + self.k_e_iron * (B_core_arr[k] ** 2) * (omega_s_k ** 2)
                )

                # Core-loss branch current: Ohm's law across the back-EMF, using the
                # FIXED R_fe_iron calibrated in __init__ (not derived from P_iron_arr[k]
                # itself -- see the note there on why that independence matters).
                E_d_k = -omega_e_k * lambda_q_k
                E_q_k = omega_e_k * lambda_d_k
                i_Fe_d_arr[k] = E_d_k / self.R_fe_iron
                i_Fe_q_arr[k] = E_q_k / self.R_fe_iron

            K_pd_now = Ld_now * self.omega_cc
            K_pq_now = Lq_now * self.omega_cc
            K_i_now = R_sk * self.omega_cc

            K_pd_arr[k] = K_pd_now
            K_pq_arr[k] = K_pq_now
            K_i_arr[k] = K_i_now

            # ---- current loop PI ----
            err_d = id_ref_k - i_dk
            err_q = iq_ref_k_fw - i_qk

            u_d_uns = K_pd_now * err_d + xint_d[k]
            u_q_uns = K_pq_now * err_q + xint_q[k]

            u_d_uns_arr[k] = u_d_uns
            u_q_uns_arr[k] = u_q_uns

            # ---- decoupling + back-EMF feedforward ----
            vd_uns = u_d_uns - omega_frame * Lq_now * i_qk
            vq_uns = u_q_uns + omega_frame * (Ld_now * i_dk + phi_fk)

            vd_uns_arr[k] = vd_uns
            vq_uns_arr[k] = vq_uns

            v_mag_uns_arr[k] = np.sqrt(vd_uns * vd_uns + vq_uns * vq_uns)

            # ---- inverter magnitude saturation ----
            vd_cmd, vq_cmd = dq_voltage_limit(vd_uns, vq_uns, self.Vmax)

            vd_cmd_arr[k] = vd_cmd
            vq_cmd_arr[k] = vq_cmd

            v_mag_cmd_arr[k] = np.sqrt(vd_cmd * vd_cmd + vq_cmd * vq_cmd)

            # ------------------------------------------
            # SVPWM + dead-time reconstruction for plots
            # ------------------------------------------
            if self.frame_mode == "rotor_foc":
                theta_frame_k = self.polepairs * theta_m[k]
            else:
                theta_frame_k = self.omega_e * tk

            inv_k = self.dq_cmd_to_inverter_output(
                v_d_cmd=vd_cmd,
                v_q_cmd=vq_cmd,
                theta_frame=theta_frame_k,
                i_d=i_dk,
                i_q=i_qk,
                Vdc=self.Vdc,
                T_pwm=self.T_pwm,
                deadtime=self.deadtime,
                use_deadtime=self.include_deadtime
            )

            va_ref_arr[k], vb_ref_arr[k], vc_ref_arr[k] = inv_k["v_abc_ref"]
            v_offset_arr[k] = inv_k["v_offset"]

            van_ref_arr[k], vbn_ref_arr[k], vcn_ref_arr[k] = inv_k["v_pole_ref"]
            da_cmd_arr[k], db_cmd_arr[k], dc_cmd_arr[k] = inv_k["d_cmd"]
            da_eff_arr[k], db_eff_arr[k], dc_eff_arr[k] = inv_k["d_eff"]

            van_avg_arr[k], vbn_avg_arr[k], vcn_avg_arr[k] = inv_k["v_pole_avg"]
            va_avg_arr[k], vb_avg_arr[k], vc_avg_arr[k] = inv_k["v_abc_avg"]

            vd_applied_arr[k] = float(inv_k["v_d_applied"])
            vq_applied_arr[k] = float(inv_k["v_q_applied"])

        # =============================================================================
        # PWM ZOOM WINDOW FOR PLOTTING
        # =============================================================================
        t_zoom_0 = max(self.t0, self.plot_pwm_zoom_start)
        t_zoom_1 = min(self.tfinal, self.plot_pwm_zoom_end)
        if t_zoom_1 <= t_zoom_0:
            raise ValueError(
                f"Invalid PWM zoom window: start={self.plot_pwm_zoom_start}, end={self.plot_pwm_zoom_end}"
            )

        t_zoom = np.linspace(t_zoom_0, t_zoom_1, self.pwm_plot_samples)

        # interpolate slow envelopes from solver time grid
        van_ref_zoom = np.interp(t_zoom, t, van_ref_arr)
        vbn_ref_zoom = np.interp(t_zoom, t, vbn_ref_arr)
        vcn_ref_zoom = np.interp(t_zoom, t, vcn_ref_arr)

        da_zoom = np.interp(t_zoom, t, da_cmd_arr)
        db_zoom = np.interp(t_zoom, t, db_cmd_arr)
        dc_zoom = np.interp(t_zoom, t, dc_cmd_arr)

        # normalize pole refs for carrier comparison: m in [-1, 1]
        ma_zoom = 2.0 * van_ref_zoom / self.Vdc
        mb_zoom = 2.0 * vbn_ref_zoom / self.Vdc
        mc_zoom = 2.0 * vcn_ref_zoom / self.Vdc

        carrier_zoom = triangular_carrier_from_time(t_zoom - t_zoom_0, self.T_pwm)

        # gate waveforms with dead-time for all 3 phases
        gA_up, gA_lo = make_center_aligned_gates_with_deadtime(
            t_zoom=t_zoom - t_zoom_0,
            duty_cmd=da_zoom,
            T_pwm=self.T_pwm,
            deadtime=self.deadtime if self.include_deadtime else 0.0
        )

        gB_up, gB_lo = make_center_aligned_gates_with_deadtime(
            t_zoom=t_zoom - t_zoom_0,
            duty_cmd=db_zoom,
            T_pwm=self.T_pwm,
            deadtime=self.deadtime if self.include_deadtime else 0.0
        )

        gC_up, gC_lo = make_center_aligned_gates_with_deadtime(
            t_zoom=t_zoom - t_zoom_0,
            duty_cmd=dc_zoom,
            T_pwm=self.T_pwm,
            deadtime=self.deadtime if self.include_deadtime else 0.0
        )

        # =============================================================================
        # NUMERICAL SUMMARY: PI CURRENT vs MTPA CURRENT
        # =============================================================================
        # pi_mtpa_summary = [
        #     "--- PI current vs MTPA current summary ---",
        #     f"machine_type = {self.machine_type}",
        #     f"Max |i_q(PI before MTPA)|       = {np.max(np.abs(iq_ref_pi_arr)):.6f} A",
        #     f"Max |i_q(after MTPA)|          = {np.max(np.abs(iq_mtpa_arr)):.6f} A",
        #     f"Max |i_d(after MTPA)|          = {np.max(np.abs(id_mtpa_arr)):.6f} A",
        #     f"Max |I_s|(PI-only)             = {np.max(is_pi_arr):.6f} A",
        #     f"Max |I_s|(after MTPA)          = {np.max(is_mtpa_arr):.6f} A",
        #     f"Max reduction in |I_s| by MTPA = {np.max(is_pi_arr - is_mtpa_arr):.6f} A",
        # ]
        pi_mtpa_summary = [
            "--- Current-reference comparison summary ---",
            f"machine_type = {self.machine_type}",
            f"use_mtpa = {self.use_mtpa}",
            f"use_field_weakening = {self.use_field_weakening}",
            f"Max |i_q(speed PI)|              = {np.max(np.abs(iq_ref_pi_arr)):.6f} A",
            f"Max |i_q(before FW)|             = {np.max(np.abs(iq_mtpa_arr)):.6f} A",
            f"Max |i_d(before FW)|             = {np.max(np.abs(id_mtpa_arr)):.6f} A",
            f"Max |I_s|(speed PI)              = {np.max(is_pi_arr):.6f} A",
            f"Max |I_s|(before FW)             = {np.max(is_mtpa_arr):.6f} A",
        ]

        # figures = {
        #     "speed_time": make_xy_plot(
        #         t,
        #         [(speed_rpm, "Speed (rpm)"), (speed_ref_rpm, "Speed ref (rpm)")],
        #         "Speed-Time Development",
        #         "Time (s)",
        #         "Speed (rpm)",
        #     ),
        #     "torque_time": make_xy_plot(
        #         t,
        #         [(Torque, "Torque (Nm)"), (T_load_arr, "Load torque (Nm)")],
        #         "Torque-Time Development",
        #         "Time (s)",
        #         "Torque (Nm)",
        #     ),
        #     "phase_currents": make_xy_plot(
        #         t,
        #         [(i_phase_a, "i_a"), (i_phase_b, "i_b"), (i_phase_c, "i_c")],
        #         "Phase Currents (abc)",
        #         "Time (s)",
        #         "Current (A)",
        #     ),
        #     "torque_speed": make_xy_plot(
        #         speed_rpm,
        #         [(Torque, "Torque–Speed")],
        #         "Torque–Speed Characteristic",
        #         "Speed (rpm)",
        #         "Torque (Nm)",
        #     ),
        #     "id_tracking": make_xy_plot(
        #         t,
        #         [(i_d, "i_d"), (id_ref_arr, "i_d_ref")],
        #         "d-axis current tracking",
        #         "Time (s)",
        #         "Current (A)",
        #     ),
        #     "iq_tracking": make_xy_plot(
        #         t,
        #         [(i_q, "i_q (actual)"), (iq_ref_arr, "i_q_ref (clamped)")],
        #         "q-axis current tracking",
        #         "Time (s)",
        #         "Current (A)",
        #     ),
        #     "current_pi": make_xy_plot(
        #         t,
        #         [(u_d_uns_arr, "u_d (PI out)"), (u_q_uns_arr, "u_q (PI out)")],
        #         "Current PI output (unsat)",
        #         "Time (s)",
        #         "Voltage (V)",
        #     ),
        #     "dq_voltage": make_xy_plot(
        #         t,
        #         [(vd_cmd_arr, "v_d (applied)"), (vq_cmd_arr, "v_q (applied)")],
        #         "Commanded dq voltages (sat)",
        #         "Time (s)",
        #         "Voltage (V)",
        #     ),
        #     "speed_sat": make_xy_plot(
        #         t,
        #         [(iq_ref_uns_arr, "i_q_ref* (unsat)"), (iq_ref_clamp_arr, "i_q_ref (clamped)"), (i_q, "i_q (actual)")],
        #         "Speed loop saturation",
        #         "Time (s)",
        #         "Current (A)",
        #         hlines=[self.Imax, -self.Imax],
        #     ),
        #     "voltage_sat": make_xy_plot(
        #         t,
        #         [(v_mag_uns_arr, "||v*|| (request)"), (v_mag_cmd_arr, "||v|| (applied)"), (np.full_like(t, self.Vmax), "Vmax")],
        #         "Voltage saturation",
        #         "Time (s)",
        #         "Voltage (V)",
        #     ),
        #     "pwm_compare": make_xy_plot(
        #         t_zoom,
        #         [(carrier_zoom, "carrier"), (ma_zoom, "m_a"), (mb_zoom, "m_b"), (mc_zoom, "m_c")],
        #         "SVPWM carrier comparison (zoom)",
        #         "Time (s)",
        #         "Normalized",
        #     ),
        #     "switch_states": make_switch_plot(t_zoom, gA_up, gB_up, gC_up),
        #     "thermal": make_xy_plot(
        #         t,
        #         [(T_s, "T_s"), (T_m, "T_m")],
        #         "Thermal states",
        #         "Time (s)",
        #         "Temperature (°C)",
        #         hlines=[self.T_warn_stator, self.T_fault_stator, self.T_warn_magnet, self.T_fault_magnet, self.T_demag],
        #     ),
        #     "vehicle": make_xy_plot(
        #         t,
        #         [(v_vehicle * 3.6, "Vehicle speed (km/h)"), (mass_arr, "Mass (kg)"), (slope_arr, "Slope (deg)"), (crr_arr, "Crr")],
        #         "Vehicle / load schedules",
        #         "Time (s)",
        #         "Mixed units",
        #     ),
        #     # "mtpa_compare": make_xy_plot(
        #     #     t,
        #     #     [(iq_ref_pi_arr, "i_q from PI (before MTPA)"), (iq_mtpa_arr, "i_q after MTPA"), (iq_ref_arr, "i_q final (after MTPA + FW)")],
        #     #     "q-axis current: PI output vs after MTPA",
        #     #     "Time (s)",
        #     #     "Current (A)",
        #     # ),
        #     "mtpa_q_compare": make_xy_plot(
        #         t,
        #         [(iq_ref_pi_arr, "i_q from PI (before MTPA)"),
        #         (iq_mtpa_arr, "i_q after MTPA"),
        #         (iq_ref_arr, "i_q final (after MTPA + FW)")],
        #         "q-axis current: PI output vs after MTPA",
        #         "Time (s)",
        #         "Current (A)",
        #     ),

        #     "mtpa_d_compare": make_xy_plot(
        #         t,
        #         [(np.zeros_like(t), "i_d before MTPA (=0)"),
        #         (id_mtpa_arr, "i_d after MTPA"),
        #         (id_ref_arr, "i_d final (after FW)")],
        #         "d-axis current introduced by MTPA",
        #         "Time (s)",
        #         "Current (A)",
        #     ),

        #     "mtpa_is_compare": make_xy_plot(
        #         t,
        #         [(is_pi_arr, "|I_s| from PI-only"),
        #         (is_mtpa_arr, "|I_s| after MTPA")],
        #         "Current magnitude: PI-only vs MTPA",
        #         "Time (s)",
        #         "Current magnitude (A)",
        #         hlines=[self.Imax],
        #     ),

        #     "mtpa_torque_cmd": make_xy_plot(
        #         t,
        #         [(torque_cmd_arr, "Torque command")],
        #         "Torque command produced from speed PI",
        #         "Time (s)",
        #         "Torque command (Nm)",
        #     ),
        # }

        
        print("max |iq_ref_uns| =", np.max(np.abs(iq_ref_uns_arr)))
        print("max |iq_ref_clamp| =", np.max(np.abs(iq_ref_clamp_arr)))
        print("max |Torque| =", np.max(np.abs(Torque)))
        print("max |T_load| =", np.max(np.abs(T_load_arr)))
        print("max v_mag_uns =", np.max(v_mag_uns_arr))
        print("Vmax =", self.Vmax)
        print("max torque_cmd =", np.max(np.abs(torque_cmd_arr)))

        # =================================================================================
        # Energy balance check (validation aid, not part of the control physics)
        # ---------------------------------------------------------------------------------
        # Over the run [0, t[-1]]:
        #   E_elec_in  = integral of 1.5*(v_d*i_d_terminal + v_q*i_q_terminal) dt, where
        #                i_*_terminal = i_d/i_q (torque-producing current) PLUS i_Fe_d/
        #                i_Fe_q (the core-loss branch current from the fixed R_fe_iron
        #                calibrated in __init__ -- 0 if include_iron_loss is False, so
        #                this reduces to the plain 1.5*(v_d*i_d + v_q*i_q) exactly as
        #                before whenever iron loss is off).
        #   E_mech     = integral of Torque * omega_m dt          (electromagnetic energy
        #                delivered to the shaft -- covers both rotor KE change and any
        #                load work; it does not need to be split further for this check)
        #   E_cu       = integral of P_cu_arr dt                  (copper/I^2R loss energy)
        #   E_iron     = integral of P_iron_arr dt                (iron/core loss energy,
        #                0 J if include_iron_loss is False) -- this is the Steinmetz/
        #                thermal estimate, independent of the R_fe/i_Fe electrical draw
        #                above (see E_iron_electrical below for the comparison).
        #   dE_mag     = change in stored dq inductor coenergy, 0.5*(Ld*i_d^2 + Lq*i_q^2),
        #                from t=0 to t=t[-1] (small, but included so the residual reflects
        #                real unmodeled effects rather than this known/expected term)
        #
        # Ideally: E_elec_in == E_mech + E_cu + E_iron + dE_mag
        # The leftover (E_elec_in - the rest) is reported as the residual, in both J and
        # as a % of E_elec_in, as a sanity/validation indicator -- not a strict physics
        # guarantee (see caveats in the printed message for open-loop / dead-time cases).
        # =================================================================================
        # Must mirror the ODE's actual branch exactly (see "DRIVE MODE SECTION" above):
        # v_d/v_q come from the SVPWM reconstruction whenever use_svpwm is True --
        # include_deadtime only controls whether dead-time distortion is added INSIDE
        # that reconstruction, it does not gate whether SVPWM averaging happens at all.
        # (An earlier version of this check also required include_deadtime here, which
        # compared the wrong voltage -- pre-SVPWM commanded vs. actually-applied -- for
        # use_svpwm=True/include_deadtime=False and produced a spurious ~30% residual.)
        if self.drive_mode == "closed_loop_foc" and self.use_svpwm:
            v_d_used = vd_applied_arr
            v_q_used = vq_applied_arr
            v_source_note = "applied dq voltage (after SVPWM" + (" + dead-time)" if self.include_deadtime else ")")
        else:
            v_d_used = vd_cmd_arr
            v_q_used = vq_cmd_arr
            v_source_note = "commanded dq voltage"

        # Terminal current actually drawn from the supply = torque-producing current
        # (i_d, i_q) + core-loss branch current (i_Fe_d_arr, i_Fe_q_arr -- zero unless
        # include_iron_loss is True). See __init__ for the R_fe_iron calibration and the
        # postprocessing loop above for how i_Fe_*_arr is computed each step.
        i_d_terminal_arr = i_d + i_Fe_d_arr
        i_q_terminal_arr = i_q + i_Fe_q_arr

        P_elec_in_arr = 1.5 * (v_d_used * i_d_terminal_arr + v_q_used * i_q_terminal_arr)
        P_mech_arr = Torque * omega_m
        # Independent cross-check: power actually dissipated in the R_fe branch itself,
        # P = I^2*R = 1.5*R_fe_iron*(i_Fe_d^2+i_Fe_q^2) (equivalent to E_backemf * i_Fe,
        # since i_Fe = E_backemf/R_fe_iron by construction), as opposed to E_iron below,
        # which is the separate Steinmetz/thermal estimate. These are two different
        # models of the same physical loss -- they should track each other reasonably
        # closely if both are implemented correctly, but are not defined to be
        # identical (see the R_fe_iron calibration note in __init__).
        # (R_fe_iron is +inf when iron loss is off, and i_Fe_*_arr are all zero then --
        # guard against 0 * inf = nan and just report zero.)
        if self.include_iron_loss:
            P_iron_electrical_arr = 1.5 * self.R_fe_iron * (i_Fe_d_arr ** 2 + i_Fe_q_arr ** 2)
        else:
            P_iron_electrical_arr = np.zeros_like(t)

        E_elec_in = float(_trapz(P_elec_in_arr, t))
        E_mech = float(_trapz(P_mech_arr, t))
        E_cu = float(_trapz(P_cu_arr, t))
        E_iron = float(_trapz(P_iron_arr, t))
        E_iron_electrical = float(_trapz(P_iron_electrical_arr, t))

        W_mag_0 = 0.5 * (Ld_arr[0] * i_d[0] ** 2 + Lq_arr[0] * i_q[0] ** 2)
        W_mag_f = 0.5 * (Ld_arr[-1] * i_d[-1] ** 2 + Lq_arr[-1] * i_q[-1] ** 2)
        dE_mag = float(W_mag_f - W_mag_0)

        E_accounted = E_mech + E_cu + E_iron + dE_mag
        E_residual = E_elec_in - E_accounted
        E_residual_pct = 100.0 * E_residual / max(abs(E_elec_in), 1e-9)

        if abs(E_residual_pct) < 2.0:
            balance_verdict = f"OK -- residual is {E_residual_pct:+.2f}% of E_elec_in, within expected numerical tolerance."
        else:
            balance_verdict = (
                f"CHECK -- residual is {E_residual_pct:+.2f}% of E_elec_in, larger than the "
                f"~2% expected from numerical integration alone. This can be normal for "
                f"open_loop_abc (voltage source isn't the closed-loop dq command path used "
                f"here) or short/near-zero-current runs (E_elec_in close to 0 makes the % "
                f"noisy); otherwise it may be worth checking rtol/atol or reported settings."
            )

        energy_balance_lines = [
            "Energy balance check (0 to t_final, validation aid):",
            f"  Using {v_source_note} for E_elec_in (includes the R_fe core-loss branch "
            f"current whenever iron loss is on).",
            f"  E_elec_in (electrical energy in)         = {E_elec_in:12.4f} J",
            f"  E_mech    (mechanical energy to shaft)   = {E_mech:12.4f} J",
            f"  E_cu      (copper loss energy)           = {E_cu:12.4f} J",
            f"  E_iron    (iron loss energy, Steinmetz)  = {E_iron:12.4f} J",
            f"  dE_mag    (stored dq coenergy change)    = {dE_mag:12.4f} J",
            f"  Accounted (E_mech+E_cu+E_iron+dE_mag)    = {E_accounted:12.4f} J",
            f"  Residual  (E_elec_in - Accounted)        = {E_residual:12.4f} J  ({E_residual_pct:+.2f}%)",
            f"  Verdict: {balance_verdict}",
        ]

        if self.include_iron_loss:
            energy_balance_lines.append(
                f"  (E_iron_electrical, the independent R_fe-branch estimate actually "
                f"drawn into E_elec_in above, = {E_iron_electrical:12.4f} J -- compare "
                f"to the Steinmetz E_iron above; they're two separate loss models, not "
                f"defined to match exactly.)"
            )

        energy_balance = {
            "E_elec_in_J": E_elec_in,
            "E_mech_J": E_mech,
            "E_cu_J": E_cu,
            "E_iron_J": E_iron,
            "E_iron_electrical_J": E_iron_electrical,
            "dE_mag_J": dE_mag,
            "E_residual_J": E_residual,
            "E_residual_pct": E_residual_pct,
            "voltage_source": v_source_note,
        }

        if stop_info["success"] and abs(t[-1] - self.tfinal) < 1e-9:
            sim_status_line = "Simulation completed successfully."
        elif stop_info["status"] == 1:
            sim_status_line = f"Simulation terminated by event at t = {t[-1]:.6f} s."
        else:
            sim_status_line = (
                f"Simulation stopped before tfinal. "
                f"Last time = {t[-1]:.6f} s. "
                f"Solver status = {stop_info['status']}. "
                f"Message: {stop_info['message']}"
            )

        summary_lines = [
            sim_status_line,
            "",
            f"Machine type: {self.machine_type}",
            f"Drive mode: {self.drive_mode}",
            f"Frame mode: {self.frame_mode}",
            # f"Region: {self.region}",
            f"Requested tfinal: {self.tfinal} s",
            f"Returned final time: {t[-1]:.6f} s",
            f"Solver success: {stop_info['success']}",
            f"Solver status: {stop_info['status']}",
            f"Solver message: {stop_info['message']}",
            # f"tfinal: {self.tfinal} s",
            f"Vdc: {self.Vdc} V",
            f"Imax: {self.Imax} A",
            f"poles: {self.poles}",
            f"f_pwm: {self.f_pwm} Hz",
            f"Peak speed: {np.max(np.abs(speed_rpm)):.3f} rpm",
            f"Peak torque: {np.max(np.abs(Torque)):.3f} Nm",
            # temp ignored
            f"Peak stator temp: {max_Ts:.3f} °C",
            f"Peak magnet temp: {max_Tm:.3f} °C",
            "",
            max_speed_message,
            "",
            # temp ignored
            # "Thermal / event messages:",
            # *event_messages,
            # *warning_messages,
            "Event messages:",
            *event_messages,
            "",
            *pi_mtpa_summary,
            "",
            *energy_balance_lines,
        ]

        return {
            "summary": "\n".join(summary_lines),
            # "figures": figures,
            "time": t,
            "energy_balance": energy_balance,
            "signals": {
                "t": t,
                "Vmax": self.Vmax,
                "i_d": i_d,
                "i_q": i_q,
                "omega_m": omega_m,
                "theta_m": theta_m,
                "Torque": Torque,
                "speed_rpm": speed_rpm,
                "speed_ref_rpm": speed_ref_rpm,
                # temp ignored
                "T_s": T_s,
                "T_m": T_m,
                "P_cu_arr": P_cu_arr,
                "P_iron_arr": P_iron_arr,
                "B_core_arr": B_core_arr,
                "id_ref_arr": id_ref_arr,
                "iq_ref_arr": iq_ref_arr,
                "iq_ref_pi_arr": iq_ref_pi_arr,
                "id_mtpa_arr": id_mtpa_arr,
                "iq_mtpa_arr": iq_mtpa_arr,
                "is_pi_arr": is_pi_arr,
                "is_mtpa_arr": is_mtpa_arr,
                "torque_cmd_arr": torque_cmd_arr,
                "vd_uns_arr": vd_uns_arr,
                "vq_uns_arr": vq_uns_arr,
                "vd_cmd_arr": vd_cmd_arr,
                "vq_cmd_arr": vq_cmd_arr,
                "v_mag_uns_arr": v_mag_uns_arr,
                "v_mag_cmd_arr": v_mag_cmd_arr,
                "omega_base_arr": omega_base_arr,
                "omega_max_arr": omega_max_arr,
                "max_speed_reached_arr": max_speed_reached_arr,
                "i_phase_a": i_phase_a,
                "i_phase_b": i_phase_b,
                "i_phase_c": i_phase_c,
                "t_zoom": t_zoom,
                "carrier_zoom": carrier_zoom,
                "ma_zoom": ma_zoom,
                "mb_zoom": mb_zoom,
                "mc_zoom": mc_zoom,
                "gA_up": gA_up,
                "gB_up": gB_up,
                "gC_up": gC_up,
                "gA_lo": gA_lo,
                "gB_lo": gB_lo,
                "gC_lo": gC_lo,
                "T_load_arr": T_load_arr,
                "v_vehicle": v_vehicle,
                "mass_arr": mass_arr,
                "slope_arr": slope_arr,
                "crr_arr": crr_arr,
                "va_ref_arr": va_ref_arr,
                "vb_ref_arr": vb_ref_arr,
                "vc_ref_arr": vc_ref_arr,
                "v_offset_arr": v_offset_arr,
                "van_ref_arr": van_ref_arr,
                "vbn_ref_arr": vbn_ref_arr,
                "vcn_ref_arr": vcn_ref_arr,
                "da_cmd_arr": da_cmd_arr,
                "db_cmd_arr": db_cmd_arr,
                "dc_cmd_arr": dc_cmd_arr,
                "da_eff_arr": da_eff_arr,
                "db_eff_arr": db_eff_arr,
                "dc_eff_arr": dc_eff_arr,
                "van_avg_arr": van_avg_arr,
                "vbn_avg_arr": vbn_avg_arr,
                "vcn_avg_arr": vcn_avg_arr,
                "va_avg_arr": va_avg_arr,
                "vb_avg_arr": vb_avg_arr,
                "vc_avg_arr": vc_avg_arr,
                # "sol_status": sol.status,
                # "sol_message": sol.message,
                "sol_success": stop_info["success"],
                "sol_status": stop_info["status"],
                "sol_message": stop_info["message"],
                "sol_tfinal_returned": t[-1],
                # "solver_breakpoints": self.get_schedule_breakpoints(),
                "event_messages": event_messages,
                # "warning_messages": warning_messages,
                # "sol_success": sol.success,
                # "sol_tfinal_returned": sol.t[-1],
                "omega_ref_f_arr": omega_ref_f_arr,
                "mass_f_arr": mass_f_arr,
                "slope_f_arr": slope_f_arr,
                "crr_f_arr": crr_f_arr,
                "u_d_uns_arr": u_d_uns_arr,
                "u_q_uns_arr": u_q_uns_arr,
                "iq_ref_uns_arr": iq_ref_uns_arr,
                "iq_ref_clamp_arr": iq_ref_clamp_arr,
                "ed_cross_arr": ed_cross_arr,
                "eq_backemf_arr": eq_backemf_arr,
                "epm_arr": epm_arr,
                "Ld_arr": Ld_arr,
                "Lq_arr": Lq_arr,
                "K_pd_arr": K_pd_arr,
                "K_pq_arr": K_pq_arr,
                "K_i_arr": K_i_arr,
                "K_ps_arr": K_ps_arr,
                "K_is_arr": K_is_arr,
                "id_grid_help": id_grid_help,
                "iq_grid_help": iq_grid_help,
                "Ld_curve": Ld_curve,
                "Lq_curve": Lq_curve,
                "Ld0": self.Ld0,
                "Lq0": self.Lq0,
                "Ld_inf": self.Ld_inf,
                "Lq_inf": self.Lq_inf,
                "I_Ld_sat": self.I_Ld_sat,
                "I_Lq_sat": self.I_Lq_sat,
                "vd_applied_arr": vd_applied_arr,
                "vq_applied_arr": vq_applied_arr,
                # "mtpa_torque_max": None if self.mtpa_lut is None else self.mtpa_lut.torque_max,
                # "mtpa_n_points": None if self.mtpa_lut is None else self.mtpa_lut.n_points,
                # "mtpa_id_samples": None if self.mtpa_lut is None else self.mtpa_lut.id_samples,
            },
        }
# =============================================================================================================
# without regime B when field weakening applied (anything below from here)
# =============================================================================================================
# from __future__ import annotations

# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from scipy.integrate import solve_ivp

# # numpy >= 2.0 renamed trapz to trapezoid; keep this working on either version.
# _trapz = getattr(np, "trapezoid", None) or np.trapz

# from pmsm_backend.config import PMSMConfig
# from pmsm_backend.mtpa import MtpaLut
# from pmsm_backend.transforms import (
#     T_matrix,
#     dq0_to_abc,
#     dq_voltage_limit,
# )
# from pmsm_backend.pwm import (
#     phase_refs_to_svpwm_pole_refs,
#     pole_refs_to_duties,
#     apply_deadtime_to_duties,
#     duties_to_average_pole_voltages,
#     pole_to_phase_voltages,
#     triangular_carrier_from_time,
#     make_center_aligned_gates_with_deadtime,
# )

# _MTPA_CACHE: Dict[tuple, Optional[MtpaLut]] = {}

# class PMSMSimulation:
#     def get_mtpa_cache_key(self) -> tuple:
#         """
#         Returns a cache key containing only the parameters that materially affect the MTPA LUT.
#         Rounded values avoid pointless rebuilds from tiny floating-point differences.
#         """
#         return (
#             self.machine_type,
#             int(self.polepairs),
#             round(float(self.Imax), 8),
#             round(float(self.L_ds), 10),
#             round(float(self.L_qs), 10),
#             round(float(self.Ld_inf), 10),
#             round(float(self.Lq_inf), 10),
#             round(float(self.I_Ld_sat), 8),
#             round(float(self.I_Lq_sat), 8),
#             round(float(self.phi_f_ref), 10),
#         )

#     def __init__(self, cfg: PMSMConfig):
#         self.cfg = cfg

#         self.machine_type = cfg.machine_type  # "ipmsm" or "spmsm"
#         # self.use_mtpa = cfg.use_mtpa
#         # self.use_field_weakening = cfg.use_field_weakening
#         self.use_mtpa = bool(cfg.use_mtpa)
#         self.use_field_weakening = bool(cfg.use_field_weakening)

#         # SPMSM: disable field weakening in this model
#         if self.machine_type == "spmsm":
#             self.use_field_weakening = False

#         # steps with leaner flow:
#         # self.tau_mass = 1.0      # s
#         # self.tau_slope = 0.5     # s
#         # self.tau_crr = 0.3       # s
#         # Smooth schedule transitions.
#         # Smaller tau = faster transition.
#             # tau = 0.50 s  -> slow smooth transition
#             # tau = 0.15 s  -> noticeably faster but still smooth
#             # tau = 0.05 s  -> very fast smooth transition
#             # tau = 0.01 s  -> almost step-like
#         self.tau_mass = 0.15     # s
#         self.tau_slope = 0.05    # s
#         self.tau_crr = 0.05      # s

#         self.P_rated = cfg.P_rated

#         # self.tau_omega_ref = 0.1 # s
#         # self.omega_ref_rate_up = 200.0   # rad/s^2
#         # self.omega_ref_rate_down = 300.0 # rad/s^2
#         self.tau_omega_ref = 0.005  # 0.02
#         self.omega_ref_rate_up = 1e4  # 1000.0
#         self.omega_ref_rate_down = 1e4  #1200.0

#         # --------------------------- User settings --------------------------------
#         # self.machine_type = cfg.machine_type  # "ipmsm" or "spmsm"
#         # TODO: synchronous frame for IPMSM needs to use FOC

#         # IMPORTANT:
#         #  frame_mode="rotor_foc" for drive_mode="closed_loop_foc"
#         #  frame_mode="stator" only for open-loop demonstrations

#         # Choose what is applied at the stator:
#         #  - "closed_loop_foc": apply dq voltages from PI current controllers (recommended)
#         #  - "open_loop_abc"  : apply imposed sinusoidal abc voltages (fundamentally unstable for PMSM)
#         self.drive_mode = cfg.drive_mode

#         # Choose dq reference frame USED FOR CONTROL + MODEL EQUATIONS BELOW:
#         #  - "rotor_foc" : theta_frame = theta_r_e (rotor electrical angle)  <-- this is what you want for FOC
#         #  - "stator"    : theta_frame = omega_e * t (for demonstrations; not phase-locked)
#         self.frame_mode = cfg.frame_mode

#         if self.drive_mode == "closed_loop_foc" and self.frame_mode != "rotor_foc":
#             raise ValueError("closed_loop_foc must use frame_mode='rotor_foc'")

#         # ------------------- ELECTRICAL + MECHANICAL PARAMETERS (EDIT) -----------------
#         # # DC bus / inverter limit (simple SVPWM dq magnitude limit)
#         # Vdc = 300.0
#         # Vmax = Vdc / np.sqrt(3)

#         # For open-loop abc voltages:
#         '''
#         V_LL = phase to phase
#              = line to line inverter's usual limit, voltmeter across 2 phases
#              = This is what the inverter directly creates at its terminals
#         V_phase = This is the voltage that actually appears across one stator winding
#                 = Exists only in star (Y) connection
#         RMS is the effective DC-equivalent value of an AC signal in terms of power and heating.
#         '''
#         # self.region = cfg.region  # "EU" or "US"

#         # if self.region == "EU":
#         #     # 400 V-class industrial drive
#         #     self.f = 50.0 if abs(cfg.f - 50.0) < 1e-15 else cfg.f
#         #     self.V_LL = 400.0 if abs(cfg.V_LL - 400.0) < 1e-15 else cfg.V_LL
#         #     self.Vdc = cfg.Vdc  # inverter DC bus (typical for 400 V drives)
#         # elif self.region == "US":
#         #     # 208 V-class low-voltage drive
#         #     self.f = 60.0 if abs(cfg.f - 50.0) < 1e-15 else cfg.f
#         #     self.V_LL = 208.0 if abs(cfg.V_LL - 400.0) < 1e-15 else cfg.V_LL
#         #     self.Vdc = cfg.Vdc  # inverter DC bus (typical for 208 V drives)
#         # else:
#         self.f = cfg.f
#         self.V_LL = cfg.V_LL
#         self.Vdc = cfg.Vdc

#         # inverter voltage capability (SVPWM)
#         # also called base speed (omega_base)
#         self.Vmax = self.Vdc / np.sqrt(3)  # max dq voltage magnitude (peak)

#         # motor phase voltages
#         self.V_phase = self.V_LL / np.sqrt(3)
#         self.V_m = np.sqrt(2) * self.V_phase  # peak phase voltage

#         # electrical frequency
#         self.omega_e = 2.0 * np.pi * self.f

#         # leakage + saliency “construction”
#         self.L_ls = cfg.L_ls
#         self.L_A = cfg.L_A
#         self.L_B = cfg.L_B

#         # compute L_d and L_q as for IPMSM stator speed:
#         # between eq. 4.109 and 4.110 page 199  
#         '''
#         L_ls: stator leakage inductance
#             : some stator-produced flux goes through the main magnetic path and contributes to torque/energy conversion
#             : some flux “leaks” locally around slots, teeth, end windings, etc.
#             : that leakage part behaves like an extra inductance in series with the stator winding
#         L_A, L_B: terms describing the rotor-position-dependent saliency > after transforming the abc 
#           inductance matrix into dq coordinates, those combinations collapse into constant dq inductances
#         L_A: how much useful main flux the stator can establish through the air gap and rotor iron
#              the general strength of magnetic coupling in the machine
#         L_B: how much the rotor geometry causes the inductance to change as the rotor turns
#         '''
#         self.L_ds = self.L_ls + 1.5 * (self.L_A - self.L_B)
#         self.L_qs = self.L_ls + 1.5 * (self.L_A + self.L_B)

#         # SPMSM special case: cylindrical rotor => Ld = Lq
#         if self.machine_type == "spmsm":
#             self.L_qs = self.L_ds

#         ######################################################################################################################################################
#         # MTPA LUT
#         # They are the physical parameters of the inductance-vs-current
#         # approximation and must come from startup motor data,
#         # FEM, measurement, or datasheet fitting.
#         #
#         # Low-current values already come from your L_ds and L_qs.
#         # High-current values are the inductances near the current limit.
#         self.Ld0 = self.L_ds
#         self.Lq0 = self.L_qs

#         self.I_rated = cfg.I_rated  # motor rated current (typically the rated phase current, in amps)

#         self.Ld_inf = cfg.Ld_inf_factor * self.Ld0  # strong d-axis saturation inductance; typical range is about 30% to 70% of Ld0
#         self.Lq_inf = cfg.Lq_inf_factor * self.Lq0  # strong q-axis saturation inductance; typical range is about 40% to 80% of Lq0

#         self.I_Ld_sat = cfg.I_Ld_sat_factor * self.I_rated  # d-axis saturation current scale; typical range is about 0.5 to 1.5 × rated phase current
#         self.I_Lq_sat = cfg.I_Lq_sat_factor * self.I_rated  # q-axis saturation current scale; typical range is about 0.5 to 2.0 × rated phase current
#         # If Ld_inf or Lq_inf is set too low, the model may make the motor appear
#         # unrealistically saturated at high current. If I_L*_sat is set too small,
#         # the inductance drops too early; if set too large, the inductance curve
#         # stays too flat.
#         ######################################################################################################################################################

#         # IMPORTANT:
#         #  The magnetic flux varies considerably with the operating temperature.
#         # phi_f = 0.9  # Wb (example) # turning it into temperature dependant

#         # Stator resistance
#         # R_s = 2.0  # turning it into temperature dependant

#         # temperature dependant variables:
#         self.T_ref = cfg.T_ref  # degC
#         self.alpha_cu = cfg.alpha_cu  # 1/degC
#         self.alpha_phi = cfg.alpha_phi  # 1/degC  (NdFeB approx from your note)

#         # TODO: C_th_s and C_th_m??
#         self.C_th_s = cfg.C_th_s
#         self.C_th_m = cfg.C_th_m
#         self.R_th_sa = cfg.R_th_sa
#         self.R_th_sm = cfg.R_th_sm
#         self.R_th_ma = cfg.R_th_ma
#         self.T_amb = cfg.T_amb

#         # Iron (core) loss model — Steinmetz-style, from Mi/Slemon/Bonert
#         # (IEEE Trans. Ind. Appl., 2003):
#         #   p_iron = k_h*B^beta*omega_s + k_e*B^2*omega_s^2   [W/m^3]
#         # Modeled as an extra heat source feeding the stator thermal node.
#         self.include_iron_loss = cfg.include_iron_loss
#         self.k_h_iron = cfg.k_h_iron
#         self.k_e_iron = cfg.k_e_iron
#         self.beta_iron = cfg.beta_iron
#         self.core_volume_m3 = cfg.core_volume_m3
#         self.B_iron_rated = cfg.B_iron_rated

#         # ---- Iron-loss electrical branch (core-loss / "R_fe" current) -----------------
#         # P_iron_arr above (Steinmetz) only heats the stator thermally -- it never draws
#         # any extra current from the supply, so E_elec_in (computed purely from the
#         # actually-solved v_d,i_d,v_q,i_q) has nothing in it to pay for that loss. That
#         # makes the energy-balance residual go strongly negative whenever iron loss is
#         # on (see energy-balance section below).
#         #
#         # The physically correct fix is the classical iron-loss equivalent-circuit model:
#         # a core-loss resistance R_fe in parallel with the back-EMF branch, so the
#         # terminal current actually drawn is i_d + i_Fe_d, i_q + i_Fe_q, where i_Fe is
#         # the current that flows into R_fe (Ohm's law: i_Fe = E_backemf / R_fe).
#         #
#         # R_fe is calibrated ONCE here from the rated operating point (B_iron_rated at
#         # the rated electrical frequency cfg.f), using the same Steinmetz formula, then
#         # held FIXED for the whole run. This is deliberate: if R_fe were instead
#         # recomputed every step to force-match the instantaneous Steinmetz P_iron, the
#         # resulting i_Fe would just be P_iron/E_backemf by construction, and the
#         # energy-balance residual would become tautologically ~0% -- true by algebra,
#         # not because anything was actually verified. With a FIXED R_fe, the resulting
#         # electrical draw (E_iron_electrical, see energy-balance section) generally will
#         # NOT exactly equal the Steinmetz E_iron away from rated conditions -- the two
#         # stay independent estimates of the same physical loss, so the residual is still
#         # a meaningful check, not a guaranteed zero.
#         #
#         # IMPORTANT CAVEAT -- this is computed in POSTPROCESSING, after solve_ivp has
#         # already finished solving for i_d(t), i_q(t) (see the postprocessing loop
#         # below, where i_Fe_d_arr/i_Fe_q_arr are filled in). i_Fe never feeds back into
#         # the ODE, the current-loop PI controller, di_d/dt, di_q/dt, or torque -- those
#         # are all computed exactly as if iron loss did not exist. i_Fe only gets added
#         # onto the terminal current afterward, purely to make E_elec_in (the energy-
#         # balance report) more honest. This is a one-way/decoupled approximation, not a
#         # full resimulation of a machine with iron loss -- worth being explicit about,
#         # since it differs from how this plays out on real hardware:
#         #
#         # In a real drive, the phase current SENSOR physically measures the TOTAL
#         # current in the winding -- there is no way to separate "the part going to
#         # torque" from "the part going to core loss" with a single current sensor;
#         # they are the same physical current. So the current-loop feedback automatically
#         # sees and reacts to whatever extra current a real iron-loss branch adds, purely
#         # as a side effect of ordinary closed-loop control -- the integrator drives
#         # MEASURED current to the commanded reference regardless of what's contributing
#         # to that measured current. No special iron-loss term needs to be designed into
#         # the PI gains for this to work; the loop's normal disturbance-rejection handles
#         # it, the same way it implicitly handles other things not present in the
#         # standard Kp=L*omega_cc / Ki=R*omega_cc gain formulas either (temperature drift
#         # in R_s, saturation-dependent L, manufacturing tolerance, etc.).
#         #
#         # That's exactly what's different here: i_Fe_d_arr/i_Fe_q_arr are invisible to
#         # everything in this simulation, including K_pd_now/K_pq_now and the current-
#         # loop error terms (err_d, err_q) below -- nothing "sees" i_Fe, so nothing
#         # reacts to it, unlike a real current sensor which would.
#         #
#         # Where iron loss DOES get explicitly modeled in real control design is one
#         # level up from the PI gains: efficiency-focused drives (common in EV traction)
#         # use "loss-minimizing control" (a.k.a. maximum-efficiency control), where the
#         # id_ref/iq_ref REFERENCE split is computed to minimize total loss at a given
#         # torque and speed -- and that optimization does explicitly use a core-loss-
#         # resistance model much like R_fe_iron here. But that changes what current is
#         # COMMANDED, not the PI gains themselves -- the low-level tracking loop
#         # underneath still just uses plain R_s/L-based tuning, exactly as this
#         # simulation's current-loop PI does today.
#         if self.include_iron_loss:
#             _lambda_rated_for_Rfe = float(np.hypot(cfg.phi_f_ref, self.Lq0 * cfg.I_rated))
#             _omega_s_rated = 2.0 * np.pi * cfg.f
#             _E_rated = _omega_s_rated * _lambda_rated_for_Rfe
#             _P_iron_rated = self.core_volume_m3 * (
#                 self.k_h_iron * (self.B_iron_rated ** self.beta_iron) * _omega_s_rated
#                 + self.k_e_iron * (self.B_iron_rated ** 2) * (_omega_s_rated ** 2)
#             )
#             self.R_fe_iron = 1.5 * _E_rated ** 2 / max(_P_iron_rated, 1e-9)
#         else:
#             self.R_fe_iron = float("inf")
#         # NOTE:
#         #  So in model:
#         #  current → copper loss → stator heats up
#         #  stator heats magnets
#         #  magnet temperature changes 𝜙_f(T_m)

#         self.R_s_ref = cfg.R_s_ref
#         self.phi_f_ref = cfg.phi_f_ref

#         # ------------------- thermal protection / fault thresholds -------------------
#         # NOTE:
#         #  These are example values only. In real work, use datasheet values.
#         self.T_warn_stator = cfg.T_warn_stator
#         self.T_fault_stator = cfg.T_fault_stator

#         self.T_warn_magnet = cfg.T_warn_magnet
#         self.T_fault_magnet = cfg.T_fault_magnet

#         self.T_demag = cfg.T_demag
#         self.phi_f_min = cfg.phi_f_min_factor * self.phi_f_ref  # do not let flux become negative/unphysical

#         # mechanical / poles
#         # TODO: to be option for user
#         self.poles = cfg.poles
#         self.polepairs = self.poles // 2
#         # J = 0.01
#         # B_visc = 0.0005
#         # load_torque = 1.0

#         # ==============================================================================================================================================
#         # ------------------- VEHICLE / LOAD PARAMETERS -------------------
#         self.g = 9.81
#         self.rho_air = cfg.rho_air  # kg/m^3

#         # Drivetrain
#         self.gear_ratio = cfg.gear_ratio  # motor speed / wheel speed
#         self.drivetrain_eff = cfg.drivetrain_eff
#         self.wheel_radius = cfg.wheel_radius  # m

#         # Vehicle / road
#         self.Cd = cfg.Cd
#         self.A_front = cfg.A_front
#         self.b_vehicle = cfg.b_vehicle

#         # Default values (can change by time using schedules below)
#         # self.mass_default = cfg.mass_default
#         # self.slope_default_deg = cfg.slope_default_deg
#         # self.Crr_default = cfg.Crr_default

#         # =============================================================================
#         # INVERTER / PWM SETTINGS
#         # =============================================================================
#         self.use_svpwm = cfg.use_svpwm
#         self.include_deadtime = cfg.include_deadtime

#         self.f_pwm = cfg.f_pwm
#         self.T_pwm = 1.0 / self.f_pwm

#         self.deadtime = cfg.deadtime
#         # TODO: input on from where to show the 20 PWM switches
#         self.plot_pwm_zoom_start = cfg.plot_pwm_zoom_start
#         self.plot_pwm_num_periods = cfg.plot_pwm_num_periods
#         self.plot_pwm_zoom_span = self.plot_pwm_num_periods * self.T_pwm
#         self.plot_pwm_zoom_end = self.plot_pwm_zoom_start + self.plot_pwm_zoom_span
#         self.pwm_plot_samples = cfg.pwm_plot_samples

#         # ------------------- STEP SCHEDULES -------------------
#         # simulation time
#         self.t0 = 0.0
#         self.tfinal = cfg.tfinal
#         # TODO: SET ALSO THE NUM MAYBE
#         # self.t_eval = np.linspace(self.t0, self.tfinal, int(self.tfinal * 10000))
#         n_eval = max(2, int(self.tfinal * 5000))
#         self.t_eval = np.linspace(self.t0, self.tfinal, n_eval)

#         # Speed reference schedule (mechanical rad/s)
#         self.t_steps = cfg.t_steps
#         self.omega_steps = cfg.omega_steps

#         self.mass_step_times = cfg.mass_step_times
#         self.mass_step_values = cfg.mass_step_values

#         self.slope_step_times = cfg.slope_step_times
#         self.slope_step_values_deg = cfg.slope_step_values_deg

#         self.crr_step_times = cfg.crr_step_times
#         self.crr_step_values = cfg.crr_step_values

#         # NOTE:
#         #  start on flat asphalt
#         #  then add passengers/cargo
#         #  then go uphill onto rough/sandy surface

#         self.J_motor = cfg.J_motor
#         # TODO: choose nominal or worst-case design mass
#         # if J_total gets larger:
#         # acceleration gets smaller
#         # speed response becomes slower
#         # same torque produces less speed change
#         # So if your controller was designed for a smaller J, the real
#         # system may respond more slowly than expected.
#         # self.J_ctrl = self.J_motor + self.mass_default * self.wheel_radius ** 2 / self.gear_ratio ** 2

#         # self._seg_mass = self.mass_default
#         # self._seg_slope_deg = self.slope_default_deg
#         # self._seg_crr = self.Crr_default
#         # self._seg_omega_ref = 0.0
#         # self._seg_J_total = self.J_motor + self.mass_default * self.wheel_radius ** 2 / (self.gear_ratio ** 2)
#         # =============================================================================
#         # CONTROL BANDWIDTH + GAINS
#         # =============================================================================
#         # Current loop bandwidth selection (rule-of-thumb)
#         # page 71 - 2.6.1.1 Selection of the bandwidth for current control
#         # IMPORTANT:
#         #  If the current is sampled twice every switching period, as a rule of
#         #  thumb, the maximum available bandwidth can be up to 1/10 of the
#         #  switching frequency. On the other hand, if it is sampled once every
#         #  switching period, the maximum available bandwidth can be up to 1/20 of
#         #  switching frequency. For this case, it is desirable to restrict
#         #  the bandwidth to 1/25 of the sampling frequency
#         self.omega_cc = 2.0 * np.pi * (self.f_pwm / 25.0)

#         # IMPORTANT:
#         #  controllers:
#         #  - current control bandwidth ωcc should be at least five times
#         #    wider than the speed control bandwidth ωcs
#         # Speed loop bandwidth
#         # omega_cs = omega_cc / 5.0
#         self.omega_cs = self.omega_cc / 5.0

#         # PI current regulator (eq. 6.28/6.29):
#         #   K_pd = L_d*ω_c,  K_pq = L_q*ω_c
#         #   K_i  = R_s*ω_c
#         self.K_pd = self.L_ds * self.omega_cc
#         self.K_pq = self.L_qs * self.omega_cc
#         self.K_i = self.R_s_ref * self.omega_cc

#         # Anti-windup gain (note: K_a = 1/K_p)
#         self.K_aw_d = self.K_i / max(self.K_pd, 1e-12)
#         self.K_aw_q = self.K_i / max(self.K_pq, 1e-12)

#         # IMPORTANT:
#         #  page 73 - 2.6.2 ANTI-WINDUP CONTROLLER
#         #  When we saturate the commanded voltage, the PI “thinks” it can apply
#         #  more voltage than the inverter allows. The integrator would keep
#         #  accumulating error (“wind up”) and then, when saturation clears, the
#         #  integrator is huge → overshoot.
#         #  u = Kp*e + x_int (Kp*e is proportional term, u is controller output
#         #                    and x_int is the accumulated integral action)
#         #                    It stores past error and provides steady-state accuracy
#         #  xint_dot = K_i*e + K_aw(u_sat−u)
#         #   >> the delta(u) = measures how much the controller is asking for more
#         #                     than the actuator can deliver
#         #                   = delta is negative
#         #   >> any excess output during saturation is strongly influenced by 𝐾_𝑝.
#         #      To “undo” that excess at a comparable rate: The integrator
#         #      correction should scale inversely with 𝐾_𝑝.

#         # Speed PI from “DC motor-like” procedure:
#         # For PMSM a common torque constant near id≈0 is:
#         # K_T ≈ 1.5*polepairs*phi_f
#         # self.K_T = 1.5 * self.polepairs * self.phi_f_ref
#         # self.K_ps = self.J_ctrl * self.omega_cs / max(self.K_T, 1e-12)
#         # self.K_is = self.K_ps * self.omega_cs / 5.0
#         # # K_aw_s = 1.0 / max(K_ps, 1e-12)
#         # self.K_aw_s = self.K_is / max(self.K_ps, 1e-12)
#         # TODO: maybe implement combination of PI and IP control (pages 78-81)

#         # IMPORTANT:
#         #  Current limits for speed loop output (iq_ref clamp)
#         #  Given on the motor datasheet as rated current or continuous current
#         #    - What sets it physically
#         #    - Copper loss I^2*R
#         #    - Thermal path from windings → stator → housing → air
#         #  If you exceed this current continuously, the motor will overheat
#         #  Inverter current limit → hard electrical ceiling
#         #  - Maximum current the inverter can physically deliver
#         #  Limited by:
#         #    - Semiconductor peak current rating
#         #    - DC bus voltage
#         #    - Gate driver protection
#         #    - Current sensor range
#         #  examples:
#         #    - Small servo (400 W) >> 2.5 A
#         #    - Industrial servo (3 kW) >> 7 A
#         #    - Large PMSM (20 kW) >> 35 A
#         #  there can be 2 limits > long term and short term
#         self.Imax = cfg.Imax

#         #########################################################################################################################################################
#         # MTPA LUT - beg
#         #########################################################################################################################################################
#         self.mtpa_model_params = {
#             "Ld_inf": self.Ld_inf,  # measured / FEM / identified at high current
#             "Lq_inf": self.Lq_inf,
#             "I_Ld_sat": self.I_Ld_sat,
#             "I_Lq_sat": self.I_Lq_sat,
#         }

#         self.mtpa_lut = self.build_mtpa_object(
#             machine_type=self.machine_type,
#             polepairs=self.polepairs,
#             Imax=self.Imax,
#             L_ds=self.L_ds,
#             L_qs=self.L_qs,
#             mtpa_model_params=self.mtpa_model_params,
#             phi_f_initial=self.phi_f_ref,
#         )
#         #########################################################################################################################################################
#         # MTPA LUT - end
#         #########################################################################################################################################################

#         self.FW_EPS = 1e-12
#         # It’s a tiny number used as a safety threshold to avoid:
#         # division by zero
#         # square root of negative numbers (due to floating-point noise)
#         # unstable behavior at very low speeds
#         self.MAX_SPEED_REACHED_TOL = 0.995

#         self.rtol = cfg.rtol
#         self.atol = cfg.atol
#         self.solver_method = cfg.solver_method

#         # Storage used by thermal event functions
#         # self._fault_events_info: List[str] = []

#     # =============================================================================
#     # HELPER FUNCTIONS
#     # =============================================================================
#     # def set_segment_constants(self, t_segment_start: float) -> None:
#     #     """
#     #     Freeze all schedule-driven values for the current solver segment.
#     #     Since solve_piecewise splits exactly at discontinuities, these values
#     #     remain constant inside the segment.
#     #     """
#     #     self._seg_mass = self.piecewise_step(
#     #         t_segment_start, self.mass_step_times, self.mass_step_values
#     #     )
#     #     self._seg_slope_deg = self.piecewise_step(
#     #         t_segment_start, self.slope_step_times, self.slope_step_values_deg
#     #     )
#     #     self._seg_crr = self.piecewise_step(
#     #         t_segment_start, self.crr_step_times, self.crr_step_values
#     #     )
#     #     self._seg_omega_ref = self.omega_ref(t_segment_start)

#     #     self._seg_J_total = (
#     #         self.J_motor
#     #         + self._seg_mass * self.wheel_radius ** 2 / (self.gear_ratio ** 2)
#     #     )
    
#     # def total_resisting_torque_segment(self, omega_m: float) -> float:
#     #     """
#     #     Resisting torque referred to the motor shaft for the current segment.
#     #     Uses frozen schedule values, so no schedule lookup is done inside the ODE.
#     #     """
#     #     slope_rad = np.deg2rad(self._seg_slope_deg)
#     #     v = (omega_m / self.gear_ratio) * self.wheel_radius

#     #     s_v = np.tanh(v / 0.05)

#     #     F_grade = self._seg_mass * self.g * np.sin(slope_rad)
#     #     F_roll = 
#     #         self._seg_crr * self._seg_mass * self.g * np.cos(slope_rad) * np.sign(s_v)
#     #         if abs(s_v) > 1e-9 else 0.0
#     #     )
#     #     F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v * abs(v)
#     #     F_visc = self.b_vehicle * v

#     #     F_total = F_grade + F_roll + F_aero + F_visc
#     #     T_wheel = F_total * self.wheel_radius
#     #     T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)

#     #     return T_motor_load

#     # def get_schedule_breakpoints(self) -> np.ndarray:
#     #     """
#     #     Return sorted unique time breakpoints where any schedule changes.
#     #     Includes t0 and tfinal.
#     #     """
#     #     all_breaks = (
#     #         [self.t0, self.tfinal]
#     #         + list(np.asarray(self.t_steps, dtype=float))
#     #         + list(np.asarray(self.mass_step_times, dtype=float))
#     #         + list(np.asarray(self.slope_step_times, dtype=float))
#     #         + list(np.asarray(self.crr_step_times, dtype=float))
#     #     )

#     #     # keep only times inside [t0, tfinal]
#     #     all_breaks = [tb for tb in all_breaks if self.t0 <= tb <= self.tfinal]

#     #     # sorted unique
#     #     return np.array(sorted(set(all_breaks)), dtype=float)
    
#     # def solve_piecewise(self, x0: np.ndarray):
#     #     """
#     #     Solve the ODE piecewise across all schedule discontinuities.
#     #     Returns:
#     #         t_all, y_all, segment_solutions, stop_info
#     #     """
#     #     breaks = self.get_schedule_breakpoints()

#     #     t_parts = []
#     #     y_parts = []
#     #     segment_solutions = []

#     #     x_init = x0.copy()

#     #     stop_info = {
#     #         "success": True,
#     #         "status": 0,
#     #         "message": "Completed successfully.",
#     #         "failed_segment": None,
#     #         "failed_time": None,
#     #         "t_events": [[], []],
#     #     }

#     #     for k in range(len(breaks) - 1):
#     #         t_start = float(breaks[k])
#     #         t_end = float(breaks[k + 1])

#     #         if t_end <= t_start:
#     #             continue
            
#     #         # self.set_segment_constants(t_start)

#     #         # restrict requested output points to this segment
#     #         seg_mask = (self.t_eval >= t_start) & (self.t_eval <= t_end)
#     #         t_eval_seg = self.t_eval[seg_mask]

#     #         # make sure segment end points are included
#     #         if t_eval_seg.size == 0 or abs(t_eval_seg[0] - t_start) > 1e-15:
#     #             t_eval_seg = np.insert(t_eval_seg, 0, t_start)
#     #         if abs(t_eval_seg[-1] - t_end) > 1e-15:
#     #             t_eval_seg = np.append(t_eval_seg, t_end)

#     #         def stator_fault_event(t, x):
#     #             return self.T_fault_stator - x[7]
#     #         stator_fault_event.terminal = True
#     #         stator_fault_event.direction = -1

#     #         def magnet_fault_event(t, x):
#     #             return self.T_fault_magnet - x[8]
#     #         magnet_fault_event.terminal = True
#     #         magnet_fault_event.direction = -1

#     #         sol_seg = solve_ivp(
#     #             self.motor_ode_pmsm_controlled,
#     #             (t_start, t_end),
#     #             x_init,
#     #             method=self.solver_method,
#     #             t_eval=t_eval_seg,
#     #             rtol=self.rtol,
#     #             atol=self.atol,
#     #             events=[stator_fault_event, magnet_fault_event],
#     #         )

#     #         segment_solutions.append(sol_seg)

#     #         # collect event times
#     #         for i_ev in range(2):
#     #             if len(sol_seg.t_events[i_ev]) > 0:
#     #                 stop_info["t_events"][i_ev].extend(sol_seg.t_events[i_ev].tolist())

#     #         # append segment data, avoiding duplicate boundary sample
#     #         if len(t_parts) == 0:
#     #             t_parts.append(sol_seg.t)
#     #             y_parts.append(sol_seg.y)
#     #         else:
#     #             # skip first point because it is the same as previous segment end
#     #             t_parts.append(sol_seg.t[1:])
#     #             y_parts.append(sol_seg.y[:, 1:])

#     #         if sol_seg.status == 1:
#     #             stop_info["success"] = False
#     #             stop_info["status"] = 1
#     #             stop_info["message"] = "Terminated by event."
#     #             stop_info["failed_segment"] = (t_start, t_end)
#     #             stop_info["failed_time"] = float(sol_seg.t[-1]) if sol_seg.t.size else t_start
#     #             break

#     #         if not sol_seg.success:
#     #             stop_info["success"] = False
#     #             stop_info["status"] = sol_seg.status
#     #             stop_info["message"] = sol_seg.message
#     #             stop_info["failed_segment"] = (t_start, t_end)
#     #             stop_info["failed_time"] = float(sol_seg.t[-1]) if sol_seg.t.size else t_start
#     #             break

#     #         # use final state of this segment as initial state of next one
#     #         x_init = sol_seg.y[:, -1].copy()

#     #     if len(t_parts) == 0:
#     #         raise RuntimeError("Piecewise solver produced no output.")

#     #     t_all = np.concatenate(t_parts)
#     #     y_all = np.concatenate(y_parts, axis=1)

#     #     return t_all, y_all, segment_solutions, stop_info
# #--------------------------------------------------------------------------------
#     def first_order_tracking(self, x: float, x_target: float, tau: float) -> float:
#         """
#         Compute the first-order tracking rate that moves x toward x_target.

#         A smaller tau makes the response faster, while a larger tau makes it slower.
#         A tiny lower bound is applied to tau to avoid division by zero.

#         Args:
#             x: Current value.
#             x_target: Desired target value.
#             tau: Time constant controlling tracking speed.

#         Returns:
#             The rate of change needed for x to approach x_target.
#         """
#         tau_eff = max(tau, 1e-9)
#         return (x_target - x) / tau_eff
    
#     def rate_limited_tracking(
#         self,
#         x: float,
#         x_target: float,
#         tau: float,
#         rate_up: float,
#         rate_down: float
#     ) -> float:
#         tau_eff = max(tau, 1e-9)
#         dx = (x_target - x) / tau_eff
#         return np.clip(dx, -abs(rate_down), abs(rate_up))
    
#     # def omega_ref_target(self, t: float) -> float:
#     #     return self.piecewise_step(t, self.t_steps, self.omega_steps)

#     def mass_target(self, t: float) -> float:
#         return self.piecewise_step(t, self.mass_step_times, self.mass_step_values)

#     def slope_target(self, t: float) -> float:
#         return self.piecewise_step(t, self.slope_step_times, self.slope_step_values_deg)

#     def crr_target(self, t: float) -> float:
#         return self.piecewise_step(t, self.crr_step_times, self.crr_step_values)
# #--------------------------------------------------------------------------------
#     def smooth_sign(self, v: float, v_eps: float = 0.05) -> float:
#         """
#         Smooth approximation of sign(v).
#         v_eps sets the transition width around zero speed.
#         """
#         return np.tanh(v / v_eps)

#     def piecewise_step(self, t: float, times: np.ndarray, values: np.ndarray) -> float:
#         if len(times) != len(values):
#             raise ValueError("times and values must have same length")
#         idx = np.searchsorted(times, t, side="right") - 1
#         if idx < 0:
#             return float(values[0])
#         return float(values[idx])

#     def vehicle_speed_from_motor_speed(self, omega_m: float) -> float:
#         '''
#         Force (N)
#            ↓ × wheel radius
#         Wheel torque (Nm)
#            ↓ ÷ gear ratio
#         Motor torque (Nm)
#         | Quantity | Symbol | Units      | Where it acts     |
#         | -------- | ------ | ---------- | ----------------- |
#         | Force    | (F)    | Newton (N) | linear motion     |
#         | Torque   | (T)    | Nm         | rotational motion |
#         '''
#         omega_wheel = omega_m / self.gear_ratio
#         v = omega_wheel * self.wheel_radius
#         return v

#     # def total_resisting_torque(self, t: float, omega_m: float) -> float:
#     #     """
#     #     Return resisting torque referred to the MOTOR shaft [Nm].
#     #     Inputs are intuitive vehicle parameters (kg, slope, road coefficient).
#     #     """
#     #     # Scheduled inputs
#     #     mass = self.piecewise_step(t, self.mass_step_times, self.mass_step_values)  # kg
#     #     slope_deg = self.piecewise_step(t, self.slope_step_times, self.slope_step_values_deg)  # deg
#     #     Crr = self.piecewise_step(t, self.crr_step_times, self.crr_step_values)  # -

#     #     slope_rad = np.deg2rad(slope_deg)
#     #     v = self.vehicle_speed_from_motor_speed(omega_m)  # m/s

#     #     s_v = self.smooth_sign(v, v_eps=0.05)

#     #     '''
#     #     Aerodynamic drag should always oppose velocity
#     #     Viscous loss should always oppose velocity
#     #     Grade force depends on road slope, not on velocity direction
#     #     '''

#     #     # Resistive forces at vehicle
#     #     F_grade = mass * self.g * np.sin(slope_rad)

#     #     # Rolling resistance usually opposes motion
#     #     # F_roll = Crr * mass * g * np.cos(slope_rad)
#     #     # F_roll = Crr * mass * self.g * np.cos(slope_rad) * np.sign(s_v) if abs(s_v) > 1e-9 else 0.0
#     #     F_roll = Crr * mass * self.g * np.cos(slope_rad) * s_v

#     #     # Aero drag opposes motion and should not help reverse motion
#     #     F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v * abs(v)

#     #     # Lumped viscous term
#     #     F_visc = self.b_vehicle * v

#     #     # Total road/load force
#     #     F_total = F_grade + F_roll + F_aero + F_visc

#     #     # Convert force -> wheel torque
#     #     T_wheel = F_total * self.wheel_radius

#     #     # Convert wheel torque -> motor shaft torque
#     #     # TODO: input
#     #     T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)

#     #     return T_motor_load

#     # eq. 1.69
#     # def equivalent_vehicle_inertia(self, t: float) -> float:
#     #     mass = self.piecewise_step(t, self.mass_step_times, self.mass_step_values)
#     #     return mass * self.wheel_radius ** 2 / (self.gear_ratio ** 2)

#     # def total_inertia(self, t: float) -> float:
#     #     return self.J_motor + self.equivalent_vehicle_inertia(t)

#     # source: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://eprints.whiterose.ac.uk/id/eprint/164680/1/System%20Identification%20AG_SX_finalV.pdf
#     # page 3 eq. (2)
#     def R_s_of_T(self, Ts: float) -> float:
#         '''
#         :param Ts: temperature of the stator
#         :return:
#         '''
#         return self.R_s_ref * (1.0 + self.alpha_cu * (Ts - self.T_ref))

#     # source?
#     def phi_f_of_T(self, Tm: np.ndarray | float) -> np.ndarray | float:
#         '''
#         :param Tm: PM temperature
#         :return:
#         '''
#         phi = self.phi_f_ref * (1.0 - self.alpha_phi * (Tm - self.T_ref))
#         return np.maximum(phi, self.phi_f_min)

#     # =============================================================================
#     # REFERENCES (speed + currents)
#     # =============================================================================
#     def omega_ref(self, t: float) -> float:
#         """
#         Piecewise-constant (step) speed reference.

#         t_steps[k]      : time at which step k becomes active
#         omega_steps[k]  : speed reference after that time

#         Arrays must have the same length.
#         """
#         if len(self.t_steps) != len(self.omega_steps):
#             raise ValueError(
#                 f"omega_ref error: t_steps (len={len(self.t_steps)}) "
#                 f"and omega_steps (len={len(self.omega_steps)}) must have the same length."
#             )

#         idx = np.searchsorted(self.t_steps, t, side="right") - 1

#         if idx < 0:
#             return float(self.omega_steps[0])

#         return float(self.omega_steps[idx])

#     #########################################################################################################################################################
#     # MTPA LUT - beg
#     #########################################################################################################################################################

#     def build_mtpa_object(
#         self,
#         machine_type: str,
#         polepairs: int,
#         Imax: float,
#         L_ds: float,
#         L_qs: float,
#         mtpa_model_params: Dict[str, float],
#         phi_f_initial: float,
#     ) -> Optional[MtpaLut]:
#         if machine_type == "spmsm":
#             return None

#         key = self.get_mtpa_cache_key()

#         if key in _MTPA_CACHE:
#             return _MTPA_CACHE[key]

#         # mtpa = MtpaLut(
#         #     polepairs=polepairs,
#         #     i_max=Imax,
#         #     n_points=121,
#         #     id_samples=200,
#         # )
#         mtpa = MtpaLut(
#             polepairs=polepairs,
#             i_max=Imax,

#             # None means automatic sizing
#             n_points=None,
#             id_samples=None,

#             # Tuning:
#             # 4 points/Nm  -> about 0.25 Nm torque spacing
#             # 10 points/A  -> about 0.1 A id spacing
#             torque_points_per_nm=4.0,
#             id_points_per_amp=10.0,

#             # Safety limits
#             min_torque_points=121,
#             max_torque_points=1201,
#             min_id_samples=200,
#             max_id_samples=2000,
#         )
        
#         mtpa.configure_inductance_model(
#             Ld0=L_ds,
#             Lq0=L_qs,
#             Ld_inf=mtpa_model_params["Ld_inf"],
#             Lq_inf=mtpa_model_params["Lq_inf"],
#             I_Ld_sat=mtpa_model_params["I_Ld_sat"],
#             I_Lq_sat=mtpa_model_params["I_Lq_sat"],
#         )
#         mtpa.build(phi_f_initial)

#         _MTPA_CACHE[key] = mtpa
#         return mtpa
  

#     # def pmsm_torque_constant(self, phi_f: float) -> float:
#     #     # SPMSM / id≈0 torque constant used only to convert the speed-loop
#     #     # output to a torque request before the MTPA lookup.
#     #     # Equation used:
#     #     #   Te ≈ 1.5*p*phi_f*iq
#     #     return 1.5 * self.polepairs * phi_f

#     def clamp_current_circle(self, i_d_cmd: float, i_q_cmd: float, Imax: float) -> Tuple[float, float]:
#         """
#         Enforce current magnitude <= Imax.
#         """
#         mag = np.hypot(i_d_cmd, i_q_cmd)
#         # Computes the magnitude (length) of the current vector:
#         # |I_s| = sqrt(i_d^2 + i_q^2)
#         # This is the Euclidean norm of the dq current vector.
#         if mag <= Imax or mag < self.FW_EPS:
#             return i_d_cmd, i_q_cmd
#         scale = Imax / mag
#         return scale * i_d_cmd, scale * i_q_cmd

#     def pmsm_torque_to_current_refs(self, machine_type: str, torque_ref: float, phi_f: float, mtpa_lut: Optional[MtpaLut]) -> Tuple[float, float]:
#         """
#         Current references before field weakening.

#         SPMSM:
#           Eq. (5.69), (5.70)
#             id*=0
#             iq*=Te/(1.5*p*phi_f)

#         IPMSM:
#           use the prebuilt MTPA LUT.
#         """
#         if machine_type == "spmsm" or not self.use_mtpa:
#             id_ref = 0.0
#             iq_ref = torque_ref / max(1.5 * self.polepairs * phi_f, 1e-12)
#             return self.clamp_current_circle(id_ref, iq_ref, self.Imax)

#         if mtpa_lut is None:
#             raise ValueError("IPMSM requires mtpa_lut.")
        
#         id_ref, iq_ref = mtpa_lut.lookup(torque_ref)
#         return self.clamp_current_circle(id_ref, iq_ref, self.Imax)
#     #########################################################################################################################################################
#     # MTPA LUT - end
#     #########################################################################################################################################################

#     # FIELD - WEAKENING BEGIN -------------------------------------------------------------------------------------------------------
#     # def ipmsm_base_speed(self, phi_f: float, i_d_mtpa: float, i_q_mtpa: float) -> float:
#     #     """
#     #     Eq. (8.31):
#     #         omega_r_base = V_s_max / sqrt((L_ds*i_ds + phi_f)^2 + (L_qs*i_qs)^2)
#     #     """
#     #     denom = np.sqrt((self.L_ds * i_d_mtpa + phi_f) ** 2 + (self.L_qs * i_q_mtpa) ** 2)
#     #     if denom < self.FW_EPS:
#     #         return np.inf
#     #     return self.Vmax / denom

#     def ipmsm_max_speed_finite(self, phi_f: float) -> Optional[float]:
#         """
#         Eq. (8.35), finite-speed case only:
#             omega_r_max = V_s_max / (phi_f - L_ds*I_s_max)
#         valid only if phi_f > L_ds*Imax
#         """
#         margin = phi_f - self.L_ds * self.Imax
#         if margin <= self.FW_EPS:
#             return None  # not finite-speed case
#         return self.Vmax / margin

#     def ipmsm_fw_currents_finite(self, omega_r_e: float, phi_f: float) -> Optional[Tuple[float, float]]:
#         """
#         Eqs. (8.36), (8.37) for IPMSM finite-speed FW.
#         """
#         # TODO: ASK - do we actually use I_s or I_max??
#         denom = (self.L_qs ** 2 - self.L_ds ** 2)
#         if abs(denom) < self.FW_EPS:
#             return None

#         if abs(omega_r_e) < self.FW_EPS:
#             return 0.0, 0.0
#         # <0 ==> reverse rotation

#         rad = ((self.L_ds * phi_f) ** 2
#                + denom * (phi_f ** 2 + (self.L_qs * self.Imax) ** 2 - (self.Vmax / abs(omega_r_e)) ** 2))

#         rad = max(rad, 0.0)

#         i_d_fw = (self.L_ds * phi_f - np.sqrt(rad)) / denom
#         i_d_fw = np.clip(i_d_fw, -self.Imax, 0.0)

#         i_q_fw_mag = np.sqrt(max(self.Imax ** 2 - i_d_fw ** 2, 0.0))
#         i_q_fw = np.sign(omega_r_e if abs(omega_r_e) > self.FW_EPS else 1.0) * i_q_fw_mag
#         # always produce torque in the same direction as rotation
#         # TODO: Ask if we want to take regenerative braking into consideration
#         return i_d_fw, i_q_fw

#     def spmsm_base_speed(self, phi_f: float) -> float:
#         """
#         Base-speed condition for SPMSM at point M (i_d*=0, i_q*=I_smax),
#         from the voltage limit with Eqs. (8.42), (8.43):

#             V_s_max^2 = omega_r^2 * [ (L_s I_s_max)^2 + phi_f^2 ]

#         therefore
#             omega_r_base = V_s_max / sqrt(phi_f^2 + (L_s I_s_max)^2)
#         """
#         denom = np.sqrt(phi_f ** 2 + (self.L_ds * self.Imax) ** 2)
#         if denom < self.FW_EPS:
#             return np.inf
#         return self.Vmax / denom

#     def spmsm_max_speed_finite(self, phi_f: float) -> Optional[float]:
#         """
#         Finite-speed case for SPMSM:
#             omega_r_max = V_s_max / (phi_f - L_s*I_s_max)
#         valid only if phi_f > L_s*I_s_max
#         """
#         margin = phi_f - self.L_ds * self.Imax
#         if margin <= self.FW_EPS:
#             return None
#         return self.Vmax / margin
#     # TODO: combine with ipmsm >> it is the same

#     def spmsm_fw_currents_finite(self, omega_r_e: float, phi_f: float) -> Tuple[float, float]:
#         """
#         Eqs. (8.46), (8.47) for SPMSM finite-speed FW.
#         """
#         if abs(omega_r_e) < self.FW_EPS:
#             return 0.0, 0.0

#         denom = 2.0 * self.L_ds * phi_f
#         if abs(denom) < self.FW_EPS:
#             return 0.0, 0.0

#         i_d_fw = (((self.Vmax / abs(omega_r_e)) ** 2 - phi_f ** 2 - (self.L_ds * self.Imax) ** 2) / denom)
#         i_d_fw = np.clip(i_d_fw, -self.Imax, 0.0)

#         i_q_fw_mag = np.sqrt(max(self.Imax ** 2 - i_d_fw ** 2, 0.0))
#         i_q_fw = np.sign(omega_r_e if abs(omega_r_e) > self.FW_EPS else 1.0) * i_q_fw_mag

#         return i_d_fw, i_q_fw

#     def motor_inductances(self, machine_type: str, i_d: float, i_q: float, mtpa_lut: Optional[MtpaLut]) -> Tuple[float, float]:
#         """
#         Current-dependent inductances used by controller and ODE.

#         SPMSM:
#             Ld = Lq = constant cylindrical-rotor value

#         IPMSM:
#             use the same saturation-law model as in the MTPA LUT
#         """
#         if machine_type == "spmsm" or mtpa_lut is None:
#             return self.L_ds, self.L_qs

#         return mtpa_lut.ld(i_d), mtpa_lut.lq(i_q)


#     def current_references_with_fw(self, machine_type, omega_m, torque_cmd, phi_f, mtpa_lut):
#         """
#         Returns:
#             id_ref, iq_ref, omega_base, omega_max, max_speed_reached_now

#         Below base speed:
#           - SPMSM: id*=0, iq*=Te/(1.5*p*phi_f)
#           - IPMSM: MTPA LUT

#         Above base speed:
#           - use existing finite-speed field-weakening equations
#         """
#         omega_r_e = abs(self.polepairs * omega_m)

#         # ---------- base current references ----------
#         if machine_type == "spmsm":
#             id_base = 0.0
#             iq_base = torque_cmd / max(1.5 * self.polepairs * phi_f, 1e-12)
#             id_base, iq_base = self.clamp_current_circle(id_base, iq_base, self.Imax)

#             omega_base = self.spmsm_base_speed(phi_f)
#             omega_max = self.spmsm_max_speed_finite(phi_f)
#             if omega_max is None:
#                 omega_max = np.inf

#         elif machine_type == "ipmsm":
#             if self.use_mtpa and mtpa_lut is not None:
#                 id_base, iq_base = mtpa_lut.lookup(torque_cmd)
#             else:
#                 id_base = 0.0
#                 iq_base = torque_cmd / max(1.5 * self.polepairs * phi_f, 1e-12)

#             id_base, iq_base = self.clamp_current_circle(id_base, iq_base, self.Imax)

#             """
#             Eq. (8.31):
#                 omega_r_base = V_s_max / sqrt((L_ds*i_ds + phi_f)^2 + (L_qs*i_qs)^2)
#             """
#             Ld_base, Lq_base = self.motor_inductances(machine_type, id_base, iq_base, mtpa_lut)
#             denom = np.sqrt((Ld_base * id_base + phi_f) ** 2 + (Lq_base * iq_base) ** 2)
#             omega_base = np.inf if denom < self.FW_EPS else self.Vmax / denom

#             omega_max = self.ipmsm_max_speed_finite(phi_f)
#             if omega_max is None:
#                 omega_max = np.inf

#         else:
#             raise ValueError(f"Unknown machine_type: {machine_type}")

#         # ---------- no field weakening ----------
#         if (not self.use_field_weakening) or machine_type == "spmsm":
#             return id_base, iq_base, omega_base, omega_max, False

#         # ---------- field weakening enabled ----------
#         if omega_r_e >= omega_max:
#             return -self.Imax, 0.0, omega_base, omega_max, True

#         if omega_r_e <= omega_base:
#             return id_base, iq_base, omega_base, omega_max, False

#         if machine_type == "spmsm":
#             i_d_fw, i_q_fw_mag = self.spmsm_fw_currents_finite(omega_r_e, phi_f)
#         else:
#             fw = self.ipmsm_fw_currents_finite(omega_r_e, phi_f)
#             if fw is None:
#                 return id_base, iq_base, omega_base, omega_max, False
#             i_d_fw, i_q_fw_mag = fw

#         i_q_fw = np.sign(torque_cmd) * abs(i_q_fw_mag)
#         i_d_fw, i_q_fw = self.clamp_current_circle(i_d_fw, i_q_fw, self.Imax)
#         return i_d_fw, i_q_fw, omega_base, omega_max, False

#     # def torque_from_iq_command(self, machine_type: str, iq_cmd: float, phi_f: float) -> float:
#     #     """
#     #     Converts the existing speed-loop q-axis current command into a torque command.

#     #     Equation used:
#     #         Te* ≈ 1.5 * p * phi_f * iq*
#     #     This preserves the original speed-PI tuning.
#     #     """
#     #     return 1.5 * self.polepairs * phi_f * iq_cmd

#     def torque_from_iq_command(self, machine_type: str, iq_cmd: float, phi_f: float) -> float:
#         """
#         Convert the speed-loop q-axis current command into a torque command.

#         SPMSM:
#             Te* = 1.5 * p * phi_f * iq*

#         IPMSM:
#             First form a provisional torque using the same expression above,
#             then refine it using the MTPA LUT and the full saliency torque law:
#                 Te = 1.5 * p * (phi_f * iq + (Ld - Lq) * id * iq)

#         This keeps the original speed-PI tuning nearly unchanged while giving a
#         better torque command for IPMSM.
#         """
#         # Base conversion that is exact for SPMSM and still a good first guess for IPMSM
#         torque_cmd = 1.5 * self.polepairs * phi_f * iq_cmd

#         # SPMSM: no reluctance torque term
#         if machine_type == "spmsm":
#             return torque_cmd

#         # IPMSM without MTPA/LUT: nothing better can be inferred from iq_cmd alone
#         if not self.use_mtpa or self.mtpa_lut is None:
#             '''
#             if MTPA is disabled, or
#             if the IPMSM MTPA lookup table does not exist,
#             '''
#             return torque_cmd

#         # One-step refinement using the actual MTPA operating point
#         id_est, iq_est = self.mtpa_lut.lookup(torque_cmd)
#         Ld_est, Lq_est = self.motor_inductances(machine_type, id_est, iq_est, self.mtpa_lut)

#         return 1.5 * self.polepairs * (
#             phi_f * iq_est + (Ld_est - Lq_est) * id_est * iq_est
#         )

#     # IMPORTANT:
#     #  Why do we not weaken the field in SPMSM:
#     #  Because a surface-mounted PMSM (SPMSM) has very low inductance, so
#     #  it can’t generate enough opposing magnetic field to safely or
#     #  efficiently weaken the permanent-magnet flux at high speed.
#     #  - SPMSMs have magnets on the surface → air-gap is large → inductance is small.
#     #  - Small inductance = d-axis current produces very little flux reduction.
#     #  - Demagnetization risk
#     #    - Strong negative d-axis current can partially demagnetize the
#     #      surface magnets, permanently reducing torque.
#     #  Why IPMSMs are more resistant:
#     #  - Magnets are buried inside the rotor
#     #    - They are shielded by rotor iron, which reduces the demagnetizing
#     #      field seen by the magnets.
#     #  - Higher d-axis inductance (Ld)
#     #    - You need much less neg. d-axis current to weaken the air-gap flux.
#     #    - That means lower demagnetizing stress on the magnets.
#     #  - Saliency helps
#     #    - Torque can come from reluctance torque, not just magnet torque,
#     #      so extreme negative d-axis current is unnecessary.

#     def dq_cmd_to_inverter_output(self, v_d_cmd: float, v_q_cmd: float, theta_frame: float, i_d: float, i_q: float, Vdc: float, T_pwm: float, deadtime: float, use_deadtime: bool = True) -> Dict[str, np.ndarray | float]:
#         """
#         Full inverter chain:
#             dq cmd -> abc phase refs
#                    -> SVPWM offset-voltage pole refs
#                    -> duty ratios
#                    -> dead-time correction
#                    -> average pole voltages
#                    -> average phase voltages
#                    -> back to dq applied voltage

#         Returns a dictionary so you can inspect everything later if needed.
#         """
#         # dq -> abc phase refs (line-neutral references)
#         v_abc_ref = dq0_to_abc(np.array([v_d_cmd, v_q_cmd, 0.0]), theta_frame)

#         # offset-voltage SVPWM
#         v_pole_ref, v_offset = phase_refs_to_svpwm_pole_refs(v_abc_ref)

#         # duty commands
#         d_cmd = pole_refs_to_duties(v_pole_ref, Vdc)

#         # phase currents for dead-time sign
#         i_abc = dq0_to_abc(np.array([i_d, i_q, 0.0]), theta_frame)

#         if use_deadtime:
#             d_eff = apply_deadtime_to_duties(d_cmd, i_abc, deadtime, T_pwm)
#         else:
#             d_eff = d_cmd.copy()

#         # actual average voltages after dead-time
#         v_pole_avg = duties_to_average_pole_voltages(d_eff, Vdc)
#         v_abc_avg = pole_to_phase_voltages(v_pole_avg)

#         # back to dq applied voltage
#         v_dq0_applied = T_matrix(theta_frame) @ v_abc_avg
#         v_d_applied, v_q_applied, _ = v_dq0_applied

#         return {
#             "v_abc_ref": v_abc_ref,
#             "v_offset": v_offset,
#             "v_pole_ref": v_pole_ref,
#             "d_cmd": d_cmd,
#             "i_abc": i_abc,
#             "d_eff": d_eff,
#             "v_pole_avg": v_pole_avg,
#             "v_abc_avg": v_abc_avg,
#             "v_d_applied": v_d_applied,
#             "v_q_applied": v_q_applied,
#         }

#     # =============================================================================
#     # EVENT FUNCTIONS
#     # =============================================================================
#     # event functions when temperature fault happens = demagnetisation = stop solver
#     # def stator_fault_event(self, t: float, x: np.ndarray) -> float:
#     #     T_s = x[7]
#     #     return self.T_fault_stator - T_s

#     # def magnet_fault_event(self, t: float, x: np.ndarray) -> float:
#     #     T_m = x[8]
#     #     return self.T_fault_magnet - T_m

#     def total_resisting_torque_from_inputs(
#         self,
#         omega_m: float,
#         mass: float,
#         slope_deg: float,
#         crr: float
#     ) -> float:
#         slope_rad = np.deg2rad(slope_deg)
#         v = self.vehicle_speed_from_motor_speed(omega_m)
#         s_v = self.smooth_sign(v, v_eps=0.05)

#         # source: https://x-engineer.org/modeling-simulation-vehicle-automatic-transmission/5/ 
#         # Grade or slope resistance (19)
#         F_grade = mass * self.g * np.sin(slope_rad)

#         # Rolling resistance (16)
#         F_roll = crr * mass * self.g * np.cos(slope_rad) * s_v

#         # Aerodynamic drag (15)
#         F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v * abs(v)

#         # lumped modeling approximation: a linear viscous-damping force proportional to speed, added to represent losses 
#         # that grow roughly with velocity, such as driveline parasitics, bearing drag, and other unmodeled speed-proportional effects
#         F_visc = self.b_vehicle * v

#         F_total = F_grade + F_roll + F_aero + F_visc

#         # Convert force at the tire to wheel torque 
#         T_wheel = F_total * self.wheel_radius
#         # T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)
#         # because of downhill
#         '''
#         This handles the difference between driving and regenerative braking (downhill) in terms of drivetrain efficiency.
#         When Twheel≥0
#         Twheel​≥0 (driving, uphill, flat road):The motor is pushing the vehicle forward. Energy flows from motor to wheels, so the drivetrain efficiency 
#         ηdt​ acts as a loss — the motor must produce more torque than what arrives at the wheel
#         '''
#         if T_wheel >= 0.0:
#             T_motor_load = T_wheel / max(self.drivetrain_eff * self.gear_ratio, 1e-12)
#         else:
#             T_motor_load = T_wheel * self.drivetrain_eff / max(self.gear_ratio, 1e-12)
#         return T_motor_load

#     def apply_rated_power_limit(self, torque_cmd: float, omega_m: float) -> float:
#         """
#         Limit torque command so that |T * omega_m| <= P_rated.

#         This is a motor rated mechanical power limit, not an inverter limit.
#         It should act as a supervisory constraint on torque request.
#         """
#         if self.P_rated <= 0.0:
#             return torque_cmd

#         omega_abs = abs(omega_m)

#         # Near zero speed, do not apply the power limit.
#         # Current limit / torque limit should dominate there.
#         if omega_abs < 1e-3:
#             return torque_cmd

#         torque_power_limit = self.P_rated / omega_abs
#         return float(np.clip(torque_cmd, -torque_power_limit, torque_power_limit))

#     # =============================================================================
#     # ODE
#     # =============================================================================
#     def motor_ode_pmsm_controlled(self, t: float, x: np.ndarray) -> List[float]:
#         """IPMSM ODE in chosen reference frame (dq axes)."""
#         # i_d, i_q, omega_m, theta_m, xint_d, xint_q, xint_s = x
#         (
#             i_d, i_q, omega_m, theta_m,
#             xint_d, xint_q, xint_s,
#             T_s, T_m,
#             omega_ref_f, mass_f, slope_f_deg, crr_f
#         ) = x
#         # temp ignored
#         # (
#         #     i_d, i_q, omega_m, theta_m,
#         #     xint_d, xint_q, xint_s,
#         #     omega_ref_f, mass_f, slope_f_deg, crr_f
#         # ) = x
#         R_s = self.R_s_of_T(T_s)
#         phi_f = float(self.phi_f_of_T(T_m))

#         # R_s = self.R_s_ref * (1.0 + self.alpha_cu * (T_s - self.T_ref))
#         # phi_f = self.phi_f_ref * (1.0 - self.alpha_phi * (T_m - self.T_ref))
#         # if phi_f < self.phi_f_min:
#         #     phi_f = self.phi_f_min
#         # temp ignored
#         # R_s = self.R_s_ref
#         # phi_f = self.phi_f_ref
#         # xint = integral part is a state

#         # rotor electrical angle: theta_r_e = p * theta_m (p=polepairs)
#         # rotor electrical quantities
#         omega_r_e = self.polepairs * omega_m
#         theta_r_e = self.polepairs * theta_m

#         # MTPA LUT - beg
#         Ld_now, Lq_now = self.motor_inductances(self.machine_type, i_d, i_q, self.mtpa_lut)
#         K_pd_now = Ld_now * self.omega_cc  # Eq. (6.28)
#         K_pq_now = Lq_now * self.omega_cc  # Eq. (6.29)
#         K_i_now = R_s * self.omega_cc
#         K_aw_d_now = K_i_now / max(K_pd_now, 1e-12)
#         K_aw_q_now = K_i_now / max(K_pq_now, 1e-12)
#         # MTPA LUT - end

#         # IMPORTANT:
#         #  The electrical angle is faster than the mechanical angle.
#         #  Reason: each mechanical revolution contains p electrical revolutions.
#         #  Because the stator rotating field completes one electrical cycle for each pole pair.

#         # ==========================================================
#         # REFERENCE FRAME SECTION (what dq means right now)
#         # ----------------------------------------------------------
#         # frame_mode == "rotor_foc":
#         #   theta_frame = theta_r_e
#         #   omega_frame = omega_r_e
#         #   -> classic rotor-aligned FOC frame for PMSM
#         #
#         # frame_mode == "stator":
#         #   theta_frame = omega_e*t
#         #   omega_frame = omega_e
#         #   -> stator synchronous frame (needs phase locking to be stable)
#         # ==========================================================

#         # IMPORTANT:
#         #  Phase locking in a PMSM means keeping the stator magnetic field
#         #  perfectly synchronized with the rotor’s magnetic field so torque
#         #  is produced smoothly and efficiently.
#         #  - Phase locking ensures these two fields rotate at the same
#         #    electrical speed and maintain the correct angle between them.
#         #  - If they stay locked → constant torque
#         #  - If they lose lock → torque drops, vibration, or the motor stalls

#         # choose frame angle (where dq variables are referenced)
#         if self.frame_mode == 'rotor_foc':
#             theta_frame = theta_r_e
#             omega_frame = omega_r_e
#         else:
#             theta_frame = self.omega_e * t
#             omega_frame = self.omega_e

#         omega_ref_tgt = self.omega_ref(t)
#         mass_tgt = self.mass_target(t)
#         slope_tgt_deg = self.slope_target(t)
#         crr_tgt = self.crr_target(t)

#         domega_ref_f_dt = self.rate_limited_tracking(
#             omega_ref_f,   # current filtered speed reference
#             omega_ref_tgt,  # desired target value
#             self.tau_omega_ref,  # how quickly to track the target
#             self.omega_ref_rate_up,  # maximum rate of increase
#             self.omega_ref_rate_down  # maximum rate of decrease
#         )    
#         dmass_f_dt = self.first_order_tracking(mass_f, mass_tgt, self.tau_mass)
#         dslope_f_dt = self.first_order_tracking(slope_f_deg, slope_tgt_deg, self.tau_slope)
#         dcrr_f_dt = self.first_order_tracking(crr_f, crr_tgt, self.tau_crr)

#         # ==========================================================
#         # SPEED CONTROLLER (outer loop)  -> produces torque_ref -> iq_ref
#         # ----------------------------------------------------------
#         # w_ref = self.omega_ref(t)
#         # w_ref = self._seg_omega_ref
#         w_ref = omega_ref_f

#         # Clamp commanded speed to finite max speed if applicable
#         if self.machine_type == "ipmsm":
#             omega_max_e_pre = self.ipmsm_max_speed_finite(phi_f)
#         else:
#             omega_max_e_pre = self.spmsm_max_speed_finite(phi_f)

#         if omega_max_e_pre is not None:
#             omega_max_m_pre = omega_max_e_pre / self.polepairs
#             w_ref = np.clip(w_ref, -omega_max_m_pre, omega_max_m_pre)

#         # err_omega = w_ref - omega_m

#         # iq_ref_uns = self.K_ps * err_omega + xint_s
#         # iq_ref_speed = np.clip(iq_ref_uns, -self.Imax, self.Imax)
#         # xint_s_dot = self.K_is * err_omega + self.K_aw_s * (iq_ref_speed - iq_ref_uns)

#         J_ctrl_now = self.J_motor + mass_f * self.wheel_radius**2 / (self.gear_ratio**2)
#         K_T_now = 1.5 * self.polepairs * phi_f
#         K_ps_now = J_ctrl_now * self.omega_cs / max(K_T_now, 1e-12)
#         K_is_now = K_ps_now * self.omega_cs / 5.0
#         K_aw_s_now = K_is_now / max(K_ps_now, 1e-12)

#         err_omega = w_ref - omega_m

#         iq_ref_uns = K_ps_now * err_omega + xint_s
#         iq_ref_speed = np.clip(iq_ref_uns, -self.Imax, self.Imax)
#         xint_s_dot = K_is_now * err_omega + K_aw_s_now * (iq_ref_speed - iq_ref_uns)


#         # Current references with field-weakening
#         torque_cmd = self.torque_from_iq_command(self.machine_type, iq_ref_speed, phi_f)
#         torque_cmd = self.apply_rated_power_limit(torque_cmd, omega_m)

#         id_ref, iq_ref, omega_base_e, omega_max_e, max_speed_reached_now = \
#             self.current_references_with_fw(self.machine_type, omega_m, torque_cmd, phi_f, self.mtpa_lut)

#         # ==========================================================
#         # CURRENT CONTROLLERS (inner loop) -> produce v_d, v_q
#         # ==========================================================
#         err_d = id_ref - i_d
#         err_q = iq_ref - i_q

#         # PI (unsaturated) -> feedback control (PI)
#         u_d_uns = K_pd_now * err_d + xint_d
#         u_q_uns = K_pq_now * err_q + xint_q

#         # Decoupling + back-EMF feedforward (rotor-aligned dq form)
#         v_d_cmd_uns = u_d_uns - omega_frame * Lq_now * i_q
#         v_q_cmd_uns = u_q_uns + omega_frame * (Ld_now * i_d + phi_f)

#         # inverter dq vector magnitude limit
#         v_d_cmd, v_q_cmd = dq_voltage_limit(v_d_cmd_uns, v_q_cmd_uns, self.Vmax)

#         # anti-windup should compare the SATURATED controller command
#         # against the unsaturated controller request
#         xint_d_dot = K_i_now * err_d + K_aw_d_now * (v_d_cmd - v_d_cmd_uns)
#         xint_q_dot = K_i_now * err_q + K_aw_q_now * (v_q_cmd - v_q_cmd_uns)

#         # ==========================================================
#         # DRIVE MODE SECTION (what voltages actually applied)
#         # ==========================================================
#         if self.drive_mode == "closed_loop_foc":
#             if self.use_svpwm:
#                 inv = self.dq_cmd_to_inverter_output(
#                     v_d_cmd=v_d_cmd,
#                     v_q_cmd=v_q_cmd,
#                     theta_frame=theta_frame,
#                     i_d=i_d,
#                     i_q=i_q,
#                     Vdc=self.Vdc,
#                     T_pwm=self.T_pwm,
#                     deadtime=self.deadtime,
#                     use_deadtime=self.include_deadtime
#                 )
#                 v_d = float(inv["v_d_applied"])
#                 v_q = float(inv["v_q_applied"])
#             else:
#                 v_d, v_q = v_d_cmd, v_q_cmd
#         else:
#             # CRITICAL: slip speed (works for both frames)
#             v_as = self.V_m * np.cos(self.omega_e * t)
#             v_bs = self.V_m * np.cos(self.omega_e * t - 2.0 * np.pi / 3.0)
#             v_cs = self.V_m * np.cos(self.omega_e * t - 4.0 * np.pi / 3.0)
#             v_dq0 = T_matrix(theta_frame) @ np.array([v_as, v_bs, v_cs])
#             v_d, v_q, _ = v_dq0
#             v_d, v_q = dq_voltage_limit(v_d, v_q, self.Vmax)

#         # ==========================================================
#         # ELECTRICAL DYNAMICS (dq currents)
#         # ==========================================================
#         if self.frame_mode == "rotor_foc":
#             di_d_dt = (v_d - R_s * i_d + omega_frame * Lq_now * i_q) / Ld_now
#             di_q_dt = (v_q - R_s * i_q - omega_frame * (Ld_now * i_d + phi_f)) / Lq_now
#         else:
#             # ----------------------------------------------------------
#             # STATOR SYNCHRONOUS dq frame (theta_frame = omega_e*t)
#             # Saliency + PM flux rotate into this frame -> L(t), phi(t)
#             # ----------------------------------------------------------
#             theta_err = theta_r_e - theta_frame
#             theta_err_dot = omega_r_e - omega_frame

#             deltaL = (self.L_ds - self.L_qs)

#             # time-varying inductances (your Eq. 4.110-style form)
#             Ld_eff = 0.5 * (self.L_ds + self.L_qs) + 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
#             Lq_eff = 0.5 * (self.L_ds + self.L_qs) - 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
#             Ldq_eff = 0.5 * (self.L_ds - self.L_qs) * np.sin(2.0 * theta_err)

#             # if SPMSM (deltaL == 0), explicitly zero out coupling to avoid numeric noise
#             if abs(deltaL) < 1e-12:
#                 Ld_eff = self.L_ds
#                 Lq_eff = self.L_ds
#                 Ldq_eff = 0.0

#             # PM flux projection into the stator synchronous dq frame
#             phi_d = phi_f * np.cos(theta_err)
#             phi_q = -phi_f * np.sin(theta_err)

#             # flux linkages in this frame
#             lambda_d = Ld_eff * i_d + Ldq_eff * i_q + phi_d
#             lambda_q = Ldq_eff * i_d + Lq_eff * i_q + phi_q

#             # voltage equations in rotating dq frame
#             dlambda_d_dt = v_d - R_s * i_d + omega_frame * lambda_q
#             dlambda_q_dt = v_q - R_s * i_q - omega_frame * lambda_d
#             dlambda_dt = np.array([dlambda_d_dt, dlambda_q_dt])

#             # time-derivatives of inductances (chain rule)
#             if abs(deltaL) < 1e-12:
#                 dLd_dt = dLq_dt = dLdq_dt = 0.0
#             else:
#                 dLd_dt = -deltaL * np.sin(2.0 * theta_err) * theta_err_dot
#                 dLq_dt = deltaL * np.sin(2.0 * theta_err) * theta_err_dot
#                 dLdq_dt = deltaL * np.cos(2.0 * theta_err) * theta_err_dot

#             # time derivative of PM flux projections
#             dphi_d_dt = -phi_f * np.sin(theta_err) * theta_err_dot
#             dphi_q_dt = -phi_f * np.cos(theta_err) * theta_err_dot

#             # assemble L matrix and its derivative effect
#             Lmat = np.array([[Ld_eff, Ldq_eff],
#                              [Ldq_eff, Lq_eff]])

#             dL_times_i = np.array([
#                 dLd_dt * i_d + dLdq_dt * i_q,
#                 dLdq_dt * i_d + dLq_dt * i_q
#             ])

#             dphi_dt = np.array([dphi_d_dt, dphi_q_dt])

#             rhs = dlambda_dt - dL_times_i - dphi_dt
#             di_d_dt, di_q_dt = np.linalg.solve(Lmat, rhs)

#         # ==========================================================
#         # TORQUE
#         # ==========================================================
#         if self.frame_mode == "rotor_foc":
#             T_e = 1.5 * self.polepairs * (phi_f * i_q + (Ld_now - Lq_now) * i_d * i_q)
#         else:
#             T_e = 1.5 * self.polepairs * (lambda_d * i_q - lambda_q * i_d)

#         # Mechanical dynamics
#         # eq. 1.59
#         # J_total = self.total_inertia(t)
#         # T_load = self.total_resisting_torque(t, omega_m)
#         # J_total = self._seg_J_total
#         # T_load = self.total_resisting_torque_segment(omega_m)
#         # J_total = self.J_motor + mass_f * self.wheel_radius**2 / (self.gear_ratio**2)
#         # T_load = self.total_resisting_torque_from_inputs(omega_m, mass_f, slope_f_deg, crr_f)
#         # J_total = self.J_motor + mass_tgt * self.wheel_radius**2 / (self.gear_ratio**2)
#         # T_load = self.total_resisting_torque_from_inputs(omega_m, mass_tgt, slope_tgt_deg, crr_tgt)
#         J_total = self.J_motor + mass_f * self.wheel_radius**2 / (self.gear_ratio**2)
#         T_load = self.total_resisting_torque_from_inputs(omega_m, mass_f, slope_f_deg, crr_f)
#         domega_m_dt = (T_e - T_load) / J_total
#         dtheta_m_dt = omega_m

#         # Thermal model =================================================================================================================================
#         # temp ignored
#         # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://repositum.tuwien.at/bitstream/20.500.12708/196526/1/Fischer%20Andreas%20-%202024%20-%20Model-based%20thermal%20protection%20strategy%20for%20a%20permanent...pdf
#             # eq. 2.7
#         P_cu = 1.5 * R_s * (i_d ** 2 + i_q ** 2)  # Copper loss

#         # Iron (core) loss — Steinmetz model, gated by include_iron_loss.
#         # B_core is approximated from the dq flux-linkage magnitude, scaled so that
#         # the flux-linkage magnitude at RATED operation (id=0, iq=I_rated) maps to
#         # B_iron_rated. Using phi_f_ref alone (the no-load PM flux) as the
#         # reference would badly under-scale it whenever armature-reaction flux
#         # (Lq*iq) dominates, which is common for weak-magnet/high-leakage motors.
#         # This model has no explicit tooth/yoke geometry, so this lumped scaling
#         # stands in for the article's per-tooth/per-yoke flux-density waveform.
#         if self.include_iron_loss:
#             lambda_d = Ld_now * i_d + phi_f
#             lambda_q = Lq_now * i_q
#             psi_s_mag = float(np.hypot(lambda_d, lambda_q))
#             lambda_rated = float(np.hypot(self.phi_f_ref, self.Lq0 * self.I_rated))
#             B_core = self.B_iron_rated * psi_s_mag / max(lambda_rated, 1e-12)
#             omega_s = abs(omega_r_e)
#             P_iron = self.core_volume_m3 * (
#                 self.k_h_iron * (B_core ** self.beta_iron) * omega_s
#                 + self.k_e_iron * (B_core ** 2) * (omega_s ** 2)
#             )
#         else:
#             P_iron = 0.0

#         # # TODO: Simple rotor/magnet loss placeholder
#         P_rotor = 0.0

#         # # Thermal dynamics
#         # # rate of change of stator temperature
#         # # stator temperature rate =
#             #     # (copper loss heat input
#             #     # + iron loss heat input
#             #     # - heat from stator to ambient
#             #     # - heat from stator to magnet/rotor)
#             #     # / stator thermal capacitance
#         dT_s_dt = (P_cu + P_iron - (T_s - self.T_amb) / self.R_th_sa - (T_s - T_m) / self.R_th_sm) / self.C_th_s
#         # # rate of change of rotor/magnet temperature
#         # # magnet/rotor temperature rate =
#             #     # (rotor loss heat input
#             #     # + heat from stator to magnet/rotor
#             #     # - heat from magnet/rotor to ambient)
#             #     # / magnet/rotor thermal capacitance
#         dT_m_dt = (P_rotor + (T_s - T_m) / self.R_th_sm - (T_m - self.T_amb) / self.R_th_ma) / self.C_th_m
#         # ===============================================================================================================================================

#         # return [di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt, xint_d_dot, xint_q_dot, xint_s_dot]
#         # return [di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt, xint_d_dot, xint_q_dot, xint_s_dot, dT_s_dt, dT_m_dt]
#         # temp ignored
#         return [
#             di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt,
#             xint_d_dot, xint_q_dot, xint_s_dot,
#             dT_s_dt, dT_m_dt,
#             domega_ref_f_dt, dmass_f_dt, dslope_f_dt, dcrr_f_dt
#         ]
#         # return [
#         #     di_d_dt, di_q_dt, domega_m_dt, dtheta_m_dt,
#         #     xint_d_dot, xint_q_dot, xint_s_dot,
#         #     domega_ref_f_dt, dmass_f_dt, dslope_f_dt, dcrr_f_dt
#         # ]
#     # =============================================================================
#     # SOLVE + POSTPROCESS + FIGURES
#     # =============================================================================
#     def run(self) -> Dict[str, Any]:
#         # =============================================================================
#         # INITIAL CONDITIONS
#         # =============================================================================
#         i_d0, i_q0 = 0.0, 0.0
#         omega_m0 = 0.0
#         theta_m0 = 0.0
#         xint_d0 = 0.0
#         xint_q0 = 0.0
#         xint_s0 = 0.0
#         # temp ignored
#         T_s0 = self.T_amb
#         T_m0 = self.T_amb

#         # steps
#         omega_ref_f0 = self.omega_ref(self.t0)
#         mass_f0 = self.mass_target(self.t0)
#         slope_f0 = self.slope_target(self.t0)
#         crr_f0 = self.crr_target(self.t0)

#         # x0 = np.array([i_d0, i_q0, omega_m0, theta_m0, xint_d0, xint_q0, xint_s0, T_s0, T_m0])
#         # temp ignored
#         x0 = np.array([
#             i_d0, i_q0, omega_m0, theta_m0,
#             xint_d0, xint_q0, xint_s0,
#             T_s0, T_m0,
#             omega_ref_f0, mass_f0, slope_f0, crr_f0
#         ])
#         # x0 = np.array([
#         #     i_d0, i_q0, omega_m0, theta_m0,
#         #     xint_d0, xint_q0, xint_s0,
#         #     omega_ref_f0, mass_f0, slope_f0, crr_f0
#         # ])

#         # =============================================================================
#         # SOLVE
#         # =============================================================================
#         # TODO: solve_ivp research and mention in the thesis how it works
#         '''
#         - Numerically computes the state trajectory of a system starting from an initial state
#         - So for each time t and state vector x, it returns the derivative dx/dt.
#         - Explicit solver: directly estimates the next value from the current slope
#         - Implicit solver: solves an equation involving the unknown next value itself
#           - That means the solver must solve for it numerically.
#           - This is more computationally expensive per step, but much more stable for stiff systems.
#         - The solver is always estimating its own local error and adjusting step size.
#         - allowed error≈atol+rtol⋅∣y∣
#         '''
#         # temp ignored
#         def stator_fault_event(t, x):
#             T_s = x[7]
#             return self.T_fault_stator - T_s

#         # stator_fault_event.terminal = True
#         # stator_fault_event.direction = -1

#         def magnet_fault_event(t, x):
#             T_m = x[8]
#             return self.T_fault_magnet - T_m

#         # magnet_fault_event.terminal = True
#         # magnet_fault_event.direction = -1

#         print("BEFORE SOLVE:")
#         print("tfinal =", self.tfinal)
#         print("t_eval start/end =", self.t_eval[0], self.t_eval[-1])
#         print("t_steps =", self.t_steps)
#         print("omega_steps =", self.omega_steps)
#         # print("breakpoints =", self.get_schedule_breakpoints())

#         # t, y, segment_solutions, stop_info = self.solve_piecewise(x0)
#         # temp ignored
#         # sol = solve_ivp(
#         #     self.motor_ode_pmsm_controlled,
#         #     (self.t0, self.tfinal),
#         #     x0,
#         #     method=self.solver_method,
#         #     t_eval=self.t_eval,
#         #     rtol=self.rtol,
#         #     atol=self.atol,
#         #     events=[stator_fault_event, magnet_fault_event],
#         # )
#         sol = solve_ivp(
#             self.motor_ode_pmsm_controlled,
#             (self.t0, self.tfinal),
#             x0,
#             method=self.solver_method,
#             t_eval=self.t_eval,
#             rtol=self.rtol,
#             atol=self.atol,
#         )

#         t = sol.t
#         y = sol.y

#         stop_info = {
#             "success": sol.success,
#             "status": sol.status,
#             "message": sol.message,
#             "failed_segment": None,
#             "failed_time": float(sol.t[-1]) if sol.t.size else None,
#             # temp ignored
#             # "t_events": [
#             #     sol.t_events[0].tolist() if len(sol.t_events) > 0 else [],
#             #     sol.t_events[1].tolist() if len(sol.t_events) > 1 else [],
#             # ],
#         }

#         print("AFTER SOLVE:")
#         print("final returned time =", t[-1])
#         print("success =", stop_info["success"])
#         print("status =", stop_info["status"])
#         print("message =", stop_info["message"])
#         print("failed segment =", stop_info["failed_segment"])
#         print("failed time =", stop_info["failed_time"])

#         print(
#             "piecewise summary:",
#             stop_info["success"],
#             stop_info["status"],
#             t[-1],
#             stop_info["message"]
#         )

#         # ------------------- check event triggers -------------------
#         # event_messages: List[str] = []

#         # if sol.t_events[0].size > 0:
#         #     event_messages.append(f"FAULT: stator temperature exceeded {self.T_fault_stator:.1f} °C at t = {sol.t_events[0][0]:.6f} s")

#         # if sol.t_events[1].size > 0:
#         #     event_messages.append(f"FAULT: magnet temperature exceeded {self.T_fault_magnet:.1f} °C at t = {sol.t_events[1][0]:.6f} s")

#         # if sol.status == 1:
#         #     event_messages.append("Simulation terminated due to thermal fault event.")

#         # if sol.t_events[0].size > 0:
#         #     event_messages.append(f"Stator fault at t = {sol.t_events[0][0]:.6f} s")

#         # if sol.t_events[1].size > 0:
#         #     event_messages.append(f"Magnet fault at t = {sol.t_events[1][0]:.6f} s")
        
#         event_messages: List[str] = []
#         # temp ignored
#         # stator_events = stop_info["t_events"][0]
#         # magnet_events = stop_info["t_events"][1]

#         # if len(stator_events) > 0:
#         #     event_messages.append(
#         #         f"FAULT: stator temperature exceeded {self.T_fault_stator:.1f} °C at t = {stator_events[0]:.6f} s"
#         #     )

#         # if len(magnet_events) > 0:
#         #     event_messages.append(
#         #         f"FAULT: magnet temperature exceeded {self.T_fault_magnet:.1f} °C at t = {magnet_events[0]:.6f} s"
#         #     )

#         # if stop_info["status"] == 1:
#         #     event_messages.append("Simulation terminated due to thermal fault event.")

#         # if len(stator_events) > 0:
#         #     event_messages.append(f"Stator fault at t = {stator_events[0]:.6f} s")

#         # if len(magnet_events) > 0:
#         #     event_messages.append(f"Magnet fault at t = {magnet_events[0]:.6f} s")
#         # ------------------------------------------------------------

#         # t = sol.t
#         # i_d = sol.y[0]
#         # i_q = sol.y[1]
#         # omega_m = sol.y[2]
#         # theta_m = sol.y[3]

#         i_d = y[0]
#         i_q = y[1]
#         omega_m = y[2]
#         theta_m = y[3]

#         # =============================================================================
#         # MAX-SPEED REACHED MESSAGE (finite-speed FW cases)
#         # =============================================================================
#         # phi_f_arr = np.asarray(self.phi_f_of_T(sol.y[8]), dtype=float)
#         # temp ignored
#         phi_f_arr = np.asarray(self.phi_f_of_T(y[8]), dtype=float)
#         # phi_f_arr = np.full_like(t, self.phi_f_ref, dtype=float)

#         omega_max_e_arr = np.zeros_like(t)
#         for k in range(len(t)):
#             if self.machine_type == "ipmsm":
#                 omg = self.ipmsm_max_speed_finite(phi_f_arr[k])
#             else:
#                 omg = self.spmsm_max_speed_finite(phi_f_arr[k])

#             omega_max_e_arr[k] = np.nan if omg is None else omg

#         omega_max_m_arr = omega_max_e_arr / self.polepairs
#         reached_mask = np.abs(omega_m) >= self.MAX_SPEED_REACHED_TOL * omega_max_m_arr

#         max_speed_message = ""
#         if np.any(reached_mask):
#             k_hit = np.argmax(reached_mask)
#             max_speed_message = (
#                 f"MAX SPEED REACHED: {self.machine_type.upper()} finite-speed limit reached at "
#                 f"t = {t[k_hit]:.6f} s | "
#                 f"omega_m = {omega_m[k_hit]:.3f} rad/s | "
#                 f"speed = {omega_m[k_hit] * 60.0 / (2.0 * np.pi):.1f} rpm"
#             )
#         else:
#             max_speed_message = f"Max-speed status: {self.machine_type.upper()} finite-speed limit not reached."

#         # # load info for plotting
#         # T_load_arr = np.array([self.total_resisting_torque(tk, omega_m[k]) for k, tk in enumerate(t)])
#         # v_vehicle = np.array([self.vehicle_speed_from_motor_speed(w) for w in omega_m])
#         # mass_arr = np.array([self.piecewise_step(tk, self.mass_step_times, self.mass_step_values) for tk in t])
#         # slope_arr = np.array([self.piecewise_step(tk, self.slope_step_times, self.slope_step_values_deg) for tk in t])
#         # crr_arr = np.array([self.piecewise_step(tk, self.crr_step_times, self.crr_step_values) for tk in t])

#         # Extract temperatures after the solver
#         # T_s = sol.y[7]
#         # T_m = sol.y[8]

#         # temp ignored
#         T_s = y[7]
#         T_m = y[8]

#         # ================================================================================================================================================
#         # temperature warning and fault detection (demagnetisation)
#         '''
#         | Part              | Main risk              | Consequence           |
#         | ----------------- | ---------------------- | --------------------- |
#         | **Stator**        | insulation overheating | electrical failure    |
#         | **Rotor magnets** | demagnetization        | permanent torque loss |
#         Stator temp → protects windings
#         Rotor temp → protects magnets
#         '''

#         warning_messages = []
#         # temp ignored
#         max_Ts = np.max(T_s)
#         max_Tm = np.max(T_m)

#         # if max_Ts >= self.T_warn_stator:
#         #     t_warn_s = t[np.argmax(T_s >= self.T_warn_stator)]
#         #     warning_messages.append(
#         #         f"WARNING: stator temperature exceeded warning threshold "
#         #         f"({self.T_warn_stator:.1f} °C) at t = {t_warn_s:.4f} s."
#         #     )

#         # if max_Ts >= self.T_fault_stator:
#         #     t_fault_s = t[np.argmax(T_s >= self.T_fault_stator)]
#         #     warning_messages.append(
#         #         f"FAULT: stator temperature exceeded fault threshold "
#         #         f"({self.T_fault_stator:.1f} °C) at t = {t_fault_s:.4f} s."
#         #     )

#         # if max_Tm >= self.T_warn_magnet:
#         #     t_warn_m = t[np.argmax(T_m >= self.T_warn_magnet)]
#         #     warning_messages.append(
#         #         f"WARNING: magnet temperature exceeded warning threshold "
#         #         f"({self.T_warn_magnet:.1f} °C) at t = {t_warn_m:.4f} s."
#         #     )

#         # if max_Tm >= self.T_fault_magnet:
#         #     t_fault_m = t[np.argmax(T_m >= self.T_fault_magnet)]
#         #     warning_messages.append(
#         #         f"FAULT: magnet temperature exceeded fault threshold "
#         #         f"({self.T_fault_magnet:.1f} °C) at t = {t_fault_m:.4f} s."
#         #     )

#         # if max_Tm >= self.T_demag:
#         #     t_demag = t[np.argmax(T_m >= self.T_demag)]
#         #     warning_messages.append(
#         #         f"DEMAG RISK: magnet temperature exceeded irreversible-demagnetization "
#         #         f"risk threshold ({self.T_demag:.1f} °C) at t = {t_demag:.4f} s."
#         #     )

#         # if not warning_messages:
#         #     warning_messages.append("Thermal status: OK. No thermal thresholds exceeded.")
#         # ================================================================================================================================================

#         # Post torque
#         # IMPORTANT:
#         #  What is the difference between T_e and Torque?
#         #  - T_e = instantaneous electromagnetic torque computed inside the ODE
#         #          at that time step
#         #        = scalar
#         #  * Torque = an array of torque values computed after the solver for
#         #             plotting
#         #           = vector
#         #  So they represent the same physical quantity; one is the current
#         #    instant, one is the saved trajectory.

#         if self.frame_mode == "rotor_foc":
#             Torque = np.zeros_like(t)
#             for k in range(len(t)):
#                 if self.machine_type == "spmsm" or self.mtpa_lut is None:
#                     Ld_k, Lq_k = self.L_ds, self.L_qs
#                 else:
#                     Ld_k, Lq_k = self.mtpa_lut.ld(i_d[k]), self.mtpa_lut.lq(i_q[k])

#                 Torque[k] = 1.5 * self.polepairs * (
#                         phi_f_arr[k] * i_q[k] + (Ld_k - Lq_k) * i_d[k] * i_q[k]
#                 )
#         else:
#             Torque = np.zeros_like(t)

#             for k, tk in enumerate(t):
#                 omega_mk = omega_m[k]
#                 theta_mk = theta_m[k]
#                 i_dk, i_qk = i_d[k], i_q[k]

#                 theta_r_e = self.polepairs * theta_mk
#                 theta_frame = self.omega_e * tk
#                 theta_err = theta_r_e - theta_frame

#                 Ld_eff = 0.5 * (self.L_ds + self.L_qs) + 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
#                 Lq_eff = 0.5 * (self.L_ds + self.L_qs) - 0.5 * (self.L_ds - self.L_qs) * np.cos(2.0 * theta_err)
#                 Ldq_eff = 0.5 * (self.L_ds - self.L_qs) * np.sin(2.0 * theta_err)

#                 # temp ignored
#                 # phi_fk = float(self.phi_f_of_T(T_m[k]))
#                 phi_fk = self.phi_f_ref
#                 phi_d = phi_fk * np.cos(theta_err)
#                 phi_q = -phi_fk * np.sin(theta_err)

#                 lambda_d = Ld_eff * i_dk + Ldq_eff * i_qk + phi_d
#                 lambda_q = Ldq_eff * i_dk + Lq_eff * i_qk + phi_q

#                 Torque[k] = 1.5 * self.polepairs * (lambda_d * i_qk - lambda_q * i_dk)

#         # Speed in rpm
#         speed_rpm = omega_m * 60.0 / (2.0 * np.pi)

#         # =============================================================================
#         # BUILD PHASE CURRENTS FOR PLOTTING (dq -> abc in THE SAME theta_frame)
#         # =============================================================================
#         i_phase_a = np.zeros_like(t)
#         i_phase_b = np.zeros_like(t)
#         i_phase_c = np.zeros_like(t)

#         for k, tk in enumerate(t):
#             theta_r_e = self.polepairs * theta_m[k]
#             if self.frame_mode == "rotor_foc":
#                 theta_frame = theta_r_e
#             else:
#                 theta_frame = self.omega_e * tk
#             i_abc = dq0_to_abc(np.array([i_d[k], i_q[k], 0.0]), theta_frame)
#             i_phase_a[k], i_phase_b[k], i_phase_c[k] = i_abc

#         # =============================================================================
#         # PLOTS (ALL-IN-ONE SCREEN)
#         # =============================================================================

#         # ---------- Rebuild controller/reference signals for plotting ----------
#         # xint_d = sol.y[4]
#         # xint_q = sol.y[5]
#         # xint_s = sol.y[6]

#         xint_d = y[4]
#         xint_q = y[5]
#         xint_s = y[6]

#         # steps
#         # temp ignored
#         omega_ref_f_arr = y[9]
#         mass_f_arr = y[10]
#         slope_f_arr = y[11]
#         crr_f_arr = y[12]
#         # omega_ref_f_arr = y[7]
#         # mass_f_arr = y[8]
#         # slope_f_arr = y[9]
#         # crr_f_arr = y[10]

#         # steps - beg
#         # T_load_arr = np.array([
#         #     self.total_resisting_torque_from_inputs(omega_m[k], mass_f_arr[k], slope_f_arr[k], crr_f_arr[k])
#         #     for k in range(len(t))
#         # ])

#         # mass_arr = mass_f_arr.copy()
#         # slope_arr = slope_f_arr.copy()
#         # crr_arr = crr_f_arr.copy()
#         # mass_arr = np.array([self.mass_target(tk) for tk in t])
#         # slope_arr = np.array([self.slope_target(tk) for tk in t])
#         # crr_arr = np.array([self.crr_target(tk) for tk in t])

#         v_vehicle = np.array([self.vehicle_speed_from_motor_speed(w) for w in omega_m])

#         # T_load_arr = np.array([
#         #     self.total_resisting_torque_from_inputs(omega_m[k], mass_arr[k], slope_arr[k], crr_arr[k])
#         #     for k in range(len(t))
#         # ])

#         mass_arr = mass_f_arr.copy()
#         slope_arr = slope_f_arr.copy()
#         crr_arr = crr_f_arr.copy()

#         T_load_arr = np.array([
#             self.total_resisting_torque_from_inputs(omega_m[k], mass_arr[k], slope_arr[k], crr_arr[k])
#             for k in range(len(t))
#         ])
#         # steps - end


#         # =============================================================================
#         # INDUCTANCE-HELPER CURVES FOR FRONTEND PLOT
#         # =============================================================================
#         id_grid_help = np.linspace(-self.Imax, 0.0, 300)
#         iq_grid_help = np.linspace(0.0, self.Imax, 300)

#         if self.machine_type == "spmsm" or self.mtpa_lut is None:
#             Ld_curve = np.full_like(id_grid_help, self.L_ds, dtype=float)
#             Lq_curve = np.full_like(iq_grid_help, self.L_qs, dtype=float)
#         else:
#             Ld_curve = np.array([self.mtpa_lut.ld(v) for v in id_grid_help], dtype=float)
#             Lq_curve = np.array([self.mtpa_lut.lq(v) for v in iq_grid_help], dtype=float)



#         id_ref_arr = np.zeros_like(t)
#         iq_ref_arr = np.zeros_like(t)  # clamped iq_ref used by current loop

#         # NEW: comparison arrays
#         iq_ref_pi_arr = np.zeros_like(t)  # q-current from speed PI before MTPA / FW
#         id_mtpa_arr = np.zeros_like(t)  # d-current after MTPA (before FW overwrite)
#         iq_mtpa_arr = np.zeros_like(t)  # q-current after MTPA (before FW overwrite)
#         is_pi_arr = np.zeros_like(t)  # current magnitude from PI-only request
#         is_mtpa_arr = np.zeros_like(t)  # current magnitude after MTPA
#         torque_cmd_arr = np.zeros_like(t)  # torque command produced from PI current request

#         # speed reference (rad/s and rpm for plotting)
#         omega_ref_arr = np.zeros_like(t)
#         speed_ref_rpm = np.zeros_like(t)

#         u_d_uns_arr = np.zeros_like(t)  # PI output before decoupling
#         u_q_uns_arr = np.zeros_like(t)

#         vd_uns_arr = np.zeros_like(t)  # after decoupling, before saturation
#         vq_uns_arr = np.zeros_like(t)

#         vd_cmd_arr = np.zeros_like(t)  # after saturation
#         vq_cmd_arr = np.zeros_like(t)

#         iq_ref_uns_arr = np.zeros_like(t)  # speed PI output before clamp
#         iq_ref_clamp_arr = np.zeros_like(t)  # speed PI output after clamp (same as iq_ref_arr)

#         v_mag_uns_arr = np.zeros_like(t)  # ||v*|| before saturation (Volts)
#         v_mag_cmd_arr = np.zeros_like(t)  # ||v||  after saturation (Volts)

#         omega_base_arr = np.zeros_like(t)
#         omega_max_arr = np.zeros_like(t)
#         max_speed_reached_arr = np.zeros_like(t, dtype=bool)

#         # SVPWM / inverter arrays for plotting ----
#         va_ref_arr = np.zeros_like(t)
#         vb_ref_arr = np.zeros_like(t)
#         vc_ref_arr = np.zeros_like(t)

#         v_offset_arr = np.zeros_like(t)

#         van_ref_arr = np.zeros_like(t)
#         vbn_ref_arr = np.zeros_like(t)
#         vcn_ref_arr = np.zeros_like(t)

#         da_cmd_arr = np.zeros_like(t)
#         db_cmd_arr = np.zeros_like(t)
#         dc_cmd_arr = np.zeros_like(t)

#         da_eff_arr = np.zeros_like(t)
#         db_eff_arr = np.zeros_like(t)
#         dc_eff_arr = np.zeros_like(t)

#         van_avg_arr = np.zeros_like(t)
#         vbn_avg_arr = np.zeros_like(t)
#         vcn_avg_arr = np.zeros_like(t)

#         va_avg_arr = np.zeros_like(t)
#         vb_avg_arr = np.zeros_like(t)
#         vc_avg_arr = np.zeros_like(t)

#         ed_cross_arr = np.zeros_like(t)
#         eq_backemf_arr = np.zeros_like(t)
#         epm_arr = np.zeros_like(t)
#         Ld_arr = np.zeros_like(t)
#         Lq_arr = np.zeros_like(t)

#         # Current-loop PI gains actually used at each timestep. K_pd/K_pq are
#         # NOT constant -- they scale with Ld_now/Lq_now, the current-dependent
#         # (saturating) dq inductances, so they move whenever i_d/i_q move into
#         # saturation. K_i is shared between both axes (it only depends on R_s,
#         # which itself drifts with stator temperature T_s) -- see K_pd_now/
#         # K_pq_now/K_i_now just above where they're computed each step.
#         K_pd_arr = np.zeros_like(t)
#         K_pq_arr = np.zeros_like(t)
#         K_i_arr = np.zeros_like(t)

#         # Speed-loop (outer) PI gains actually used at each timestep. Also not
#         # constant: K_ps ("DC-motor-like" gain, see the speed-loop comment
#         # below) scales with J_ctrl_now (effective control inertia, which
#         # grows with vehicle mass_f_arr when a load schedule is active) and
#         # with K_T_now = 1.5*polepairs*phi_f (the torque constant, which
#         # moves whenever phi_f itself moves -- flux weakening, temperature).
#         # K_is is tied to K_ps by a fixed ratio (omega_cs/5), so it moves in
#         # lockstep -- see K_ps_now/K_is_now just below where they're computed.
#         K_ps_arr = np.zeros_like(t)
#         K_is_arr = np.zeros_like(t)

#         P_cu_arr = np.zeros_like(t)
#         P_stator_loss_arr = np.zeros_like(t)
#         P_iron_arr = np.zeros_like(t)
#         B_core_arr = np.zeros_like(t)
#         # Core-loss (R_fe) branch current -- see __init__ for R_fe_iron calibration.
#         # This is what actually gets added to the terminal current for E_elec_in in the
#         # energy-balance section; it's an independent estimate from P_iron_arr above.
#         i_Fe_d_arr = np.zeros_like(t)
#         i_Fe_q_arr = np.zeros_like(t)

#         # actually applied voltages after PWM with deadtime
#         vd_applied_arr = np.zeros_like(t)
#         vq_applied_arr = np.zeros_like(t)
#         # PWM -------------------------------------

#         for k, tk in enumerate(t):
#             i_dk, i_qk = i_d[k], i_q[k]
#             omega_mk = omega_m[k]

#             omega_r_e = self.polepairs * omega_mk
#             omega_frame = omega_r_e if self.frame_mode == "rotor_foc" else self.omega_e

#             # ---- speed reference ----
#             # w_ref = self.omega_ref(tk)
#             w_ref = omega_ref_f_arr[k]
#             omega_ref_arr[k] = w_ref
#             speed_ref_rpm[k] = w_ref * 60.0 / (2.0 * np.pi)

#             # ---- speed loop -> iq_ref ----
#             err_omega = w_ref - omega_mk

#             # temp ignored
#             phi_fk = float(self.phi_f_of_T(T_m[k]))
#             # phi_fk = self.phi_f_ref

#             # ---------------------------------------------------------
#             # 1) Raw q-axis current request from speed PI
#             # ---------------------------------------------------------
#             # iq_ref_uns = self.K_ps * err_omega + xint_s[k]
#             J_ctrl_now = self.J_motor + mass_f_arr[k] * self.wheel_radius**2 / (self.gear_ratio**2)
#             K_T_now = 1.5 * self.polepairs * phi_fk
#             K_ps_now = J_ctrl_now * self.omega_cs / max(K_T_now, 1e-12)
#             K_is_now = K_ps_now * self.omega_cs / 5.0
#             K_aw_s_now = K_is_now / max(K_ps_now, 1e-12)

#             K_ps_arr[k] = K_ps_now
#             K_is_arr[k] = K_is_now

#             iq_ref_uns = K_ps_now * err_omega + xint_s[k]
#             iq_ref_k = np.clip(iq_ref_uns, -self.Imax, self.Imax)

#             iq_ref_uns_arr[k] = iq_ref_uns
#             iq_ref_clamp_arr[k] = iq_ref_k
#             iq_ref_pi_arr[k] = iq_ref_k

#             # PI-only current magnitude:
#             # below FW this is equivalent to id=0, iq=iq_ref_k
#             is_pi_arr[k] = np.abs(iq_ref_k)

#             # Convert speed-loop q-current request to torque command
#             torque_ref_k = self.torque_from_iq_command(self.machine_type, iq_ref_k, phi_fk)
#             torque_ref_k = self.apply_rated_power_limit(torque_ref_k, omega_mk)
#             torque_cmd_arr[k] = torque_ref_k

#             # ---------------------------------------------------------
#             # 2) MTPA-only current references (before FW logic)
#             # ---------------------------------------------------------
#             id_mtpa_k, iq_mtpa_k = self.pmsm_torque_to_current_refs(
#                 machine_type=self.machine_type,
#                 torque_ref=torque_ref_k,
#                 phi_f=phi_fk,
#                 mtpa_lut=self.mtpa_lut,
#             )

#             id_mtpa_arr[k] = id_mtpa_k
#             iq_mtpa_arr[k] = iq_mtpa_k
#             is_mtpa_arr[k] = np.hypot(id_mtpa_k, iq_mtpa_k)

#             # ---------------------------------------------------------
#             # 3) Final references after MTPA + field weakening
#             # ---------------------------------------------------------
#             id_ref_k, iq_ref_k_fw, omega_base_e_k, omega_max_e_k, max_speed_reached_k = \
#                 self.current_references_with_fw(self.machine_type, omega_mk, torque_ref_k, phi_fk, self.mtpa_lut)

#             id_ref_arr[k] = id_ref_k
#             iq_ref_arr[k] = iq_ref_k_fw

#             omega_base_arr[k] = omega_base_e_k / self.polepairs
#             omega_max_arr[k] = omega_max_e_k / self.polepairs
#             max_speed_reached_arr[k] = max_speed_reached_k

#             # ---- current-dependent inductances and gains ----
#             # temp ignored
#             # phi_fk = float(self.phi_f_of_T(T_m[k]))
#             # phi_fk = self.phi_f_ref
#             R_sk = self.R_s_of_T(T_s[k])



#             Ld_now, Lq_now = self.motor_inductances(self.machine_type, i_dk, i_qk, self.mtpa_lut)

#             Ld_arr[k] = Ld_now
#             Lq_arr[k] = Lq_now

#             omega_e_k = self.polepairs * omega_mk  # rotor electrical speed
#             ed_cross_arr[k] = -omega_e_k * Lq_now * i_qk
#             eq_backemf_arr[k] = omega_e_k * (Ld_now * i_dk + phi_fk)
#             epm_arr[k] = omega_e_k * phi_fk

#             P_cu_arr[k] = 1.5 * R_sk * (i_d[k] ** 2 + i_q[k] ** 2)

#             if self.include_iron_loss:
#                 lambda_d_k = Ld_now * i_dk + phi_fk
#                 lambda_q_k = Lq_now * i_qk
#                 psi_s_mag_k = float(np.hypot(lambda_d_k, lambda_q_k))
#                 lambda_rated_k = float(np.hypot(self.phi_f_ref, self.Lq0 * self.I_rated))
#                 B_core_arr[k] = self.B_iron_rated * psi_s_mag_k / max(lambda_rated_k, 1e-12)
#                 omega_s_k = abs(omega_e_k)
#                 P_iron_arr[k] = self.core_volume_m3 * (
#                     self.k_h_iron * (B_core_arr[k] ** self.beta_iron) * omega_s_k
#                     + self.k_e_iron * (B_core_arr[k] ** 2) * (omega_s_k ** 2)
#                 )

#                 # Core-loss branch current: Ohm's law across the back-EMF, using the
#                 # FIXED R_fe_iron calibrated in __init__ (not derived from P_iron_arr[k]
#                 # itself -- see the note there on why that independence matters).
#                 E_d_k = -omega_e_k * lambda_q_k
#                 E_q_k = omega_e_k * lambda_d_k
#                 i_Fe_d_arr[k] = E_d_k / self.R_fe_iron
#                 i_Fe_q_arr[k] = E_q_k / self.R_fe_iron

#             K_pd_now = Ld_now * self.omega_cc
#             K_pq_now = Lq_now * self.omega_cc
#             K_i_now = R_sk * self.omega_cc

#             K_pd_arr[k] = K_pd_now
#             K_pq_arr[k] = K_pq_now
#             K_i_arr[k] = K_i_now

#             # ---- current loop PI ----
#             err_d = id_ref_k - i_dk
#             err_q = iq_ref_k_fw - i_qk

#             u_d_uns = K_pd_now * err_d + xint_d[k]
#             u_q_uns = K_pq_now * err_q + xint_q[k]

#             u_d_uns_arr[k] = u_d_uns
#             u_q_uns_arr[k] = u_q_uns

#             # ---- decoupling + back-EMF feedforward ----
#             vd_uns = u_d_uns - omega_frame * Lq_now * i_qk
#             vq_uns = u_q_uns + omega_frame * (Ld_now * i_dk + phi_fk)

#             vd_uns_arr[k] = vd_uns
#             vq_uns_arr[k] = vq_uns

#             v_mag_uns_arr[k] = np.sqrt(vd_uns * vd_uns + vq_uns * vq_uns)

#             # ---- inverter magnitude saturation ----
#             vd_cmd, vq_cmd = dq_voltage_limit(vd_uns, vq_uns, self.Vmax)

#             vd_cmd_arr[k] = vd_cmd
#             vq_cmd_arr[k] = vq_cmd

#             v_mag_cmd_arr[k] = np.sqrt(vd_cmd * vd_cmd + vq_cmd * vq_cmd)

#             # ------------------------------------------
#             # SVPWM + dead-time reconstruction for plots
#             # ------------------------------------------
#             if self.frame_mode == "rotor_foc":
#                 theta_frame_k = self.polepairs * theta_m[k]
#             else:
#                 theta_frame_k = self.omega_e * tk

#             inv_k = self.dq_cmd_to_inverter_output(
#                 v_d_cmd=vd_cmd,
#                 v_q_cmd=vq_cmd,
#                 theta_frame=theta_frame_k,
#                 i_d=i_dk,
#                 i_q=i_qk,
#                 Vdc=self.Vdc,
#                 T_pwm=self.T_pwm,
#                 deadtime=self.deadtime,
#                 use_deadtime=self.include_deadtime
#             )

#             va_ref_arr[k], vb_ref_arr[k], vc_ref_arr[k] = inv_k["v_abc_ref"]
#             v_offset_arr[k] = inv_k["v_offset"]

#             van_ref_arr[k], vbn_ref_arr[k], vcn_ref_arr[k] = inv_k["v_pole_ref"]
#             da_cmd_arr[k], db_cmd_arr[k], dc_cmd_arr[k] = inv_k["d_cmd"]
#             da_eff_arr[k], db_eff_arr[k], dc_eff_arr[k] = inv_k["d_eff"]

#             van_avg_arr[k], vbn_avg_arr[k], vcn_avg_arr[k] = inv_k["v_pole_avg"]
#             va_avg_arr[k], vb_avg_arr[k], vc_avg_arr[k] = inv_k["v_abc_avg"]

#             vd_applied_arr[k] = float(inv_k["v_d_applied"])
#             vq_applied_arr[k] = float(inv_k["v_q_applied"])

#         # =============================================================================
#         # PWM ZOOM WINDOW FOR PLOTTING
#         # =============================================================================
#         t_zoom_0 = max(self.t0, self.plot_pwm_zoom_start)
#         t_zoom_1 = min(self.tfinal, self.plot_pwm_zoom_end)
#         if t_zoom_1 <= t_zoom_0:
#             raise ValueError(
#                 f"Invalid PWM zoom window: start={self.plot_pwm_zoom_start}, end={self.plot_pwm_zoom_end}"
#             )

#         t_zoom = np.linspace(t_zoom_0, t_zoom_1, self.pwm_plot_samples)

#         # interpolate slow envelopes from solver time grid
#         van_ref_zoom = np.interp(t_zoom, t, van_ref_arr)
#         vbn_ref_zoom = np.interp(t_zoom, t, vbn_ref_arr)
#         vcn_ref_zoom = np.interp(t_zoom, t, vcn_ref_arr)

#         da_zoom = np.interp(t_zoom, t, da_cmd_arr)
#         db_zoom = np.interp(t_zoom, t, db_cmd_arr)
#         dc_zoom = np.interp(t_zoom, t, dc_cmd_arr)

#         # normalize pole refs for carrier comparison: m in [-1, 1]
#         ma_zoom = 2.0 * van_ref_zoom / self.Vdc
#         mb_zoom = 2.0 * vbn_ref_zoom / self.Vdc
#         mc_zoom = 2.0 * vcn_ref_zoom / self.Vdc

#         carrier_zoom = triangular_carrier_from_time(t_zoom - t_zoom_0, self.T_pwm)

#         # gate waveforms with dead-time for all 3 phases
#         gA_up, gA_lo = make_center_aligned_gates_with_deadtime(
#             t_zoom=t_zoom - t_zoom_0,
#             duty_cmd=da_zoom,
#             T_pwm=self.T_pwm,
#             deadtime=self.deadtime if self.include_deadtime else 0.0
#         )

#         gB_up, gB_lo = make_center_aligned_gates_with_deadtime(
#             t_zoom=t_zoom - t_zoom_0,
#             duty_cmd=db_zoom,
#             T_pwm=self.T_pwm,
#             deadtime=self.deadtime if self.include_deadtime else 0.0
#         )

#         gC_up, gC_lo = make_center_aligned_gates_with_deadtime(
#             t_zoom=t_zoom - t_zoom_0,
#             duty_cmd=dc_zoom,
#             T_pwm=self.T_pwm,
#             deadtime=self.deadtime if self.include_deadtime else 0.0
#         )

#         # =============================================================================
#         # NUMERICAL SUMMARY: PI CURRENT vs MTPA CURRENT
#         # =============================================================================
#         # pi_mtpa_summary = [
#         #     "--- PI current vs MTPA current summary ---",
#         #     f"machine_type = {self.machine_type}",
#         #     f"Max |i_q(PI before MTPA)|       = {np.max(np.abs(iq_ref_pi_arr)):.6f} A",
#         #     f"Max |i_q(after MTPA)|          = {np.max(np.abs(iq_mtpa_arr)):.6f} A",
#         #     f"Max |i_d(after MTPA)|          = {np.max(np.abs(id_mtpa_arr)):.6f} A",
#         #     f"Max |I_s|(PI-only)             = {np.max(is_pi_arr):.6f} A",
#         #     f"Max |I_s|(after MTPA)          = {np.max(is_mtpa_arr):.6f} A",
#         #     f"Max reduction in |I_s| by MTPA = {np.max(is_pi_arr - is_mtpa_arr):.6f} A",
#         # ]
#         pi_mtpa_summary = [
#             "--- Current-reference comparison summary ---",
#             f"machine_type = {self.machine_type}",
#             f"use_mtpa = {self.use_mtpa}",
#             f"use_field_weakening = {self.use_field_weakening}",
#             f"Max |i_q(speed PI)|              = {np.max(np.abs(iq_ref_pi_arr)):.6f} A",
#             f"Max |i_q(before FW)|             = {np.max(np.abs(iq_mtpa_arr)):.6f} A",
#             f"Max |i_d(before FW)|             = {np.max(np.abs(id_mtpa_arr)):.6f} A",
#             f"Max |I_s|(speed PI)              = {np.max(is_pi_arr):.6f} A",
#             f"Max |I_s|(before FW)             = {np.max(is_mtpa_arr):.6f} A",
#         ]

#         # figures = {
#         #     "speed_time": make_xy_plot(
#         #         t,
#         #         [(speed_rpm, "Speed (rpm)"), (speed_ref_rpm, "Speed ref (rpm)")],
#         #         "Speed-Time Development",
#         #         "Time (s)",
#         #         "Speed (rpm)",
#         #     ),
#         #     "torque_time": make_xy_plot(
#         #         t,
#         #         [(Torque, "Torque (Nm)"), (T_load_arr, "Load torque (Nm)")],
#         #         "Torque-Time Development",
#         #         "Time (s)",
#         #         "Torque (Nm)",
#         #     ),
#         #     "phase_currents": make_xy_plot(
#         #         t,
#         #         [(i_phase_a, "i_a"), (i_phase_b, "i_b"), (i_phase_c, "i_c")],
#         #         "Phase Currents (abc)",
#         #         "Time (s)",
#         #         "Current (A)",
#         #     ),
#         #     "torque_speed": make_xy_plot(
#         #         speed_rpm,
#         #         [(Torque, "Torque–Speed")],
#         #         "Torque–Speed Characteristic",
#         #         "Speed (rpm)",
#         #         "Torque (Nm)",
#         #     ),
#         #     "id_tracking": make_xy_plot(
#         #         t,
#         #         [(i_d, "i_d"), (id_ref_arr, "i_d_ref")],
#         #         "d-axis current tracking",
#         #         "Time (s)",
#         #         "Current (A)",
#         #     ),
#         #     "iq_tracking": make_xy_plot(
#         #         t,
#         #         [(i_q, "i_q (actual)"), (iq_ref_arr, "i_q_ref (clamped)")],
#         #         "q-axis current tracking",
#         #         "Time (s)",
#         #         "Current (A)",
#         #     ),
#         #     "current_pi": make_xy_plot(
#         #         t,
#         #         [(u_d_uns_arr, "u_d (PI out)"), (u_q_uns_arr, "u_q (PI out)")],
#         #         "Current PI output (unsat)",
#         #         "Time (s)",
#         #         "Voltage (V)",
#         #     ),
#         #     "dq_voltage": make_xy_plot(
#         #         t,
#         #         [(vd_cmd_arr, "v_d (applied)"), (vq_cmd_arr, "v_q (applied)")],
#         #         "Commanded dq voltages (sat)",
#         #         "Time (s)",
#         #         "Voltage (V)",
#         #     ),
#         #     "speed_sat": make_xy_plot(
#         #         t,
#         #         [(iq_ref_uns_arr, "i_q_ref* (unsat)"), (iq_ref_clamp_arr, "i_q_ref (clamped)"), (i_q, "i_q (actual)")],
#         #         "Speed loop saturation",
#         #         "Time (s)",
#         #         "Current (A)",
#         #         hlines=[self.Imax, -self.Imax],
#         #     ),
#         #     "voltage_sat": make_xy_plot(
#         #         t,
#         #         [(v_mag_uns_arr, "||v*|| (request)"), (v_mag_cmd_arr, "||v|| (applied)"), (np.full_like(t, self.Vmax), "Vmax")],
#         #         "Voltage saturation",
#         #         "Time (s)",
#         #         "Voltage (V)",
#         #     ),
#         #     "pwm_compare": make_xy_plot(
#         #         t_zoom,
#         #         [(carrier_zoom, "carrier"), (ma_zoom, "m_a"), (mb_zoom, "m_b"), (mc_zoom, "m_c")],
#         #         "SVPWM carrier comparison (zoom)",
#         #         "Time (s)",
#         #         "Normalized",
#         #     ),
#         #     "switch_states": make_switch_plot(t_zoom, gA_up, gB_up, gC_up),
#         #     "thermal": make_xy_plot(
#         #         t,
#         #         [(T_s, "T_s"), (T_m, "T_m")],
#         #         "Thermal states",
#         #         "Time (s)",
#         #         "Temperature (°C)",
#         #         hlines=[self.T_warn_stator, self.T_fault_stator, self.T_warn_magnet, self.T_fault_magnet, self.T_demag],
#         #     ),
#         #     "vehicle": make_xy_plot(
#         #         t,
#         #         [(v_vehicle * 3.6, "Vehicle speed (km/h)"), (mass_arr, "Mass (kg)"), (slope_arr, "Slope (deg)"), (crr_arr, "Crr")],
#         #         "Vehicle / load schedules",
#         #         "Time (s)",
#         #         "Mixed units",
#         #     ),
#         #     # "mtpa_compare": make_xy_plot(
#         #     #     t,
#         #     #     [(iq_ref_pi_arr, "i_q from PI (before MTPA)"), (iq_mtpa_arr, "i_q after MTPA"), (iq_ref_arr, "i_q final (after MTPA + FW)")],
#         #     #     "q-axis current: PI output vs after MTPA",
#         #     #     "Time (s)",
#         #     #     "Current (A)",
#         #     # ),
#         #     "mtpa_q_compare": make_xy_plot(
#         #         t,
#         #         [(iq_ref_pi_arr, "i_q from PI (before MTPA)"),
#         #         (iq_mtpa_arr, "i_q after MTPA"),
#         #         (iq_ref_arr, "i_q final (after MTPA + FW)")],
#         #         "q-axis current: PI output vs after MTPA",
#         #         "Time (s)",
#         #         "Current (A)",
#         #     ),

#         #     "mtpa_d_compare": make_xy_plot(
#         #         t,
#         #         [(np.zeros_like(t), "i_d before MTPA (=0)"),
#         #         (id_mtpa_arr, "i_d after MTPA"),
#         #         (id_ref_arr, "i_d final (after FW)")],
#         #         "d-axis current introduced by MTPA",
#         #         "Time (s)",
#         #         "Current (A)",
#         #     ),

#         #     "mtpa_is_compare": make_xy_plot(
#         #         t,
#         #         [(is_pi_arr, "|I_s| from PI-only"),
#         #         (is_mtpa_arr, "|I_s| after MTPA")],
#         #         "Current magnitude: PI-only vs MTPA",
#         #         "Time (s)",
#         #         "Current magnitude (A)",
#         #         hlines=[self.Imax],
#         #     ),

#         #     "mtpa_torque_cmd": make_xy_plot(
#         #         t,
#         #         [(torque_cmd_arr, "Torque command")],
#         #         "Torque command produced from speed PI",
#         #         "Time (s)",
#         #         "Torque command (Nm)",
#         #     ),
#         # }

        
#         print("max |iq_ref_uns| =", np.max(np.abs(iq_ref_uns_arr)))
#         print("max |iq_ref_clamp| =", np.max(np.abs(iq_ref_clamp_arr)))
#         print("max |Torque| =", np.max(np.abs(Torque)))
#         print("max |T_load| =", np.max(np.abs(T_load_arr)))
#         print("max v_mag_uns =", np.max(v_mag_uns_arr))
#         print("Vmax =", self.Vmax)
#         print("max torque_cmd =", np.max(np.abs(torque_cmd_arr)))

#         # =================================================================================
#         # Energy balance check (validation aid, not part of the control physics)
#         # ---------------------------------------------------------------------------------
#         # Over the run [0, t[-1]]:
#         #   E_elec_in  = integral of 1.5*(v_d*i_d_terminal + v_q*i_q_terminal) dt, where
#         #                i_*_terminal = i_d/i_q (torque-producing current) PLUS i_Fe_d/
#         #                i_Fe_q (the core-loss branch current from the fixed R_fe_iron
#         #                calibrated in __init__ -- 0 if include_iron_loss is False, so
#         #                this reduces to the plain 1.5*(v_d*i_d + v_q*i_q) exactly as
#         #                before whenever iron loss is off).
#         #   E_mech     = integral of Torque * omega_m dt          (electromagnetic energy
#         #                delivered to the shaft -- covers both rotor KE change and any
#         #                load work; it does not need to be split further for this check)
#         #   E_cu       = integral of P_cu_arr dt                  (copper/I^2R loss energy)
#         #   E_iron     = integral of P_iron_arr dt                (iron/core loss energy,
#         #                0 J if include_iron_loss is False) -- this is the Steinmetz/
#         #                thermal estimate, independent of the R_fe/i_Fe electrical draw
#         #                above (see E_iron_electrical below for the comparison).
#         #   dE_mag     = change in stored dq inductor coenergy, 0.5*(Ld*i_d^2 + Lq*i_q^2),
#         #                from t=0 to t=t[-1] (small, but included so the residual reflects
#         #                real unmodeled effects rather than this known/expected term)
#         #
#         # Ideally: E_elec_in == E_mech + E_cu + E_iron + dE_mag
#         # The leftover (E_elec_in - the rest) is reported as the residual, in both J and
#         # as a % of E_elec_in, as a sanity/validation indicator -- not a strict physics
#         # guarantee (see caveats in the printed message for open-loop / dead-time cases).
#         # =================================================================================
#         # Must mirror the ODE's actual branch exactly (see "DRIVE MODE SECTION" above):
#         # v_d/v_q come from the SVPWM reconstruction whenever use_svpwm is True --
#         # include_deadtime only controls whether dead-time distortion is added INSIDE
#         # that reconstruction, it does not gate whether SVPWM averaging happens at all.
#         # (An earlier version of this check also required include_deadtime here, which
#         # compared the wrong voltage -- pre-SVPWM commanded vs. actually-applied -- for
#         # use_svpwm=True/include_deadtime=False and produced a spurious ~30% residual.)
#         if self.drive_mode == "closed_loop_foc" and self.use_svpwm:
#             v_d_used = vd_applied_arr
#             v_q_used = vq_applied_arr
#             v_source_note = "applied dq voltage (after SVPWM" + (" + dead-time)" if self.include_deadtime else ")")
#         else:
#             v_d_used = vd_cmd_arr
#             v_q_used = vq_cmd_arr
#             v_source_note = "commanded dq voltage"

#         # Terminal current actually drawn from the supply = torque-producing current
#         # (i_d, i_q) + core-loss branch current (i_Fe_d_arr, i_Fe_q_arr -- zero unless
#         # include_iron_loss is True). See __init__ for the R_fe_iron calibration and the
#         # postprocessing loop above for how i_Fe_*_arr is computed each step.
#         i_d_terminal_arr = i_d + i_Fe_d_arr
#         i_q_terminal_arr = i_q + i_Fe_q_arr

#         P_elec_in_arr = 1.5 * (v_d_used * i_d_terminal_arr + v_q_used * i_q_terminal_arr)
#         P_mech_arr = Torque * omega_m
#         # Independent cross-check: power actually dissipated in the R_fe branch itself,
#         # P = I^2*R = 1.5*R_fe_iron*(i_Fe_d^2+i_Fe_q^2) (equivalent to E_backemf * i_Fe,
#         # since i_Fe = E_backemf/R_fe_iron by construction), as opposed to E_iron below,
#         # which is the separate Steinmetz/thermal estimate. These are two different
#         # models of the same physical loss -- they should track each other reasonably
#         # closely if both are implemented correctly, but are not defined to be
#         # identical (see the R_fe_iron calibration note in __init__).
#         # (R_fe_iron is +inf when iron loss is off, and i_Fe_*_arr are all zero then --
#         # guard against 0 * inf = nan and just report zero.)
#         if self.include_iron_loss:
#             P_iron_electrical_arr = 1.5 * self.R_fe_iron * (i_Fe_d_arr ** 2 + i_Fe_q_arr ** 2)
#         else:
#             P_iron_electrical_arr = np.zeros_like(t)

#         E_elec_in = float(_trapz(P_elec_in_arr, t))
#         E_mech = float(_trapz(P_mech_arr, t))
#         E_cu = float(_trapz(P_cu_arr, t))
#         E_iron = float(_trapz(P_iron_arr, t))
#         E_iron_electrical = float(_trapz(P_iron_electrical_arr, t))

#         W_mag_0 = 0.5 * (Ld_arr[0] * i_d[0] ** 2 + Lq_arr[0] * i_q[0] ** 2)
#         W_mag_f = 0.5 * (Ld_arr[-1] * i_d[-1] ** 2 + Lq_arr[-1] * i_q[-1] ** 2)
#         dE_mag = float(W_mag_f - W_mag_0)

#         E_accounted = E_mech + E_cu + E_iron + dE_mag
#         E_residual = E_elec_in - E_accounted
#         E_residual_pct = 100.0 * E_residual / max(abs(E_elec_in), 1e-9)

#         if abs(E_residual_pct) < 2.0:
#             balance_verdict = f"OK -- residual is {E_residual_pct:+.2f}% of E_elec_in, within expected numerical tolerance."
#         else:
#             balance_verdict = (
#                 f"CHECK -- residual is {E_residual_pct:+.2f}% of E_elec_in, larger than the "
#                 f"~2% expected from numerical integration alone. This can be normal for "
#                 f"open_loop_abc (voltage source isn't the closed-loop dq command path used "
#                 f"here) or short/near-zero-current runs (E_elec_in close to 0 makes the % "
#                 f"noisy); otherwise it may be worth checking rtol/atol or reported settings."
#             )

#         energy_balance_lines = [
#             "Energy balance check (0 to t_final, validation aid):",
#             f"  Using {v_source_note} for E_elec_in (includes the R_fe core-loss branch "
#             f"current whenever iron loss is on).",
#             f"  E_elec_in (electrical energy in)         = {E_elec_in:12.4f} J",
#             f"  E_mech    (mechanical energy to shaft)   = {E_mech:12.4f} J",
#             f"  E_cu      (copper loss energy)           = {E_cu:12.4f} J",
#             f"  E_iron    (iron loss energy, Steinmetz)  = {E_iron:12.4f} J",
#             f"  dE_mag    (stored dq coenergy change)    = {dE_mag:12.4f} J",
#             f"  Accounted (E_mech+E_cu+E_iron+dE_mag)    = {E_accounted:12.4f} J",
#             f"  Residual  (E_elec_in - Accounted)        = {E_residual:12.4f} J  ({E_residual_pct:+.2f}%)",
#             f"  Verdict: {balance_verdict}",
#         ]

#         if self.include_iron_loss:
#             energy_balance_lines.append(
#                 f"  (E_iron_electrical, the independent R_fe-branch estimate actually "
#                 f"drawn into E_elec_in above, = {E_iron_electrical:12.4f} J -- compare "
#                 f"to the Steinmetz E_iron above; they're two separate loss models, not "
#                 f"defined to match exactly.)"
#             )

#         energy_balance = {
#             "E_elec_in_J": E_elec_in,
#             "E_mech_J": E_mech,
#             "E_cu_J": E_cu,
#             "E_iron_J": E_iron,
#             "E_iron_electrical_J": E_iron_electrical,
#             "dE_mag_J": dE_mag,
#             "E_residual_J": E_residual,
#             "E_residual_pct": E_residual_pct,
#             "voltage_source": v_source_note,
#         }

#         if stop_info["success"] and abs(t[-1] - self.tfinal) < 1e-9:
#             sim_status_line = "Simulation completed successfully."
#         elif stop_info["status"] == 1:
#             sim_status_line = f"Simulation terminated by event at t = {t[-1]:.6f} s."
#         else:
#             sim_status_line = (
#                 f"Simulation stopped before tfinal. "
#                 f"Last time = {t[-1]:.6f} s. "
#                 f"Solver status = {stop_info['status']}. "
#                 f"Message: {stop_info['message']}"
#             )

#         summary_lines = [
#             sim_status_line,
#             "",
#             f"Machine type: {self.machine_type}",
#             f"Drive mode: {self.drive_mode}",
#             f"Frame mode: {self.frame_mode}",
#             # f"Region: {self.region}",
#             f"Requested tfinal: {self.tfinal} s",
#             f"Returned final time: {t[-1]:.6f} s",
#             f"Solver success: {stop_info['success']}",
#             f"Solver status: {stop_info['status']}",
#             f"Solver message: {stop_info['message']}",
#             # f"tfinal: {self.tfinal} s",
#             f"Vdc: {self.Vdc} V",
#             f"Imax: {self.Imax} A",
#             f"poles: {self.poles}",
#             f"f_pwm: {self.f_pwm} Hz",
#             f"Peak speed: {np.max(np.abs(speed_rpm)):.3f} rpm",
#             f"Peak torque: {np.max(np.abs(Torque)):.3f} Nm",
#             # temp ignored
#             f"Peak stator temp: {max_Ts:.3f} °C",
#             f"Peak magnet temp: {max_Tm:.3f} °C",
#             "",
#             max_speed_message,
#             "",
#             # temp ignored
#             # "Thermal / event messages:",
#             # *event_messages,
#             # *warning_messages,
#             "Event messages:",
#             *event_messages,
#             "",
#             *pi_mtpa_summary,
#             "",
#             *energy_balance_lines,
#         ]

#         return {
#             "summary": "\n".join(summary_lines),
#             # "figures": figures,
#             "time": t,
#             "energy_balance": energy_balance,
#             "signals": {
#                 "t": t,
#                 "Vmax": self.Vmax,
#                 "i_d": i_d,
#                 "i_q": i_q,
#                 "omega_m": omega_m,
#                 "theta_m": theta_m,
#                 "Torque": Torque,
#                 "speed_rpm": speed_rpm,
#                 "speed_ref_rpm": speed_ref_rpm,
#                 # temp ignored
#                 "T_s": T_s,
#                 "T_m": T_m,
#                 "P_cu_arr": P_cu_arr,
#                 "P_iron_arr": P_iron_arr,
#                 "B_core_arr": B_core_arr,
#                 "id_ref_arr": id_ref_arr,
#                 "iq_ref_arr": iq_ref_arr,
#                 "iq_ref_pi_arr": iq_ref_pi_arr,
#                 "id_mtpa_arr": id_mtpa_arr,
#                 "iq_mtpa_arr": iq_mtpa_arr,
#                 "is_pi_arr": is_pi_arr,
#                 "is_mtpa_arr": is_mtpa_arr,
#                 "torque_cmd_arr": torque_cmd_arr,
#                 "vd_uns_arr": vd_uns_arr,
#                 "vq_uns_arr": vq_uns_arr,
#                 "vd_cmd_arr": vd_cmd_arr,
#                 "vq_cmd_arr": vq_cmd_arr,
#                 "v_mag_uns_arr": v_mag_uns_arr,
#                 "v_mag_cmd_arr": v_mag_cmd_arr,
#                 "omega_base_arr": omega_base_arr,
#                 "omega_max_arr": omega_max_arr,
#                 "max_speed_reached_arr": max_speed_reached_arr,
#                 "i_phase_a": i_phase_a,
#                 "i_phase_b": i_phase_b,
#                 "i_phase_c": i_phase_c,
#                 "t_zoom": t_zoom,
#                 "carrier_zoom": carrier_zoom,
#                 "ma_zoom": ma_zoom,
#                 "mb_zoom": mb_zoom,
#                 "mc_zoom": mc_zoom,
#                 "gA_up": gA_up,
#                 "gB_up": gB_up,
#                 "gC_up": gC_up,
#                 "gA_lo": gA_lo,
#                 "gB_lo": gB_lo,
#                 "gC_lo": gC_lo,
#                 "T_load_arr": T_load_arr,
#                 "v_vehicle": v_vehicle,
#                 "mass_arr": mass_arr,
#                 "slope_arr": slope_arr,
#                 "crr_arr": crr_arr,
#                 "va_ref_arr": va_ref_arr,
#                 "vb_ref_arr": vb_ref_arr,
#                 "vc_ref_arr": vc_ref_arr,
#                 "v_offset_arr": v_offset_arr,
#                 "van_ref_arr": van_ref_arr,
#                 "vbn_ref_arr": vbn_ref_arr,
#                 "vcn_ref_arr": vcn_ref_arr,
#                 "da_cmd_arr": da_cmd_arr,
#                 "db_cmd_arr": db_cmd_arr,
#                 "dc_cmd_arr": dc_cmd_arr,
#                 "da_eff_arr": da_eff_arr,
#                 "db_eff_arr": db_eff_arr,
#                 "dc_eff_arr": dc_eff_arr,
#                 "van_avg_arr": van_avg_arr,
#                 "vbn_avg_arr": vbn_avg_arr,
#                 "vcn_avg_arr": vcn_avg_arr,
#                 "va_avg_arr": va_avg_arr,
#                 "vb_avg_arr": vb_avg_arr,
#                 "vc_avg_arr": vc_avg_arr,
#                 # "sol_status": sol.status,
#                 # "sol_message": sol.message,
#                 "sol_success": stop_info["success"],
#                 "sol_status": stop_info["status"],
#                 "sol_message": stop_info["message"],
#                 "sol_tfinal_returned": t[-1],
#                 # "solver_breakpoints": self.get_schedule_breakpoints(),
#                 "event_messages": event_messages,
#                 # "warning_messages": warning_messages,
#                 # "sol_success": sol.success,
#                 # "sol_tfinal_returned": sol.t[-1],
#                 "omega_ref_f_arr": omega_ref_f_arr,
#                 "mass_f_arr": mass_f_arr,
#                 "slope_f_arr": slope_f_arr,
#                 "crr_f_arr": crr_f_arr,
#                 "u_d_uns_arr": u_d_uns_arr,
#                 "u_q_uns_arr": u_q_uns_arr,
#                 "iq_ref_uns_arr": iq_ref_uns_arr,
#                 "iq_ref_clamp_arr": iq_ref_clamp_arr,
#                 "ed_cross_arr": ed_cross_arr,
#                 "eq_backemf_arr": eq_backemf_arr,
#                 "epm_arr": epm_arr,
#                 "Ld_arr": Ld_arr,
#                 "Lq_arr": Lq_arr,
#                 "K_pd_arr": K_pd_arr,
#                 "K_pq_arr": K_pq_arr,
#                 "K_i_arr": K_i_arr,
#                 "K_ps_arr": K_ps_arr,
#                 "K_is_arr": K_is_arr,
#                 "id_grid_help": id_grid_help,
#                 "iq_grid_help": iq_grid_help,
#                 "Ld_curve": Ld_curve,
#                 "Lq_curve": Lq_curve,
#                 "Ld0": self.Ld0,
#                 "Lq0": self.Lq0,
#                 "Ld_inf": self.Ld_inf,
#                 "Lq_inf": self.Lq_inf,
#                 "I_Ld_sat": self.I_Ld_sat,
#                 "I_Lq_sat": self.I_Lq_sat,
#                 "vd_applied_arr": vd_applied_arr,
#                 "vq_applied_arr": vq_applied_arr,
#                 # "mtpa_torque_max": None if self.mtpa_lut is None else self.mtpa_lut.torque_max,
#                 # "mtpa_n_points": None if self.mtpa_lut is None else self.mtpa_lut.n_points,
#                 # "mtpa_id_samples": None if self.mtpa_lut is None else self.mtpa_lut.id_samples,
#             },
#         }