from __future__ import annotations

import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import ScalarFormatter

from pmsm_backend.runner import run_pmsm_simulation

from pmsm_gui.help_text import PARAMETER_HELP
from pmsm_gui.settings import (
    DEFAULTS,
    MAIN_FIELDS,
    ADV_FIELDS,
    FieldSpec,
    rad_s_to_rpm,
    rpm_to_rad_s,
)
from pmsm_gui.widgets import ScrollableFrame, ScheduleEditor, InfoTooltip

import matplotlib
# matplotlib.rcParams.update({
#     "lines.linewidth":      2.8,
#     "font.size":            16,
#     "axes.titlesize":       28,
#     "axes.labelsize":       20,
#     "xtick.labelsize":      16,
#     "ytick.labelsize":      16,
#     "legend.fontsize":      16,
#     "legend.title_fontsize":16,
#     "mathtext.fontset":     "cm",   # Computer-Modern-style math rendering for $...$ titles/labels
# })
matplotlib.rcParams.update({
    "lines.linewidth":      2.8,
    "font.size":            20,
    "axes.titlesize":       28,
    "axes.labelsize":       26,
    "xtick.labelsize":      20,
    "ytick.labelsize":      20,
    "legend.fontsize":      20,
    "legend.title_fontsize":20,
    "mathtext.fontset":     "cm",   # Computer-Modern-style math rendering for $...$ titles/labels
})

# ---------------------------------------------------------------------------
# LaTeX-style plot text helpers
# ---------------------------------------------------------------------------
# All plot titles, axis labels, legend entries and tick labels in this file
# are rendered through Matplotlib's built-in "mathtext" engine (mathtext.fontset
# = "cm" above), NOT an external LaTeX installation (matplotlib's text.usetex).
# That gives genuine LaTeX-look Computer-Modern typesetting -- upright roman
# words, italic math symbols, proper subscripts/superscripts -- with no
# dependency on a TeX distribution being installed on whatever machine runs
# this app, so a plot can never fail to render for lack of LaTeX.
#
# mathtext only renders text between a pair of "$...$" as math; anything
# outside is drawn in the plain UI font instead, which looks visibly
# different from the Computer-Modern math font. To make an ENTIRE title/
# label/legend string look consistently LaTeX-typeset, the whole string is
# wrapped in exactly one "$...$" pair, with plain English words marked
# upright via \mathrm{...} (mathtext requires literal spaces inside \mathrm
# to be escaped as "\ ", since it does not collapse whitespace the way real
# LaTeX does) and math symbols (i_d, \omega_e, T, ...) left in their normal
# italic math form. Use _word() for a words-only fragment and mtext()/mmix()
# to assemble the final single-$ string.
def _word(text: str) -> str:
    """Upright ('roman') word/phrase fragment for use inside mmix()/mtext().
    NOT wrapped in $ on its own -- combine with mmix() or call mtext() directly
    for a plain all-words string.

    Besides escaping literal spaces (mathtext doesn't collapse whitespace the
    way real LaTeX does), "-" and ":" get special handling. Inside math mode
    mathtext otherwise treats a bare "-"/":" as a binary/relational operator
    and pads it with extra spacing (e.g. "d-axis" would render as "d - axis").
    Wrapping "-" in its own braces ("{-}") is enough to make mathtext treat it
    as ordinary punctuation. ":" needs one step more -- braces alone still
    leave a gap before it -- so the surrounding \mathrm{} group is closed,
    the colon is inserted between two \! (negative thin space) commands to
    cancel the spacing on both sides, and a fresh \mathrm{} group reopens for
    the rest of the text (the braces stay balanced: this is one extra
    close/reopen pair per colon). This is only for plain-English word
    fragments; a real mathematical minus (e.g. in a raw formula fragment
    passed straight to mmix()) should keep normal operator spacing and must
    NOT go through _word()."""
    escaped = text.replace(" ", r"\ ")
    escaped = escaped.replace("-", "{-}")
    escaped = escaped.replace(":", r"}\!:\!\mathrm{")
    return r"\mathrm{" + escaped + "}"


def mtext(text: str) -> str:
    """A title/label/legend string made only of plain English words, rendered
    as one LaTeX-style (Computer-Modern, upright) mathtext expression."""
    return "$" + _word(text) + "$"


def mmix(*parts: str) -> str:
    r"""Combine raw mathtext fragments -- some from _word("..."), some bare
    LaTeX math like "i_d" or r"\omega_e" -- into ONE "$...$" expression, so
    words and symbols share the same Computer-Modern LaTeX-style font instead
    of the words falling back to the plain UI font."""
    return "$" + "".join(parts) + "$"


# Plot keys whose x-axis is NOT the full-run time array `t` -- shading a
# field-weakening-vs-time background on these would be meaningless (or, for
# pwm_compare/switch_states, is plotted against a tiny zoomed-in PWM-period
# window t_zoom, not the run's actual timescale) or wrong (torque_speed's
# x-axis is speed, inductance_model's is current), so
# shade_field_weakening_background() is skipped for these from show_plot()'s
# caller.
FW_SHADE_EXCLUDED_KEYS = {"torque_speed", "pwm_compare", "switch_states", "inductance_model"}


def shade_field_weakening_background(ax, t, s: Dict[str, Any], run_use_fw: bool, run_machine_type: str) -> None:
    """
    Shades a light purple background over every stretch of the run where
    field weakening was ACTUALLY engaged, on any time-vs-t plot -- so it's
    visible directly on the plot which portions of the run were MTPA/base
    current (unshaded) vs. field-weakened (shaded), matching what
    compare_with_motulator.py's ipmsm_current_vs_time.pdf already does for
    the motulator comparison, now added to the app's own live plots too.

    "Field weakening was engaged" here means BOTH Regime B (the
    voltage-limited-only FwLut branch) and Regime C (the current-AND-voltage
    -limited fallback, see pmsm_backend/fw_lut.py's FwLut class docstring
    for the full A/B/C regime breakdown) -- this is ONE shade covering both,
    not two different colors for B vs. C, since from current_references_
    with_fw()'s point of view both are simply "omega_r_e > omega_base, FW
    branch taken" -- the B-vs-C choice inside that is an internal detail of
    HOW the FW current was computed (whether the full Imax was needed or
    not), not a different externally-visible regime the way MTPA-vs-FW is.
    (If a B/C split is wanted too, it's derivable from the same run's
    signals -- compare hypot(id_ref_arr, iq_ref_arr) against Imax: pinned at
    Imax = Regime C, strictly less = Regime B -- just ask and I'll add a
    second shade for it.)

    Uses the run's own omega_base_arr (mechanical rad/s, already computed
    once per timestep inside current_references_with_fw() itself -- see
    pmsm_backend/simulation.py -- and already accounting for the
    MTPA-point-dependent base speed and any Ld/Lq saturation) compared
    directly against the actual speed_rpm trace, so this is the EXACT same
    condition the controller used to decide MTPA vs. FW at every instant,
    not a separately re-derived approximation that could disagree with it.

    No-ops for SPMSM (and whenever field weakening wasn't enabled for this
    run): current_references_with_fw() has a hard `machine_type == "spmsm"`
    gate that always skips the FW branch and returns the base (MTPA-only)
    current regardless of speed or the use_field_weakening checkbox -- see
    the "no field weakening" branch near the top of that function -- so
    field weakening is never actually engaged for an SPMSM run in this
    codebase, and shading would be misleading (there IS no FW region to
    show).
    """
    if not run_use_fw or run_machine_type != "ipmsm":
        return
    if "omega_base_arr" not in s or "speed_rpm" not in s:
        return

    omega_base_rpm_arr = np.asarray(s["omega_base_arr"]) * 60.0 / (2.0 * np.pi)
    speed_rpm_arr = np.asarray(s["speed_rpm"])
    n = min(len(t), len(omega_base_rpm_arr), len(speed_rpm_arr))
    if n < 2:
        return
    t = np.asarray(t)[:n]
    mask = speed_rpm_arr[:n] > omega_base_rpm_arr[:n]
    if not mask.any():
        return

    # Shade every contiguous True run, not just one span from the first True
    # to the last True -- speed can dip back under the (current-dependent)
    # base speed and re-cross it more than once, especially right around
    # the crossover or under noisy/oscillating references, and a single
    # min-to-max span would incorrectly paint those below-base-speed gaps
    # as field-weakened too.
    edges = np.diff(mask.astype(int))
    starts = list(t[1:][edges == 1])
    ends = list(t[1:][edges == -1])
    if mask[0]:
        starts = [t[0]] + starts
    if mask[-1]:
        ends = ends + [t[-1]]
    for i, (start, end) in enumerate(zip(starts, ends)):
        ax.axvspan(
            start, end, color="tab:purple", alpha=0.10, lw=0,
            label=(mtext("Field weakening active (above base speed)") if i == 0 else None),
        )


class LatexScalarFormatter(ScalarFormatter):
    """A ScalarFormatter whose tick-label text is wrapped in $...$ so the
    numbers along every axis render through mathtext (Computer-Modern),
    matching the titles/axis labels/legends above instead of falling back to
    the plain UI font. It reuses ScalarFormatter's own formatting logic
    (trimmed decimals, the shared "x10^n" offset/exponent text for very
    large/small ranges) and only adds the $ wrapping -- useMathText=True
    (defaulted on here) also makes that offset/exponent text itself render
    through mathtext instead of a plain "1e+03"-style string."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("useMathText", True)
        super().__init__(*args, **kwargs)

    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        if s.startswith("$") and s.endswith("$"):
            return s
        return f"${s}$"


class SimulationApp(tk.Tk):
    '''
    Purpose:
        Main application window for configuring, validating, running,
        and plotting the PMSM/IPMSM simulation.

    Responsibilities:
        - Builds the full GUI layout
        - Stores field variables
        - Opens schedule editors
        - Validates form inputs
        - Starts the simulation
        - Displays summary text and enables plot buttons

    Output:
        Creates the main Tkinter application object.
        Runtime simulation results are stored in self.last_result.
    '''

    def __init__(self):
        '''
        Purpose:
            Initializes the main application state, widget containers,
            schedule data, and GUI layout.

        Output:
            No return value. Creates the application window and widgets.
        '''
        super().__init__()
        self.title("SPMSM/IPMSM Simulation Control Panel")
        self.geometry("1450x950")
        self.minsize(1200, 780)

        self.input_widgets: Dict[str, tk.Widget] = {}  

        self.variables: Dict[str, tk.Variable] = {}
        self.error_labels: Dict[str, ttk.Label] = {}
        self.field_map: Dict[str, FieldSpec] = {f.key: f for f in MAIN_FIELDS + ADV_FIELDS}
        self.schedules: Dict[str, List[float]] = {
            "t_steps": list(DEFAULTS["t_steps"]),
            "omega_steps": list(DEFAULTS["omega_steps"]),
            "mass_step_times": list(DEFAULTS["mass_step_times"]),
            "mass_step_values": list(DEFAULTS["mass_step_values"]),
            "slope_step_times": list(DEFAULTS["slope_step_times"]),
            "slope_step_values_deg": list(DEFAULTS["slope_step_values_deg"]),
            "crr_step_times": list(DEFAULTS["crr_step_times"]),
            "crr_step_values": list(DEFAULTS["crr_step_values"]),
        }
        self.schedule_labels: Dict[str, ttk.Label] = {}
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_params: Optional[Dict[str, Any]] = None
        self.plot_buttons: Dict[str, ttk.Button] = {}
        self.advanced_visible = False
        
        self.inductance_helper_label: Optional[ttk.Label] = None
        self.inductance_helper_visible = False
        self.inductance_helper_box: Optional[ttk.Frame] = None
        self.inductance_helper_toggle_btn: Optional[ttk.Button] = None
        self.inductance_helper_title_label: Optional[ttk.Label] = None

        self.iron_loss_helper_label: Optional[ttk.Label] = None
        self.iron_loss_helper_visible = False
        self.iron_loss_helper_box: Optional[ttk.Frame] = None
        self.iron_loss_helper_toggle_btn: Optional[ttk.Button] = None
        self.iron_loss_helper_title_label: Optional[ttk.Label] = None

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        self.build_layout()
        self.toggle_inductance_helper()
        self.toggle_iron_loss_helper()
        self.toggle_advanced_settings()
        self.reset_to_defaults()

        for key in [
            "machine_type",
            "L_ls",
            "L_A",
            "L_B",
            "I_rated",
            "Ld_inf_factor",
            "Lq_inf_factor",
            "I_Ld_sat_factor",
            "I_Lq_sat_factor",
        ]:
            self.variables[key].trace_add("write", self.update_inductance_helper)

        for key in [
            "machine_type",
            "L_ls",
            "L_A",
            "L_B",
            "I_rated",
            "phi_f_ref",
            "include_iron_loss",
            "k_h_iron",
            "k_e_iron",
            "beta_iron",
            "core_volume_m3",
            "B_iron_rated",
        ]:
            self.variables[key].trace_add("write", self.update_iron_loss_helper)

        for key in [
            "machine_type",
            "drive_mode",
            "frame_mode",
            "use_svpwm",
            "include_deadtime",
            "use_mtpa",
            "use_field_weakening",
        ]:
            self.variables[key].trace_add("write", self.update_compatibility_state)

        self.update_compatibility_state()
        self.update_inductance_helper()
        self.update_iron_loss_helper()

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------
    def build_layout(self) -> None:
        '''
        Purpose:
            Builds the full two-panel application layout.

        Output:
            No return value. Creates the left settings panel and right action panel.
        '''
        left = ScrollableFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left_panel = left.inner
        self.left_panel.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.columnconfigure(0, weight=1)
        self.right_panel = right

        self.build_main_settings(self.left_panel)
        self.build_advanced_settings(self.left_panel)
        self.build_schedule_section(self.left_panel)
        self.build_action_panel(right)

    # def build_main_settings(self, parent: ttk.Frame) -> None:
    #     '''
    #     Purpose:
    #         Creates the "Main settings" section and populates it
    #         using MAIN_FIELDS definitions.

    #     Inputs:
    #         parent: frame that will contain this section

    #     Output:
    #         No return value. Adds widgets to the GUI.
    #     '''
    #     box = ttk.LabelFrame(parent, text="Main settings", padding=12)
    #     box.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
    #     box.columnconfigure(1, weight=1)
    #     row = 0
    #     for spec in MAIN_FIELDS:
    #         row = self.add_field(box, spec, row)

    def build_main_settings(self, parent: ttk.Frame) -> None:
        '''
        Purpose:
            Creates the "Main settings" section and populates it
            using MAIN_FIELDS definitions.

        Inputs:
            parent: frame that will contain this section

        Output:
            No return value. Adds widgets to the GUI.
        '''
        box = ttk.LabelFrame(parent, text="Main settings", padding=12)
        box.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        # Single-column stacked layout (label | widget), one field per row.
        # Previously the four checkboxes (use_svpwm/include_deadtime and
        # use_mtpa/use_field_weakening) were placed side by side in a wider
        # 5-column grid (columns 0-4). On narrower screens/windows that
        # forced the "Main settings" box wider than the visible area, so
        # the right-hand checkboxes (include_deadtime, use_field_weakening)
        # and the second radio option in "Drive mode" ended up off-screen
        # and unreachable. Stacking everything in one column -- the same
        # pattern already used successfully for machine_type/drive_mode/
        # frame_mode and for every field in Advanced settings -- keeps the
        # box's minimum width small and every control reachable regardless
        # of window/screen width.
        box.columnconfigure(1, weight=1)

        row = 0
        for spec in MAIN_FIELDS:
            row = self.add_field(box, spec, row)

    def build_consistency_messages(self, parsed: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        """
        Returns:
            errors   : blocking issues
            warnings : non-blocking but important physical consistency warnings
            infos    : explanatory messages to help the user understand the model
        """
        errors: List[str] = []
        warnings: List[str] = []
        infos: List[str] = []

        # -----------------------------
        # Basic derived quantities
        # -----------------------------
        poles = int(parsed["poles"])
        polepairs = poles // 2
        Vdc = float(parsed["Vdc"])
        Vmax = Vdc / np.sqrt(3.0)

        Rs = float(parsed["R_s_ref"])
        phi_f = float(parsed["phi_f_ref"])
        Imax = float(parsed["Imax"])
        P_rated = float(parsed["P_rated"])

        wheel_radius = float(parsed["wheel_radius"])
        gear_ratio = float(parsed["gear_ratio"])
        drivetrain_eff = float(parsed["drivetrain_eff"])

        mass0 = float(self.schedules["mass_step_values"][0]) if self.schedules["mass_step_values"] else 0.0
        omega_target = max(abs(v) for v in self.schedules["omega_steps"]) if self.schedules["omega_steps"] else 0.0

        # dq inductances from your parameterization
        Ld = float(parsed["L_ls"]) + 1.5 * (float(parsed["L_A"]) - float(parsed["L_B"]))
        if parsed["machine_type"] == "spmsm":
            Lq = Ld
        else:
            Lq = float(parsed["L_ls"]) + 1.5 * (float(parsed["L_A"]) + float(parsed["L_B"]))

        # -----------------------------
        # 1) Electrical sanity checks
        # -----------------------------
        r_drop = Rs * Imax

        if r_drop > Vmax:
            warnings.append(
                "Current-limit warning:\n"
                f"R_s_ref * Imax = {Rs:.4f} * {Imax:.2f} = {r_drop:.2f} V, "
                f"which is greater than the inverter dq voltage limit Vmax = Vdc/sqrt(3) = {Vmax:.2f} V.\n"
                "Interpretation: the model will likely become voltage-limited before it can reach the requested current. "
                "This often causes weak torque production and poor speed tracking."
            )
        elif r_drop > 0.7 * Vmax:
            warnings.append(
                "Electrical headroom warning:\n"
                f"The resistive drop estimate R_s_ref * Imax = {r_drop:.2f} V is already a large fraction of Vmax = {Vmax:.2f} V.\n"
                "There may be little voltage left for back-EMF and inductive current control, especially at higher speed."
            )

        if Ld <= 0 or Lq <= 0:
            errors.append(
                f"Inductance error:\n"
                f"Computed dq inductances must be positive, but got Ld = {Ld:.6g} H and Lq = {Lq:.6g} H.\n"
                "Check L_ls, L_A, and L_B."
            )

        # -----------------------------
        # 2) Torque / power consistency
        # -----------------------------
        Kt = 1.5 * polepairs * phi_f   # exact for SPMSM, baseline estimate for IPMSM
        T_est = Kt * Imax
        P_est_at_target = T_est * omega_target

        infos.append(
            "Estimated motor capability from entered parameters:\n"
            f"- pole pairs p = {polepairs}\n"
            f"- approximate torque constant Kt ≈ 1.5 * p * phi_f = {Kt:.4f} Nm/A\n"
            f"- approximate peak torque at Imax: T ≈ Kt * Imax = {T_est:.2f} Nm"
        )

        if P_rated > 0.0 and P_est_at_target < 0.25 * P_rated:
            warnings.append(
                "Rated-power consistency warning:\n"
                f"At the largest scheduled motor speed, the simple estimate gives only about {P_est_at_target:.1f} W, "
                f"while P_rated is set to {P_rated:.1f} W.\n"
                "This suggests that P_rated may not match the rest of the motor parameters. "
                "If the true rated power is unknown, set P_rated = 0.0 to disable the supervisory power clamp."
            )

        # -----------------------------
        # 3) Vehicle suitability check
        # -----------------------------
        if mass0 > 0.0 and wheel_radius > 0.0 and gear_ratio > 0.0:
            wheel_force_est = T_est * gear_ratio * drivetrain_eff / wheel_radius
            accel_est = wheel_force_est / mass0

            infos.append(
                f"Estimated launch capability with the first vehicle mass entry:\n"
                f"- wheel force ≈ {wheel_force_est:.1f} N\n"
                f"- ideal launch acceleration ≈ {accel_est:.3f} m/s^2"
            )

            if mass0 >= 800.0 and accel_est < 0.5:
                warnings.append(
                    "Vehicle-motor mismatch warning:\n"
                    f"With mass = {mass0:.1f} kg, wheel radius = {wheel_radius:.3f} m, "
                    f"gear ratio = {gear_ratio:.2f}, and the present motor constants, "
                    f"the estimated launch acceleration is only about {accel_est:.3f} m/s^2.\n"
                    "This is very weak for a passenger-car-scale vehicle. "
                    "The chosen vehicle profile is probably too heavy for this motor."
                )
            elif accel_est < 0.2:
                warnings.append(
                    "Low-acceleration warning:\n"
                    f"The estimated launch acceleration is only about {accel_est:.3f} m/s^2.\n"
                    "Expect slow speed response even if the controller is working correctly."
                )

        # -----------------------------
        # 4) SPMSM-specific hints
        # -----------------------------
        if parsed["machine_type"] == "spmsm":
            if abs(float(parsed["L_A"])) > 1e-12 or abs(float(parsed["L_B"])) > 1e-12:
                warnings.append(
                    "SPMSM saliency warning:\n"
                    "For a surface-mounted PMSM, the usual assumption is little or no saliency, "
                    "so Ld ≈ Lq. In this model, setting L_A = 0 and L_B = 0 removes saliency "
                    "and makes Ld = Lq. If your sample machine is meant to be a non-salient "
                    "surface PMSM, use L_A = 0 and L_B = 0."
                )

            if bool(parsed["use_mtpa"]):
                infos.append(
                    "MTPA note:\n"
                    "For an SPMSM, MTPA typically adds little or no benefit because Ld ≈ Lq. "
                    "You can disable MTPA to simplify testing."
                )

        # -----------------------------
        # 5) Detect the sample motor profile approximately
        # -----------------------------
        is_sample_motor = (
            parsed["machine_type"] == "spmsm"
            and int(parsed["poles"]) == 10
            and abs(float(parsed["phi_f_ref"]) - 0.00798) < 5e-4
            and abs(float(parsed["R_s_ref"]) - 2.015) < 0.1
            and abs(float(parsed["J_motor"]) - 4.4347e-6) < 1e-6
        )

        if is_sample_motor:
            infos.append(
                "Motor-profile note:\n"
                "These values closely match the Microchip sample PMSM parameter file. "
                "That profile is consistent with a small demo/lab motor, not a full-size EV traction motor. "
                "For physically believable results, pair it with a light vehicle or bench-scale load."
            )

        # -----------------------------
        # 6) Iron-loss consistency
        # -----------------------------
        if bool(parsed.get("include_iron_loss")):
            core_volume_m3 = float(parsed.get("core_volume_m3", 0.0))
            if core_volume_m3 <= 0.0:
                warnings.append(
                    "Iron-loss warning:\n"
                    "include_iron_loss is enabled but core_volume_m3 = 0. "
                    "Iron loss scales with this volume, so it will always compute to 0 W "
                    "until you set a nonzero core_volume_m3."
                )

            B_iron_rated = float(parsed.get("B_iron_rated", 0.0))
            if B_iron_rated > 2.2:
                warnings.append(
                    "Iron-loss warning:\n"
                    f"B_iron_rated = {B_iron_rated:.2f} T is above the typical saturation "
                    "flux density of electrical steel (~2.0-2.2 T). Consider a lower value "
                    "unless this is intentional."
                )

        return errors, warnings, infos

    def build_advanced_settings(self, parent: ttk.Frame) -> None:
            '''
            Purpose:
                Creates the "Advanced settings" section, adds all advanced fields,
                and adds the PWM zoom slider.

            Inputs:
                parent: frame that will contain this section

            Output:
                No return value. Adds widgets to the GUI.
            '''
            outer = ttk.Frame(parent)
            outer.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
            outer.columnconfigure(0, weight=1)

            header = ttk.Frame(outer)
            header.grid(row=0, column=0, sticky="ew")
            header.columnconfigure(1, weight=1)

            self.adv_toggle_btn = ttk.Button(
                header,
                text="▼",
                width=3,
                command=self.toggle_advanced_settings
            )
            self.adv_toggle_btn.grid(row=0, column=0, sticky="w")

            self.adv_title_label = ttk.Label(
                header,
                text="Advanced settings",
                font=("Segoe UI", 10, "bold"),
                cursor="hand2"
            )
            self.adv_title_label.grid(row=0, column=1, sticky="w", padx=(6, 0))
            self.adv_title_label.bind("<Button-1>", lambda e: self.toggle_advanced_settings())

            box = ttk.LabelFrame(outer, text="", padding=12)
            box.grid(row=1, column=0, sticky="ew")
            box.columnconfigure(1, weight=1)

            self.advanced_box = box

            row = 0
            for spec in ADV_FIELDS:
                row = self.add_field(box, spec, row)

                if spec.key == "I_Lq_sat_factor":
                    helper_outer = ttk.Frame(box)
                    helper_outer.grid(row=row, column=0, columnspan=3, sticky="ew", padx=4, pady=(6, 10))
                    helper_outer.columnconfigure(0, weight=1)

                    helper_header = ttk.Frame(helper_outer)
                    helper_header.grid(row=0, column=0, sticky="ew")
                    helper_header.columnconfigure(1, weight=1)

                    self.inductance_helper_toggle_btn = ttk.Button(
                        helper_header,
                        text="▼",
                        width=3,
                        command=self.toggle_inductance_helper
                    )
                    self.inductance_helper_toggle_btn.grid(row=0, column=0, sticky="w")

                    self.inductance_helper_title_label = ttk.Label(
                        helper_header,
                        text="Inductance / saturation helper",
                        font=("Segoe UI", 10, "bold"),
                        cursor="hand2"
                    )
                    self.inductance_helper_title_label.grid(row=0, column=1, sticky="w", padx=(6, 0))
                    self.inductance_helper_title_label.bind(
                        "<Button-1>",
                        lambda e: self.toggle_inductance_helper()
                    )

                    helper_box = ttk.LabelFrame(helper_outer, text="", padding=10)
                    helper_box.grid(row=1, column=0, sticky="ew")
                    helper_box.columnconfigure(0, weight=1)

                    self.inductance_helper_box = helper_box

                    self.inductance_helper_label = ttk.Label(
                        helper_box,
                        text="",
                        justify="left",
                        wraplength=820
                    )
                    self.inductance_helper_label.grid(row=0, column=0, sticky="w")

                    row += 1

                if spec.key == "B_iron_rated":
                    helper_outer = ttk.Frame(box)
                    helper_outer.grid(row=row, column=0, columnspan=3, sticky="ew", padx=4, pady=(6, 10))
                    helper_outer.columnconfigure(0, weight=1)

                    helper_header = ttk.Frame(helper_outer)
                    helper_header.grid(row=0, column=0, sticky="ew")
                    helper_header.columnconfigure(1, weight=1)

                    self.iron_loss_helper_toggle_btn = ttk.Button(
                        helper_header,
                        text="▼",
                        width=3,
                        command=self.toggle_iron_loss_helper
                    )
                    self.iron_loss_helper_toggle_btn.grid(row=0, column=0, sticky="w")

                    self.iron_loss_helper_title_label = ttk.Label(
                        helper_header,
                        text="Iron (core) loss helper",
                        font=("Segoe UI", 10, "bold"),
                        cursor="hand2"
                    )
                    self.iron_loss_helper_title_label.grid(row=0, column=1, sticky="w", padx=(6, 0))
                    self.iron_loss_helper_title_label.bind(
                        "<Button-1>",
                        lambda e: self.toggle_iron_loss_helper()
                    )

                    helper_box = ttk.LabelFrame(helper_outer, text="", padding=10)
                    helper_box.grid(row=1, column=0, sticky="ew")
                    helper_box.columnconfigure(0, weight=1)

                    self.iron_loss_helper_box = helper_box

                    self.iron_loss_helper_label = ttk.Label(
                        helper_box,
                        text="",
                        justify="left",
                        wraplength=820
                    )
                    self.iron_loss_helper_label.grid(row=0, column=0, sticky="w")

                    row += 1

            slider_row = row
            # ttk.Label(box, text="PWM zoom start [s]").grid(row=slider_row, column=0, sticky="w", padx=4, pady=4)
            self.add_label_with_info(
                box,
                slider_row,
                0,
                "PWM zoom start [s]",
                "plot_pwm_zoom_start",
                padx=4,
                pady=4,
            )
            self.pwm_zoom_var = tk.DoubleVar(value=DEFAULTS["plot_pwm_zoom_start"])
            self.pwm_zoom_scale = ttk.Scale(
                box,
                from_=0.0,
                to=max(0.0, DEFAULTS["tfinal"] - self.get_pwm_zoom_span(DEFAULTS)),
                variable=self.pwm_zoom_var,
                command=lambda _=None: self.update_pwm_zoom_label(),
            )
            self.pwm_zoom_scale.grid(row=slider_row, column=1, sticky="ew", padx=4, pady=4)
            self.pwm_zoom_value_label = ttk.Label(box, text="")
            self.pwm_zoom_value_label.grid(row=slider_row, column=2, sticky="w", padx=4)
            self.error_labels["plot_pwm_zoom_start"] = ttk.Label(box, text="", foreground="#b00020")
            self.error_labels["plot_pwm_zoom_start"].grid(row=slider_row + 1, column=1, sticky="w", padx=4)

    def toggle_advanced_settings(self) -> None:
        '''
        Purpose:
            Shows or hides the advanced settings section.

        Output:
            No return value. Updates the advanced section visibility.
        '''
        if self.advanced_visible:
            self.advanced_box.grid_remove()
            self.adv_toggle_btn.config(text="▶")
            self.advanced_visible = False
        else:
            self.advanced_box.grid()
            self.adv_toggle_btn.config(text="▼")
            self.advanced_visible = True

    def toggle_inductance_helper(self) -> None:
        if self.inductance_helper_box is None or self.inductance_helper_toggle_btn is None:
            return

        if self.inductance_helper_visible:
            self.inductance_helper_box.grid_remove()
            self.inductance_helper_toggle_btn.config(text="▶")
            self.inductance_helper_visible = False
        else:
            self.inductance_helper_box.grid()
            self.inductance_helper_toggle_btn.config(text="▼")
            self.inductance_helper_visible = True

    def toggle_iron_loss_helper(self) -> None:
        if self.iron_loss_helper_box is None or self.iron_loss_helper_toggle_btn is None:
            return

        if self.iron_loss_helper_visible:
            self.iron_loss_helper_box.grid_remove()
            self.iron_loss_helper_toggle_btn.config(text="▶")
            self.iron_loss_helper_visible = False
        else:
            self.iron_loss_helper_box.grid()
            self.iron_loss_helper_toggle_btn.config(text="▼")
            self.iron_loss_helper_visible = True

    # ----------------------- enabling/disabling (beg) ------------------------------------------
    def set_widget_enabled(self, key: str, enabled: bool) -> None:
        """
        Enable or disable one input widget.
        Works for Entry, Checkbutton, Combobox, and Frame containers with children.
        """
        widget = self.input_widgets.get(key)
        if widget is None:
            return

        state = "normal" if enabled else "disabled"

        # Combobox readonly needs special handling
        if isinstance(widget, ttk.Combobox):
            widget.configure(state="readonly" if enabled else "disabled")
            return

        # If widget is a frame, disable all children inside it.
        if isinstance(widget, ttk.Frame):
            for child in widget.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
            return

        try:
            widget.configure(state=state)
        except tk.TclError:
            pass

    def set_field_value(self, key: str, value: Any) -> None:
        """
        Safely set a GUI variable without repeatedly triggering traces
        when the value is already correct.
        """
        var = self.variables.get(key)
        if var is None:
            return

        if var.get() == value:
            return

        var.set(value)
    

    def update_compatibility_state(self, *_args) -> None:
        """
        Enable/disable settings that are not meaningful for the current configuration.
        Also forces incompatible settings to safe values.

        SVPWM off      >>  include_deadtime off, deadtime disabled, PWM zoom disabled
        SPMSM          >>  MTPA off, MTPA disabled, L_A/L_B forced to 0
        closed_loop    >>  frame_mode forced to rotor_foc
        open_loop_abc  >>  MTPA and field weakening disabled
        """
        machine_type = str(self.variables["machine_type"].get())
        drive_mode = str(self.variables["drive_mode"].get())
        frame_mode = str(self.variables["frame_mode"].get())

        use_svpwm = bool(self.variables["use_svpwm"].get())
        use_mtpa = bool(self.variables["use_mtpa"].get())
        use_field_weakening = bool(self.variables["use_field_weakening"].get())

        # ------------------------------------------------------------
        # 1) closed_loop_foc must use rotor_foc
        # ------------------------------------------------------------
        if drive_mode == "closed_loop_foc":
            self.set_field_value("frame_mode", "rotor_foc")
            self.set_widget_enabled("frame_mode", False)
        else:
            self.set_widget_enabled("frame_mode", True)

        # Refresh after possible forced value
        frame_mode = str(self.variables["frame_mode"].get())

        # ------------------------------------------------------------
        # 2) Dead-time only makes sense when SVPWM/inverter model is used
        # ------------------------------------------------------------
        if not use_svpwm:
            self.set_field_value("include_deadtime", False)
            self.set_widget_enabled("include_deadtime", False)
            self.set_widget_enabled("deadtime", False)
            self.set_widget_enabled("plot_pwm_num_periods", False)
            self.set_widget_enabled("pwm_plot_samples", False)

            try:
                self.pwm_zoom_scale.configure(state="disabled")
            except Exception:
                pass
        else:
            self.set_widget_enabled("include_deadtime", True)
            self.set_widget_enabled("plot_pwm_num_periods", True)
            self.set_widget_enabled("pwm_plot_samples", True)

            try:
                self.pwm_zoom_scale.configure(state="normal")
            except Exception:
                pass

            include_deadtime = bool(self.variables["include_deadtime"].get())
            self.set_widget_enabled("deadtime", include_deadtime)

        # ------------------------------------------------------------
        # 3) MTPA is mainly useful/logical for IPMSM
        # ------------------------------------------------------------
        if machine_type == "spmsm":
            self.set_field_value("use_mtpa", False)
            self.set_widget_enabled("use_mtpa", False)

            self.set_field_value("use_field_weakening", False)
            self.set_widget_enabled("use_field_weakening", False)

            # SPMSM should not use saliency parameters in this model
            self.set_field_value("L_A", "0.0")
            self.set_field_value("L_B", "0.0")
            self.set_widget_enabled("L_A", False)
            self.set_widget_enabled("L_B", False)

            # Saturation model is not used for MTPA on SPMSM in your backend
            self.set_widget_enabled("Ld_inf_factor", False)
            self.set_widget_enabled("Lq_inf_factor", False)
            self.set_widget_enabled("I_Ld_sat_factor", False)
            self.set_widget_enabled("I_Lq_sat_factor", False)

        else:
            self.set_widget_enabled("use_mtpa", True)

            # IPMSM can use saliency and saturation parameters
            self.set_widget_enabled("L_A", True)
            self.set_widget_enabled("L_B", True)

            # Only really needed when MTPA is enabled
            # use_mtpa = bool(self.variables["use_mtpa"].get())
            self.set_widget_enabled("Ld_inf_factor", True)
            self.set_widget_enabled("Lq_inf_factor", True)
            self.set_widget_enabled("I_Ld_sat_factor", True)
            self.set_widget_enabled("I_Lq_sat_factor", True)

        # ------------------------------------------------------------
        # 4) Field weakening is meaningful mainly in closed-loop FOC
        # ------------------------------------------------------------
        # if drive_mode != "closed_loop_foc":
        if machine_type == "spmsm" or drive_mode != "closed_loop_foc":
            self.set_field_value("use_field_weakening", False)
            self.set_widget_enabled("use_field_weakening", False)
        else:
            self.set_widget_enabled("use_field_weakening", True)

        # ------------------------------------------------------------
        # 5) Open-loop abc does not use FOC-only controls logically
        # ------------------------------------------------------------
        if drive_mode == "open_loop_abc":
            self.set_field_value("use_mtpa", False)
            self.set_field_value("use_field_weakening", False)

            self.set_widget_enabled("use_mtpa", False)
            self.set_widget_enabled("use_field_weakening", False)

        # ------------------------------------------------------------
        # 6) Update helper text and plot button availability
        # ------------------------------------------------------------
        self.update_inductance_helper()
        self.update_plot_button_states()
    # ----------------------- enabling/disabling (end) ------------------------------------------

    def build_inductance_helper_text(self) -> str:
        try:
            machine_type = str(self.variables["machine_type"].get())
            L_ls = float(self.variables["L_ls"].get())
            L_A = float(self.variables["L_A"].get())
            L_B = float(self.variables["L_B"].get())
            I_rated = float(self.variables["I_rated"].get())
            Ld_inf_factor = float(self.variables["Ld_inf_factor"].get())
            Lq_inf_factor = float(self.variables["Lq_inf_factor"].get())
            I_Ld_sat_factor = float(self.variables["I_Ld_sat_factor"].get())
            I_Lq_sat_factor = float(self.variables["I_Lq_sat_factor"].get())
        except Exception:
            return (
                "Inductance helper\n"
                "Enter valid numeric values to see the derived model."
            )

        Ld0 = L_ls + 1.5 * (L_A - L_B)
        Lq0_raw = L_ls + 1.5 * (L_A + L_B)
        Lq0 = Ld0 if machine_type == "spmsm" else Lq0_raw

        Ld_inf = Ld_inf_factor * Ld0
        Lq_inf = Lq_inf_factor * Lq0
        I_Ld_sat = I_Ld_sat_factor * I_rated
        I_Lq_sat = I_Lq_sat_factor * I_rated

        lines = [
            "Inductance helper",
            "",
            f"Ld0 = L_ls + 1.5·(L_A − L_B) = {Ld0:.6g} H",
            f"Lq0 = L_ls + 1.5·(L_A + L_B) = {Lq0:.6g} H",
            "",
            f"Ld_inf = (Ld_inf / Ld0) · Ld0 = {Ld_inf:.6g} H",
            f"Lq_inf = (Lq_inf / Lq0) · Lq0 = {Lq_inf:.6g} H",
            "",
            f"I_Ld_sat = (I_Ld_sat / I_rated) · I_rated = {I_Ld_sat:.6g} A",
            f"I_Lq_sat = (I_Lq_sat / I_rated) · I_rated = {I_Lq_sat:.6g} A",
            "",
            "Used by the backend:",
            "Ld(id) = Ld_inf + (Ld0 − Ld_inf) / (1 + |id| / I_Ld_sat)",
            "Lq(iq) = Lq_inf + (Lq0 − Lq_inf) / (1 + |iq| / I_Lq_sat)",
            "",
            "Meaning:",
            "• smaller Ld_inf / Ld0  → stronger d-axis saturation",
            "• smaller Lq_inf / Lq0  → stronger q-axis saturation",
            "• smaller I_Ld_sat / I_rated → Ld drops earlier with current",
            "• smaller I_Lq_sat / I_rated → Lq drops earlier with current",
        ]

        if machine_type == "spmsm":
            lines += [
                "",
                "SPMSM:",
                "• backend uses Lq = Ld",
                "• usually use L_A = 0 and L_B = 0",
            ]
        else:
            lines += [
                "",
                "IPMSM:",
                "• saliency comes from Ld ≠ Lq",
                "• larger Lq − Ld usually gives stronger MTPA effect",
            ]

        return "\n".join(lines)


    def update_inductance_helper(self, *_args) -> None:
        if self.inductance_helper_label is None:
            return
        self.inductance_helper_label.config(text=self.build_inductance_helper_text())

    def build_iron_loss_helper_text(self) -> str:
        try:
            machine_type = str(self.variables["machine_type"].get())
            L_ls = float(self.variables["L_ls"].get())
            L_A = float(self.variables["L_A"].get())
            L_B = float(self.variables["L_B"].get())
            I_rated = float(self.variables["I_rated"].get())
            phi_f_ref = float(self.variables["phi_f_ref"].get())
            include_iron_loss = bool(self.variables["include_iron_loss"].get())
            k_h = float(self.variables["k_h_iron"].get())
            k_e = float(self.variables["k_e_iron"].get())
            beta = float(self.variables["beta_iron"].get())
            core_volume_m3 = float(self.variables["core_volume_m3"].get())
            B_iron_rated = float(self.variables["B_iron_rated"].get())
        except Exception:
            return (
                "Iron (core) loss helper\n"
                "Enter valid numeric values to see the derived model."
            )

        Ld0 = L_ls + 1.5 * (L_A - L_B)
        Lq0_raw = L_ls + 1.5 * (L_A + L_B)
        Lq0 = Ld0 if machine_type == "spmsm" else Lq0_raw

        lambda_rated = float(np.hypot(phi_f_ref, Lq0 * I_rated))

        # Illustrative example only -- not tied to any schedule/speed field.
        omega_s_example = 1000.0  # rad/s (electrical), ~159 Hz, round number for scale
        B_example = B_iron_rated  # by construction, B_core = B_iron_rated at rated current
        p_h = k_h * (B_example ** beta) * omega_s_example
        p_e = k_e * (B_example ** 2) * (omega_s_example ** 2)
        p_iron_density = p_h + p_e
        P_iron_example = p_iron_density * core_volume_m3

        status_line = (
            "Iron loss is ACTIVE in simulation results."
            if include_iron_loss
            else "Iron loss is currently OFF (\"Include iron (core) losses\" unchecked) -- "
                 "the numbers below show what WOULD apply if it's turned on."
        )

        lines = [
            "Iron (core) loss helper",
            "",
            status_line,
            "",
            "Steinmetz model (Mi/Slemon/Bonert, IEEE Trans. Ind. Appl., 2003):",
            "p_iron(B, omega_s) = k_h * B^beta * omega_s + k_e * B^2 * omega_s^2   [W/m^3]",
            "P_iron = p_iron * core_volume_m3                                     [W]",
            "",
            f"Lq0 = L_ls + 1.5*(L_A + L_B) = {Lq0:.6g} H   (Ld0 = {Ld0:.6g} H)",
            f"lambda_rated = hypot(phi_f_ref, Lq0 * I_rated) = {lambda_rated:.6g} Wb",
            "B_core = B_iron_rated * |lambda_dq| / lambda_rated",
            "(so B_core = B_iron_rated exactly at id=0, iq=I_rated -- your rated point)",
            "",
            f"Illustrative example at B = B_iron_rated = {B_example:.4g} T and "
            f"omega_s = {omega_s_example:.0f} rad/s (~{omega_s_example / (2 * np.pi):.0f} Hz electrical, for scale only):",
            f"  hysteresis term  p_h = k_h*B^beta*omega_s        = {p_h:,.1f} W/m^3",
            f"  eddy-current term p_e = k_e*B^2*omega_s^2         = {p_e:,.1f} W/m^3",
            f"  p_iron = p_h + p_e                                = {p_iron_density:,.1f} W/m^3",
            f"  P_iron = p_iron * core_volume_m3 ({core_volume_m3:.4g} m^3)      = {P_iron_example:,.2f} W",
            "",
            "Meaning:",
            "- Larger k_h_iron / k_e_iron -> proportionally more loss (hysteresis / eddy respectively).",
            "- beta_iron near 2.0 is typical for electrical steel; higher means loss grows",
            "  faster with flux density.",
            "- core_volume_m3 scales W/m^3 -> W linearly. If it's 0, iron loss is always 0 W",
            "  even with include_iron_loss checked.",
            "- B_iron_rated sets the flux-density scale at your rated operating point",
            "  (id=0, iq=I_rated); values above roughly 2.0-2.2 T are above typical",
            "  electrical-steel saturation and are flagged in Validate.",
            "- Iron loss only feeds into the stator temperature (T_s) and the P_iron_arr /",
            "  B_core_arr signals when \"Include iron (core) losses\" is checked.",
        ]

        return "\n".join(lines)

    def update_iron_loss_helper(self, *_args) -> None:
        if self.iron_loss_helper_label is None:
            return
        self.iron_loss_helper_label.config(text=self.build_iron_loss_helper_text())

    # def enable_legend_toggle(self, fig: Figure, ax) -> None:
    #     """
    #     Allow clicking legend entries to show/hide plotted lines.
    #     """
    #     legend = ax.get_legend()
    #     if legend is None:
    #         return

    #     legend_lines = legend.get_lines()
    #     legend_texts = legend.get_texts()
    #     plot_lines = ax.get_lines()

    #     legend_map = {}

    #     # Map legend line handles to plotted lines by label
    #     for legline in legend_lines:
    #         label = legline.get_label()
    #         for pline in plot_lines:
    #             if pline.get_label() == label:
    #                 legline.set_picker(True)
    #                 legline.set_pickradius(8)
    #                 legend_map[legline] = pline
    #                 break

    #     # Also allow clicking text labels
    #     for text in legend_texts:
    #         label = text.get_text()
    #         for pline in plot_lines:
    #             if pline.get_label() == label:
    #                 text.set_picker(True)
    #                 legend_map[text] = pline
    #                 break

    #     def on_pick(event):
    #         artist = event.artist
    #         if artist not in legend_map:
    #             return
    #         origline = legend_map[artist]
    #         visible = not origline.get_visible()
    #         origline.set_visible(visible)

    #         # fade legend item when hidden
    #         if hasattr(artist, "set_alpha"):
    #             artist.set_alpha(1.0 if visible else 0.2)

    #         # keep legend line/text in sync
    #         legend = ax.get_legend()
    #         if legend:
    #             for legline in legend.get_lines():
    #                 if legline.get_label() == origline.get_label():
    #                     legline.set_alpha(1.0 if visible else 0.2)
    #             for text in legend.get_texts():
    #                 if text.get_text() == origline.get_label():
    #                     text.set_alpha(1.0 if visible else 0.2)

    #         fig.canvas.draw_idle()

    #     fig.canvas.mpl_connect("pick_event", on_pick)

    def enable_legend_toggle(self, fig: Figure, ax, extra_axes: list | None = None) -> None:
        """
        Allow clicking legend entries to show/hide plotted lines.

        `ax` is the primary axes whose legend is being used to drive the
        toggling. `extra_axes`, when given, lists twin axes (e.g. a secondary
        y-axis created with ax.twinx() sharing ax's x-axis, as in the
        "pi_gains" plot combining V/A proportional gains with a V/(A*s)
        integral gain on very different scales) whose lines were folded into
        that same combined legend. Each axis is autoscaled independently
        using only the lines it owns, so toggling a line on one axis never
        distorts the other axis's y-scale.
        """
        legend = ax.get_legend()
        if legend is None:
            return

        all_axes = [ax] + [a for a in (extra_axes or []) if a is not None]

        legend_lines = legend.get_lines()
        legend_texts = legend.get_texts()

        # label -> (line, axes that owns it) across ax and any twin axes
        line_by_label = {}
        for owner in all_axes:
            for pline in owner.get_lines():
                line_by_label[pline.get_label()] = (pline, owner)

        legend_map = {}

        for legline in legend_lines:
            label = legline.get_label()
            if label in line_by_label:
                legline.set_picker(True)
                legline.set_pickradius(8)
                legend_map[legline] = line_by_label[label]

        for text in legend_texts:
            label = text.get_text()
            if label in line_by_label:
                text.set_picker(True)
                legend_map[text] = line_by_label[label]

        def autoscale_visible_y(owner):
            visible_lines = [line for line in owner.get_lines() if line.get_visible()]
            if not visible_lines:
                fig.canvas.draw_idle()
                return

            ymins = []
            ymaxs = []

            x0, x1 = ax.get_xlim()  # shared x-axis across ax and any twin axes

            for line in visible_lines:
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)

                if x.size == 0 or y.size == 0:
                    continue

                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]

                if x.size == 0:
                    continue

                view_mask = (x >= x0) & (x <= x1)
                if np.any(view_mask):
                    y_view = y[view_mask]
                else:
                    y_view = y

                if y_view.size == 0:
                    continue

                ymins.append(np.min(y_view))
                ymaxs.append(np.max(y_view))

            if not ymins or not ymaxs:
                fig.canvas.draw_idle()
                return

            y_min = min(ymins)
            y_max = max(ymaxs)

            if np.isclose(y_min, y_max):
                pad = max(1.0, 0.05 * abs(y_min) + 1e-6)
            else:
                pad = 0.08 * (y_max - y_min)

            owner.set_ylim(y_min - pad, y_max + pad)

        def on_pick(event):
            artist = event.artist
            if artist not in legend_map:
                return

            origline, owner = legend_map[artist]
            visible = not origline.get_visible()
            origline.set_visible(visible)

            if hasattr(artist, "set_alpha"):
                artist.set_alpha(1.0 if visible else 0.2)

            legend = ax.get_legend()
            if legend:
                for legline in legend.get_lines():
                    if legline.get_label() == origline.get_label():
                        legline.set_alpha(1.0 if visible else 0.2)
                for text in legend.get_texts():
                    if text.get_text() == origline.get_label():
                        text.set_alpha(1.0 if visible else 0.2)

            autoscale_visible_y(owner)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)


    def build_schedule_section(self, parent: ttk.Frame) -> None:
        '''
        Purpose:
            Creates the step-schedule section and adds buttons
            to edit each schedule type.

        Inputs:
            parent: frame that will contain this section

        Output:
            No return value. Adds widgets to the GUI.
        '''
        box = ttk.LabelFrame(parent, text="Step schedules", padding=12)
        box.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        for col in range(3):
            box.columnconfigure(col, weight=1 if col == 1 else 0)

        schedule_specs = [
            ("Speed steps", "t_steps", "omega_steps", "Edit", self.edit_speed_schedule),
            ("Mass steps", "mass_step_times", "mass_step_values", "Edit", self.edit_mass_schedule),
            ("Slope steps", "slope_step_times", "slope_step_values_deg", "Edit", self.edit_slope_schedule),
            ("CRR steps", "crr_step_times", "crr_step_values", "Edit", self.edit_crr_schedule),
        ]
        for row, (title, time_key, value_key, button_text, command) in enumerate(schedule_specs):
            # ttk.Label(box, text=title).grid(row=row, column=0, sticky="w", padx=4, pady=6)
            schedule_help_key = {
                "Speed steps": "speed_schedule",
                "Mass steps": "mass_schedule",
                "Slope steps": "slope_schedule",
                "CRR steps": "crr_schedule",
            }[title]

            self.add_label_with_info(box, row, 0, title, schedule_help_key, padx=4, pady=6)
            lbl = ttk.Label(box, text="")
            lbl.grid(row=row, column=1, sticky="w", padx=4, pady=6)
            self.schedule_labels[time_key] = lbl
            ttk.Button(box, text=button_text, command=command).grid(row=row, column=2, sticky="e", padx=4, pady=6)

    def build_action_panel(self, parent: ttk.Frame) -> None:
        '''
        Purpose:
            Creates the right-side action area with reset, validate,
            simulation, summary, and plot buttons.

        Inputs:
            parent: frame that will contain the action widgets

        Output:
            No return value. Adds widgets to the GUI.
        '''
        summary_box = ttk.LabelFrame(parent, text="Simulation", padding=12)
        summary_box.grid(row=0, column=0, sticky="nsew")
        summary_box.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        top_buttons = ttk.Frame(summary_box)
        top_buttons.grid(row=0, column=0, sticky="ew")
        top_buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(top_buttons, text="Reset to defaults", command=self.reset_to_defaults).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(top_buttons, text="Validate", command=self.validate_form).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        # ttk.Button(top_buttons, text="Start simulation", command=self.start_simulation).grid(row=0, column=2, sticky="ew", padx=4, pady=4)
        self.start_button = ttk.Button(top_buttons, text="Start simulation", command=self.start_simulation)
        self.start_button.grid(row=0, column=2, sticky="ew", padx=4, pady=4)

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(
            summary_box,
            textvariable=self.status_var,
            foreground="#004d40",
            wraplength=420,
            justify="left"
        )
        self.status_label.grid(row=1, column=0, sticky="w", padx=4, pady=(8, 4))

        self.output_text = tk.Text(summary_box, height=16, wrap="word")
        self.output_text.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        summary_box.rowconfigure(2, weight=1)

        plot_box = ttk.LabelFrame(summary_box, text="Open plots", padding=12)
        plot_box.grid(row=3, column=0, sticky="ew", padx=4, pady=(10, 0))
        # Button labels are plain text (Tkinter buttons cannot render mathtext/
        # LaTeX), but are kept as the descriptive, sentence-case counterpart of
        # each plot's LaTeX-rendered title in draw_matplotlib_plot() below --
        # see PLOT_WINDOW_TITLES, the single source of truth both this button
        # list and each plot popup's window title are built from.
        plot_defs = [
            ("speed_time", "Motor speed tracking: actual vs. reference"),
            ("torque_time", "Torque tracking: actual vs. commanded"),
            ("phase_currents", "Three-phase stator winding currents"),
            ("torque_speed", "Torque-speed operating trajectory"),

            ("id_tracking", "d-axis current tracking: actual vs. reference"),
            ("iq_tracking", "q-axis current tracking: actual vs. reference"),
            ("current_pi", "Current-loop dq voltage: PI output vs. applied"),
            ("pi_gains", "Current-loop PI gains vs. time"),
            ("speed_pi_gains", "Speed-loop PI gains vs. time"),
            ("speed_sat", "Speed-loop current reference saturation"),
            ("voltage_sat", "Voltage-vector magnitude saturation"),

            ("pwm_compare", "SVPWM carrier and modulation signals"),
            ("switch_states", "Upper-switch gating states with PWM timing"),

            ("thermal", "Stator and magnet temperature vs. time"),
            ("losses", "Copper and iron (core) losses vs. time"),
            ("vehicle", "Vehicle and load-condition schedules"),

            ("mtpa_q_compare", "q-axis current reference path"),
            ("mtpa_d_compare", "d-axis current reference path"),
            ("mtpa_is_compare", "Stator current magnitude reference path"),
            # ("mtpa_torque_cmd", "Torque command from speed PI"),

            ("back_emf", "dq speed-voltage and PM back-EMF components"),
            ("inductance_model", "Current-dependent d/q inductance curves"),
            ("position_time", "Rotor mechanical position vs. time"),
        ]
        self.PLOT_WINDOW_TITLES = dict(plot_defs)
        for i, (key, text) in enumerate(plot_defs):
            btn = ttk.Button(plot_box, text=text, state="disabled", command=lambda k=key: self.show_plot(k))
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=4)
            plot_box.columnconfigure(i % 2, weight=1)
            self.plot_buttons[key] = btn

    def add_field(self, box: ttk.Frame, spec: FieldSpec, row: int) -> int:
        '''
        Purpose:
            Creates one GUI input field based on a FieldSpec definition.

        Inputs:
            box: parent frame
            spec: field specification object
            row: starting row index in the grid

        Output:
            Returns the next available row index after placing the field
            and its associated error label.
        '''
        # ttk.Label(box, text=spec.label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        self.add_label_with_info(box, row, 0, spec.label, spec.key)
        if spec.kind == "bool":
            var: tk.Variable = tk.BooleanVar(value=spec.default)
            widget = ttk.Checkbutton(box, variable=var)
            widget.grid(row=row, column=1, sticky="w", padx=4, pady=4)
        elif spec.kind == "enum" and spec.options and len(spec.options) == 2:
            var = tk.StringVar(value=str(spec.default))
            # holder = ttk.Frame(box)
            # holder.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            # for i, option in enumerate(spec.options):
            #     ttk.Radiobutton(holder, text=option, value=option, variable=var).grid(row=0, column=i, padx=(0, 10), sticky="w")
            widget = ttk.Frame(box)
            widget.grid(row=row, column=1, sticky="w", padx=4, pady=4)

            for i, option in enumerate(spec.options):
                ttk.Radiobutton(
                    widget,
                    text=option,
                    value=option,
                    variable=var
                ).grid(row=0, column=i, padx=(0, 10), sticky="w")

        elif spec.kind == "enum":
            var = tk.StringVar(value=str(spec.default))
            widget = ttk.Combobox(box, textvariable=var, values=spec.options or [], state="readonly")
            widget.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        else:
            var = tk.StringVar(value=str(spec.default))
            widget = ttk.Entry(box, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", padx=4, pady=4)

        self.variables[spec.key] = var
        self.input_widgets[spec.key] = widget

        err = ttk.Label(box, text="", foreground="#b00020")
        err.grid(row=row + 1, column=1, sticky="w", padx=4)
        self.error_labels[spec.key] = err
        return row + 2

    def add_label_with_info(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        text: str,
        info_key: str,
        padx: int | tuple[int, int] = 4,
        pady: int | tuple[int, int] = 4,
    ) -> ttk.Frame:
        """
        Creates a label plus small circled info icon.
        The icon shows PARAMETER_HELP[info_key] on hover/click.
        """
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=column, sticky="w", padx=padx, pady=pady)

        ttk.Label(holder, text=text).grid(row=0, column=0, sticky="w")

        help_text = PARAMETER_HELP.get(
            info_key,
            f"Code variable:\n{info_key}\n\nNo detailed help text has been defined yet."
        )

        icon = ttk.Label(
            holder,
            text=" ⓘ",
            cursor="hand2",
            foreground="#0066aa",
            font=("Segoe UI", 10, "bold"),
        )
        icon.grid(row=0, column=1, sticky="w", padx=(4, 0))

        InfoTooltip(icon, help_text, wraplength=650)

        return holder

    # -----------------------------------------------------------------
    # Reset and schedule summaries
    # -----------------------------------------------------------------
    def reset_to_defaults(self) -> None:
        '''
        Purpose:
            Restores all input fields, schedules, slider values, and GUI state
            to the initial default configuration.

        Output:
            No return value. Resets internal state and updates GUI text/buttons.
        '''
        for spec in MAIN_FIELDS + ADV_FIELDS:
            var = self.variables.get(spec.key)
            if var is None:
                continue
            if spec.kind == "bool":
                var.set(bool(spec.default))
            else:
                var.set(str(spec.default))
            self.error_labels[spec.key].config(text="")

        self.schedules = {
            "t_steps": list(DEFAULTS["t_steps"]),
            "omega_steps": list(DEFAULTS["omega_steps"]),
            "mass_step_times": list(DEFAULTS["mass_step_times"]),
            "mass_step_values": list(DEFAULTS["mass_step_values"]),
            "slope_step_times": list(DEFAULTS["slope_step_times"]),
            "slope_step_values_deg": list(DEFAULTS["slope_step_values_deg"]),
            "crr_step_times": list(DEFAULTS["crr_step_times"]),
            "crr_step_values": list(DEFAULTS["crr_step_values"]),
        }
        self.refresh_schedule_labels()
        self.update_pwm_slider_range()
        self.last_result = None
        self.last_params = None
        self.update_plot_button_states()
        self.output_text.delete("1.0", tk.END)
        self.status_var.set("Defaults restored.")

        self.update_compatibility_state()
        self.update_inductance_helper()

    def refresh_schedule_labels(self) -> None:
        '''
        Purpose:
            Updates the small schedule summary labels shown in the GUI.

        Output:
            No return value. Writes human-readable schedule summaries to labels.
        '''
        speed_rpm = [rad_s_to_rpm(v) for v in self.schedules["omega_steps"]]
        label_map = {
            "t_steps": f"{len(self.schedules['t_steps'])} points: {self.schedules['t_steps']} / {[round(v, 1) for v in speed_rpm]} RPM",
        #     "t_steps": f"{len(self.schedules['t_steps'])} points: {self.schedules['t_steps']} / {self.schedules['omega_steps']}",
            "mass_step_times": f"{len(self.schedules['mass_step_times'])} points: {self.schedules['mass_step_times']} / {self.schedules['mass_step_values']}",
            "slope_step_times": f"{len(self.schedules['slope_step_times'])} points: {self.schedules['slope_step_times']} / {self.schedules['slope_step_values_deg']}",
            "crr_step_times": f"{len(self.schedules['crr_step_times'])} points: {self.schedules['crr_step_times']} / {self.schedules['crr_step_values']}",
        }
        for key, text in label_map.items():
            self.schedule_labels[key].config(text=text)

    # -----------------------------------------------------------------
    # Schedule editors
    # -----------------------------------------------------------------
    def edit_speed_schedule(self) -> None:
        '''
        Purpose:
            Opens the speed-step editor window and updates the speed schedule
            if the user saves changes.

        Output:
            No return value. May modify self.schedules["t_steps"] and
            self.schedules["omega_steps"].
        '''
        editor = ScheduleEditor(
            self,
            title="Speed-step editor",
            time_key="t_steps",
            value_key="omega_steps",
            time_label="Step time [s]",
            value_label="Mechanical speed [RPM]",
            time_values=self.schedules["t_steps"],
            # step_values=self.schedules["omega_steps"],
            step_values=[rad_s_to_rpm(v) for v in self.schedules["omega_steps"]],
            value_validator=lambda v: None,
            help_kind="omega",
        )
        self.wait_window(editor)
        if editor.result:
            # self.schedules["t_steps"], self.schedules["omega_steps"] = editor.result
            # self.refresh_schedule_labels()
            times, rpm_values = editor.result
            self.schedules["t_steps"] = times
            self.schedules["omega_steps"] = [rpm_to_rad_s(v) for v in rpm_values]
            self.refresh_schedule_labels()

    def edit_mass_schedule(self) -> None:
        '''
        Purpose:
            Opens the mass-step editor window and updates the mass schedule
            if the user saves changes.

        Output:
            No return value. May modify mass step schedule arrays.
        '''
        editor = ScheduleEditor(
            self,
            title="Mass-step editor",
            time_key="mass_step_times",
            value_key="mass_step_values",
            time_label="Step time [s]",
            value_label="Vehicle mass [kg]",
            time_values=self.schedules["mass_step_times"],
            step_values=self.schedules["mass_step_values"],
            value_validator=lambda v: "mass must be >= 0." if v < 0 else None,
        )
        self.wait_window(editor)
        if editor.result:
            self.schedules["mass_step_times"], self.schedules["mass_step_values"] = editor.result
            self.refresh_schedule_labels()

    def edit_slope_schedule(self) -> None:
        '''
        Purpose:
            Opens the slope-step editor window and updates the slope schedule
            if the user saves changes.

        Output:
            No return value. May modify slope step schedule arrays.
        '''
        editor = ScheduleEditor(
            self,
            title="Slope-step editor",
            time_key="slope_step_times",
            value_key="slope_step_values_deg",
            time_label="Step time [s]",
            value_label="Road slope [deg]",
            time_values=self.schedules["slope_step_times"],
            step_values=self.schedules["slope_step_values_deg"],
            value_validator=lambda v: None,
        )
        self.wait_window(editor)
        if editor.result:
            self.schedules["slope_step_times"], self.schedules["slope_step_values_deg"] = editor.result
            self.refresh_schedule_labels()

    def edit_crr_schedule(self) -> None:
        '''
        Purpose:
            Opens the rolling-resistance editor and updates the Crr schedule
            if the user saves changes.

        Output:
            No return value. May modify Crr step schedule arrays.
        '''
        editor = ScheduleEditor(
            self,
            title="Rolling-resistance-step editor",
            time_key="crr_step_times",
            value_key="crr_step_values",
            time_label="Step time [s]",
            value_label="Crr [-]",
            time_values=self.schedules["crr_step_times"],
            step_values=self.schedules["crr_step_values"],
            value_validator=lambda v: "Crr must be >= 0." if v < 0 else None,
            help_kind="crr",
        )
        self.wait_window(editor)
        if editor.result:
            self.schedules["crr_step_times"], self.schedules["crr_step_values"] = editor.result
            self.refresh_schedule_labels()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def get_pwm_zoom_span(self, params: Dict[str, Any]) -> float:
        '''
        Purpose:
            Computes the time width of the PWM zoom window.

        Inputs:
            params: dictionary containing plot_pwm_num_periods and f_pwm

        Output:
            Returns a float representing zoom span in seconds.
        '''
        return float(params["plot_pwm_num_periods"]) / max(float(params["f_pwm"]), 1e-12)

    def update_pwm_slider_range(self) -> None:
        '''
        Purpose:
            Recomputes the allowed slider range for PWM zoom start so that
            the zoom window remains inside the simulation duration.

        Output:
            No return value. Updates the slider maximum and label text.
        '''
        params = self.read_raw_values(fast=True)
        tfinal = params.get("tfinal", DEFAULTS["tfinal"])
        f_pwm = params.get("f_pwm", DEFAULTS["f_pwm"])
        n_periods = params.get("plot_pwm_num_periods", DEFAULTS["plot_pwm_num_periods"])
        span = float(n_periods) / max(float(f_pwm), 1e-12)
        max_start = max(0.0, float(tfinal) - span)
        self.pwm_zoom_scale.configure(to=max_start)
        cur = self.pwm_zoom_var.get()
        if cur > max_start:
            self.pwm_zoom_var.set(max_start)
        self.update_pwm_zoom_label()

    def update_pwm_zoom_label(self) -> None:
        '''
        Purpose:
            Shows the current PWM zoom start value next to the slider.

        Output:
            No return value. Updates the displayed label text.
        '''
        self.pwm_zoom_value_label.config(text=f"{self.pwm_zoom_var.get():.6f}")

    def read_raw_values(self, fast: bool = False) -> Dict[str, Any]:
        '''
        Purpose:
            Reads all GUI values into a dictionary without doing full validation.

        Inputs:
            fast:
                - True: if parsing fails, uses defaults for numeric fields
                - False: leaves unparseable values as raw strings

        Output:
            Returns a dictionary containing all current field values,
            PWM zoom start, and schedules.
        '''
        params: Dict[str, Any] = {}
        for spec in MAIN_FIELDS + ADV_FIELDS:
            raw = self.variables[spec.key].get()
            if spec.kind == "bool":
                params[spec.key] = bool(raw)
            elif spec.kind == "int":
                try:
                    params[spec.key] = int(raw)
                except Exception:
                    params[spec.key] = spec.default if fast else raw
            elif spec.kind == "float":
                try:
                    params[spec.key] = float(raw)
                except Exception:
                    params[spec.key] = spec.default if fast else raw
            else:
                params[spec.key] = raw
        params["plot_pwm_zoom_start"] = float(self.pwm_zoom_var.get())
        params.update(self.schedules)
        return params

    def validate_form(self) -> Optional[Dict[str, Any]]:
        '''
        Purpose:
            Validates all user inputs and schedules before simulation.

        Validation includes:
            - numeric parsing
            - min/max constraints
            - enum membership
            - cross-field consistency
            - schedule consistency
            - PWM zoom window bounds

        Output:
            Returns:
                - a validated parameter dictionary if all checks pass
                - None if validation fails
        '''
        self.update_pwm_slider_range()
        parsed: Dict[str, Any] = {}
        errors_found = False

        for spec in MAIN_FIELDS + ADV_FIELDS:
            self.error_labels[spec.key].config(text="")
            raw = self.variables[spec.key].get()
            value: Any = raw
            if spec.kind == "bool":
                value = bool(self.variables[spec.key].get())
            elif spec.kind == "enum":
                if spec.options and raw not in spec.options:
                    self.error_labels[spec.key].config(text=f"Choose one of: {', '.join(spec.options)}")
                    errors_found = True
                    continue
                value = raw
            elif spec.kind == "int":
                try:
                    value = int(raw)
                except ValueError:
                    self.error_labels[spec.key].config(text="Enter a valid integer.")
                    errors_found = True
                    continue
            elif spec.kind == "float":
                try:
                    value = float(raw)
                except ValueError:
                    self.error_labels[spec.key].config(text="Enter a valid number.")
                    errors_found = True
                    continue

            if spec.kind in {"int", "float"}:
                if not spec.allow_negative and value < 0:
                    self.error_labels[spec.key].config(text="Negative values are not allowed here.")
                    errors_found = True
                    continue
                if spec.minimum is not None and value < spec.minimum:
                    self.error_labels[spec.key].config(text=f"Value must be ≥ {spec.minimum}.")
                    errors_found = True
                    continue
                if spec.maximum is not None and value > spec.maximum:
                    self.error_labels[spec.key].config(text=f"Value must be ≤ {spec.maximum}.")
                    errors_found = True
                    continue
            parsed[spec.key] = value

        parsed["plot_pwm_zoom_start"] = float(self.pwm_zoom_var.get())
        self.error_labels["plot_pwm_zoom_start"].config(text="")

        schedule_errors = self.validate_schedules(parsed)
        if schedule_errors:
            errors_found = True
            self.status_var.set("Validation failed. Fix the marked fields or schedules.")
            messagebox.showerror("Validation errors", "\n".join(schedule_errors), parent=self)
            return None

        # derived consistency
        #
        # NOTE: the closed_loop_foc/frame_mode, SPMSM/MTPA, SPMSM/field-weakening,
        # dead-time/SVPWM, and open_loop_abc/MTPA+FW checks that used to be
        # duplicated here were removed. They're already enforced (both forced to a
        # safe value AND grayed out) by update_compatibility_state(), which runs on
        # every relevant GUI change, so they can never actually be violated by the
        # time validate_form() reads self.variables. Keeping both was two places
        # validating the same thing; update_compatibility_state() is the one that
        # actually prevents the bad state, so it's the one that stays.
        if parsed.get("poles", 0) < 2 or parsed.get("poles", 0) % 2 != 0:
            self.error_labels["poles"].config(text="Poles must be a positive even integer.")
            errors_found = True

        if parsed.get("T_fault_stator", 0) <= parsed.get("T_warn_stator", 0):
            self.error_labels["T_fault_stator"].config(text="Fault threshold must be greater than warning threshold.")
            errors_found = True

        if parsed.get("T_fault_magnet", 0) <= parsed.get("T_warn_magnet", 0):
            self.error_labels["T_fault_magnet"].config(text="Fault threshold must be greater than warning threshold.")
            errors_found = True

        if parsed.get("phi_f_min_factor", 0) > 1:
            self.error_labels["phi_f_min_factor"].config(text="Use a value between 0 and 1.")
            errors_found = True

        span = self.get_pwm_zoom_span(parsed)
        if parsed["plot_pwm_zoom_start"] + span > parsed["tfinal"] + 1e-12:
            self.error_labels["plot_pwm_zoom_start"].config(text="Zoom window must end before simulation end time.")
            errors_found = True

        if errors_found:
            self.status_var.set("Validation failed. Fix the highlighted items.")
            return None

        parsed.update(self.schedules)

        consistency_errors, consistency_warnings, consistency_infos = self.build_consistency_messages(parsed)

        if consistency_errors:
            errors_found = True
            messagebox.showerror(
                "Validation errors",
                "\n\n".join(consistency_errors),
                parent=self
            )
            self.status_var.set("Validation failed. Fix the highlighted items.")
            return None

        # Keep warnings/infos for display even when validation passes
        parsed["_validation_warnings"] = consistency_warnings
        parsed["_validation_infos"] = consistency_infos

        messages = []
        if parsed["_validation_warnings"]:
            messages.append("Warnings:\n" + "\n\n".join(parsed["_validation_warnings"]))
        if parsed["_validation_infos"]:
            messages.append("Info:\n" + "\n\n".join(parsed["_validation_infos"]))

        if messages:
            messagebox.showwarning(
                "Validation summary",
                "\n\n".join(messages),
                parent=self
            )
            self.status_var.set("Validation passed with warnings/info messages.")
        else:
            self.status_var.set("Validation successful.")

        return parsed
    
    # def update_plot_button_states(self) -> None:
    #     if self.last_result is None:
    #         for btn in self.plot_buttons.values():
    #             btn.config(state="disabled")
    #         return

    #     for btn in self.plot_buttons.values():
    #         btn.config(state="normal")

    #     machine_type = str(self.variables["machine_type"].get())
    #     drive_mode = str(self.variables["drive_mode"].get())
    #     use_mtpa = bool(self.variables["use_mtpa"].get())
    #     # use_fw = bool(self.variables["use_field_weakening"].get())
    #     use_svpwm = bool(self.variables["use_svpwm"].get())

    #     mtpa_related = {
    #         "mtpa_q_compare",
    #         "mtpa_d_compare",
    #         "mtpa_is_compare",
    #         "mtpa_torque_cmd",
    #         # "inductance_model",
    #     }

    #     svpwm_related = {
    #         "pwm_compare",
    #         "switch_states",
    #     }

    #     # fw_related = {
    #     #     "voltage_sat",
    #     # }

    #     foc_related = {
    #         "id_tracking",
    #         "iq_tracking",
    #         "current_pi",
    #         "dq_voltage",
    #         "speed_sat",
    #         "back_emf",
    #     }

    #     # MTPA plots only for IPMSM + MTPA enabled
    #     if machine_type == "spmsm" or not use_mtpa:
    #         for key in mtpa_related:
    #             self.plot_buttons[key].config(state="disabled")

    #     # PWM plots only when SVPWM is used
    #     if not use_svpwm:
    #         for key in svpwm_related:
    #             self.plot_buttons[key].config(state="disabled")

    #     # Field-weakening-specific voltage saturation plot only if FW is enabled
    #     # You may keep this enabled if you still want to debug normal voltage saturation.
    #     # if not use_fw:
    #     #     for key in fw_related:
    #     #         self.plot_buttons[key].config(state="disabled")

    #     # FOC plots only make sense for closed-loop FOC
    #     if drive_mode != "closed_loop_foc":
    #         for key in foc_related:
    #             self.plot_buttons[key].config(state="disabled")

    def update_plot_button_states(self) -> None:
        """
        Enables plot buttons only after a simulation has finished.

        Important:
        Plot availability is based on self.last_params, not the current GUI fields.
        This prevents old simulation results from being treated as if they were
        generated with newly changed options.
        """
        if self.last_result is None or self.last_params is None:
            for btn in self.plot_buttons.values():
                btn.config(state="disabled")
            return

        params = self.last_params

        machine_type = str(params.get("machine_type", ""))
        drive_mode = str(params.get("drive_mode", ""))
        use_mtpa = bool(params.get("use_mtpa", False))
        use_fw = bool(params.get("use_field_weakening", False))
        use_svpwm = bool(params.get("use_svpwm", False))

        for btn in self.plot_buttons.values():
            btn.config(state="normal")

        current_reference_plots = {
            "mtpa_q_compare",
            "mtpa_d_compare",
            "mtpa_is_compare",
        }

        mtpa_only_plots = {
            "mtpa_torque_cmd",
        }

        svpwm_only_plots = {
            "pwm_compare",
            "switch_states",
        }

        foc_only_plots = {
            "id_tracking",
            "iq_tracking",
            "current_pi",
            "speed_sat",
            "back_emf",
            "mtpa_q_compare",
            "mtpa_d_compare",
            "mtpa_is_compare",
            "mtpa_torque_cmd",
        }

        # Current-reference comparison plots are useful if MTPA OR field weakening was active.
        # If both are off, disable them.
        if machine_type == "spmsm" or (not use_mtpa and not use_fw):
            for key in current_reference_plots:
                if key in self.plot_buttons:
                    self.plot_buttons[key].config(state="disabled")

        # Torque command plot is mainly useful for MTPA/FW debugging.
        if machine_type == "spmsm" or (not use_mtpa and not use_fw):
            for key in mtpa_only_plots:
                if key in self.plot_buttons:
                    self.plot_buttons[key].config(state="disabled")

        if not use_svpwm:
            for key in svpwm_only_plots:
                if key in self.plot_buttons:
                    self.plot_buttons[key].config(state="disabled")

        if drive_mode != "closed_loop_foc":
            for key in foc_only_plots:
                if key in self.plot_buttons:
                    self.plot_buttons[key].config(state="disabled")

    def validate_schedules(self, parsed: Dict[str, Any]) -> List[str]:
        '''
        Purpose:
            Validates all schedule arrays for length, ordering, bounds,
            and physical plausibility.

        Inputs:
            parsed: dictionary of already parsed main form values,
                    especially tfinal

        Output:
            Returns a list of error-message strings.
            Empty list means validation passed.
        '''
        errors: List[str] = []
        pairs = [
            ("t_steps", "omega_steps", "speed"),
            ("mass_step_times", "mass_step_values", "mass"),
            ("slope_step_times", "slope_step_values_deg", "slope"),
            ("crr_step_times", "crr_step_values", "crr"),
        ]
        for t_key, v_key, label in pairs:
            times = self.schedules[t_key]
            vals = self.schedules[v_key]
            if len(times) != len(vals):
                errors.append(f"{label}: time array and value array must have the same length.")
                continue
            if len(times) == 0:
                errors.append(f"{label}: at least one step is required.")
                continue
            if any(t < 0 for t in times):
                errors.append(f"{label}: step times must be >= 0.")
            if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
                errors.append(f"{label}: step times must be sorted in nondecreasing order.")
            if t_key != "t_steps" and any(t > parsed["tfinal"] for t in times):
                errors.append(f"{label}: step times must not exceed tfinal.")
            if label == "mass" and any(v < 0 for v in vals):
                errors.append("mass: values must be >= 0.")
            if label == "crr" and any(v < 0 for v in vals):
                errors.append("crr: values must be >= 0.")
        return errors

    # -----------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------
    # def start_simulation(self) -> None:
    #     '''
    #     Purpose:
    #         Validates the form, runs the simulation, displays the summary,
    #         and enables plot buttons if successful.

    #     Output:
    #         No return value.
    #         Updates:
    #             - self.last_result
    #             - summary text box
    #             - status label
    #             - plot button enabled state
    #     '''
    #     params = self.validate_form()
    #     if params is None:
    #         return
        
    #     self.start_button.config(state="disabled")
    #     self.status_var.set("Simulation running...")
    #     self.update_idletasks()

    #     try:
    #         # self.last_result = run_user_simulation(params)
    #         self.last_result = run_pmsm_simulation(params)
    #     except Exception as exc:
    #         self.status_var.set("Simulation failed.")
    #         messagebox.showerror("Simulation error", str(exc), parent=self)
    #         self.start_button.config(state="normal")
    #         return

    #     # self.output_text.delete("1.0", tk.END)
    #     # self.output_text.insert(tk.END, self.last_result["summary"])
    #     # # for btn in self.plot_buttons.values():
    #     # #     btn.config(state="normal")
    #     # self.update_plot_button_states()
    #     # self.status_var.set("Simulation finished. Use the plot buttons to open individual graphs.")
    #     # self.start_button.config(state="normal")
    #     extra_text = []

    #     for w in params.get("_validation_warnings", []):
    #         extra_text.append("[WARNING]\n" + w)

    #     for info in params.get("_validation_infos", []):
    #         extra_text.append("[INFO]\n" + info)

    #     summary_text = ""
    #     if extra_text:
    #         summary_text += "\n\n".join(extra_text) + "\n\n" + ("-" * 80) + "\n\n"

    #     summary_text += self.last_result["summary"]

    #     self.output_text.delete("1.0", tk.END)
    #     self.output_text.insert(tk.END, summary_text)

    #     self.update_plot_button_states()
    #     self.status_var.set("Simulation finished. Use the plot buttons to open individual graphs.")
    #     self.start_button.config(state="normal")

    def start_simulation(self) -> None:
        """
        Validates the form, starts the simulation in a background thread,
        and updates the GUI when finished.

        Important:
        Tkinter must not be updated directly from the worker thread.
        All GUI updates are routed back through self.after(...).
        """
        params = self.validate_form()
        if params is None:
            return

        self.start_button.config(state="disabled")
        self.last_result = None
        self.last_params = None
        self.update_plot_button_states()

        self.output_text.delete("1.0", tk.END)
        self.status_var.set("Simulation running...")
        self.status_label.update_idletasks()

        # So the simulation runs in a background thread, 
        #  while the Tkinter mainloop() continues running 
        #  in the main thread. That is why you can still 
        #  edit fields during the simulation.
        #  >> self.after(...) safely updates GUI
        worker = threading.Thread(
            target=self._simulation_worker,
            args=(params,),
            daemon=True
        )
        worker.start()

    def _simulation_worker(self, params: Dict[str, Any]) -> None:
        """
        Runs the numerical simulation outside the Tkinter GUI thread.
        """
        try:
            result = run_pmsm_simulation(params)
        except Exception as exc:
            error_text = "".join(traceback.format_exception_only(type(exc), exc))
            self.after(0, lambda: self._simulation_failed(error_text))
            return

        self.after(0, lambda: self._simulation_finished(params, result))

    def _simulation_failed(self, error_text: str) -> None:
        """
        Handles simulation failure on the Tkinter main thread.
        """
        self.status_var.set("Simulation failed.")
        self.start_button.config(state="normal")
        messagebox.showerror("Simulation error", error_text, parent=self)

    def _simulation_finished(self, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Handles successful simulation completion on the Tkinter main thread.
        """
        self.last_result = result
        self.last_params = params

        extra_text = []

        for w in params.get("_validation_warnings", []):
            extra_text.append("[WARNING]\n" + w)

        for info in params.get("_validation_infos", []):
            extra_text.append("[INFO]\n" + info)

        summary_text = ""
        if extra_text:
            summary_text += "\n\n".join(extra_text)
            summary_text += "\n\n" + ("-" * 80) + "\n\n"

        summary_text += self.last_result["summary"]

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, summary_text)

        self.update_plot_button_states()
        self.status_var.set("Simulation finished. Use the plot buttons to open individual graphs.")
        self.start_button.config(state="normal")

    def show_plot(self, key: str) -> None:
        '''
        Purpose:
            Opens the selected plot in a new Tkinter popup window
            with Matplotlib toolbar support.

        Inputs:
            key: dictionary key of the plot to show

        Output:
            No return value. Opens a popup with zoom/pan/save tools.
        '''
        if not self.last_result:
            return
        
        run_params = self.last_params or {}
        run_machine_type = str(run_params.get("machine_type", ""))
        run_use_mtpa = bool(run_params.get("use_mtpa", False))
        run_use_fw = bool(run_params.get("use_field_weakening", False))
        run_use_svpwm = bool(run_params.get("use_svpwm", False))
        
        # if key in {"pwm_compare", "switch_states"} and not bool(self.variables["use_svpwm"].get()):
        if key in {"pwm_compare", "switch_states"} and not run_use_svpwm:
            messagebox.showinfo("Plot unavailable", "PWM plots require SVPWM to be enabled.", parent=self)
            return

        # if key.startswith("mtpa_") and not bool(self.variables["use_mtpa"].get()):
        #     messagebox.showinfo("Plot unavailable", "MTPA plots require MTPA to be enabled.", parent=self)
        #     return

        # if key.startswith("mtpa_") and str(self.variables["machine_type"].get()) == "spmsm":
        #     messagebox.showinfo("Plot unavailable", "MTPA plots are disabled for SPMSM in this model.", parent=self)
        #     return

        if key in {"mtpa_q_compare", "mtpa_d_compare", "mtpa_is_compare", "mtpa_torque_cmd"}:
            if run_machine_type == "spmsm" or (not run_use_mtpa and not run_use_fw):
                messagebox.showinfo(
                    "Plot unavailable",
                    "This comparison requires MTPA or field weakening in the completed simulation.",
                    parent=self,
                )
                return

        signals = self.last_result.get("signals", {})
        if not signals:
            messagebox.showerror("Plot error", "No signal data available.", parent=self)
            return

        popup = tk.Toplevel(self)
        popup.title(self.PLOT_WINDOW_TITLES.get(key, key))
        popup.geometry("1100x750")
        popup.minsize(800, 500)

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        try:
            self.draw_matplotlib_plot(ax, key, signals)
            twin_ax = getattr(ax, "_twin_ax", None)
            self.enable_legend_toggle(fig, ax, extra_axes=[twin_ax] if twin_ax is not None else None)
        except Exception as exc:
            popup.destroy()
            messagebox.showerror("Plot error", str(exc), parent=self)
            return

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, popup)
        toolbar.update()
        toolbar.pack(fill="x")

    def draw_matplotlib_plot(self, ax, key: str, s: Dict[str, Any]) -> None:
        '''
        Purpose:
            Draws one selected plot onto a Matplotlib Axes object
            using simulation signals returned by the backend.

        Inputs:
            ax: Matplotlib axes
            key: plot key
            s: signal dictionary from backend

        Output:
            No return value. Draws onto ax.
        '''
        t = s["t"]

        run_params = self.last_params or {}
        run_machine_type = str(run_params.get("machine_type", ""))
        run_use_mtpa = bool(run_params.get("use_mtpa", False))
        run_use_fw = bool(run_params.get("use_field_weakening", False))
        run_use_svpwm = bool(run_params.get("use_svpwm", False))

        # Light purple background wherever field weakening was actually
        # engaged (drawn first so it sits behind every line plotted below).
        # See shade_field_weakening_background()'s docstring for exactly
        # what "engaged" means and why SPMSM runs never get any shading.
        if key not in FW_SHADE_EXCLUDED_KEYS:
            shade_field_weakening_background(ax, t, s, run_use_fw, run_machine_type)

        if key == "speed_time":
            # ax.plot(t, s["speed_rpm"], label=mmix(_word("Actual motor speed "), "n"))
            # ax.plot(t, s["speed_ref_rpm"], linestyle="--", label=mmix(_word("Reference motor speed "), "n_{\\mathrm{ref}}"))
            # ax.set_title(mtext("Motor speed tracking: actual vs. reference"))
            # ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            # ax.set_ylabel(mmix(_word("Motor speed "), "n", _word(" [rpm]")))
            ax.plot(t, s["speed_rpm"], label=mmix(_word("Actual motor speed "), "\\omega_{\\mathrm{m}}"))
            ax.plot(t, s["speed_ref_rpm"], linestyle="--", label=mmix(_word("Reference motor speed "), "\\omega_{\\mathrm{m}}^{\\mathrm{ref}}"))
            ax.set_title(mtext("Motor speed tracking: actual vs. reference"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Motor speed "), "\\omega_{\\mathrm{m}}", _word(" [rpm]")))

        elif key == "torque_time":
            ax.plot(t, s["Torque"], label=mmix(_word("Actual torque "), "T"))
            if "torque_cmd_arr" in s:
                ax.plot(t, s["torque_cmd_arr"], linestyle="--", label=mmix(_word("Commanded torque "), "T_{\\mathrm{cmd}}"))

            if "T_load_arr" in s:
                ax.plot(t, s["T_load_arr"], label=mmix(_word("Load torque "), "T_{\\mathrm{load}}"))
            ax.set_title(mtext("Torque tracking: actual vs. commanded"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Torque "), "T", _word(" [N"), "{\\cdot}", _word("m]")))

        elif key == "phase_currents":
            ax.plot(t, s["i_phase_a"], label=mmix(_word("Phase A current "), "i_a"))
            ax.plot(t, s["i_phase_b"], label=mmix(_word("Phase B current "), "i_b"))
            ax.plot(t, s["i_phase_c"], label=mmix(_word("Phase C current "), "i_c"))
            ax.set_title(mtext("Three-phase stator winding currents"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current "), "i", _word(" [A]")))

        elif key == "torque_speed":
            ax.plot(s["speed_rpm"], s["Torque"], label=mtext("Torque-speed trajectory"))
            ax.set_title(mtext("Torque-speed operating trajectory"))
            # ax.set_xlabel(mmix(_word("Speed "), "n", _word(" [rpm]")))
            ax.set_xlabel(mmix(_word("Speed "), "\\omega_{\\mathrm{m}}", _word(" [rpm]")))
            ax.set_ylabel(mmix(_word("Torque "), "T", _word(" [N"), "{\\cdot}", _word("m]")))

        elif key == "id_tracking":
            ax.plot(t, s["i_d"], label=mmix(_word("Actual "), "i_d"))
            ax.plot(t, s["id_ref_arr"], linestyle="--", label=mmix(_word("Reference "), "i_d^{\\mathrm{ref}}"))
            ax.set_title(mtext("d-axis current tracking: actual vs. reference"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current "), "i_d", _word(" [A]")))

        elif key == "iq_tracking":
            ax.plot(t, s["i_q"], label=mmix(_word("Actual "), "i_q"))
            ax.plot(t, s["iq_ref_arr"], linestyle="--", label=mmix(_word("Reference "), "i_q^{\\mathrm{ref}}"))
            ax.set_title(mtext("q-axis current tracking: actual vs. reference"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current "), "i_q", _word(" [A]")))

        # elif key == "current_pi":
        #     ax.plot(t, s["u_d_uns_arr"], label="u_d")
        #     ax.plot(t, s["u_q_uns_arr"], label="u_q")
        #     ax.set_title("Current PI output")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Voltage (V)")
        # elif key == "current_pi":
        #     ax.plot(t, s["u_d_uns_arr"], label="d-axis PI output before decoupling")
        #     ax.plot(t, s["u_q_uns_arr"], label="q-axis PI output before decoupling")
        #     ax.plot(t, s["vd_cmd_arr"], label="applied d-axis voltage command after limiting")
        #     ax.plot(t, s["vq_cmd_arr"], label="applied q-axis voltage command after limiting")

        #     ax.set_title("Applied dq voltage compared with PI output voltage")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Voltage (V)")
        # elif key == "current_pi":
        #     ax.plot(t, s["vd_uns_arr"], linestyle="--", label="requested d-axis voltage before limiting")
        #     ax.plot(t, s["vq_uns_arr"], linestyle="--", label="requested q-axis voltage before limiting")
        #     ax.plot(t, s["vd_cmd_arr"], label="applied d-axis voltage after limiting")
        #     ax.plot(t, s["vq_cmd_arr"], label="applied q-axis voltage after limiting")

        #     ax.set_title("Requested compared with applied dq voltage")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Voltage (V)")
        elif key == "current_pi":
            ax.plot(t, s["vd_cmd_arr"], label=mmix(_word("Applied "), "v_d", _word(" (after limiting)")))
            ax.plot(t, s["vq_cmd_arr"], label=mmix(_word("Applied "), "v_q", _word(" (after limiting)")))
            ax.plot(t, s["vd_uns_arr"], linestyle="--", label=mmix(_word("Requested "), "v_d", _word(" (before limiting)")))
            ax.plot(t, s["vq_uns_arr"], linestyle="--", label=mmix(_word("Requested "), "v_q", _word(" (before limiting)")))

            if run_use_svpwm and "vd_applied_arr" in s:
                ax.plot(t, s["vd_applied_arr"], linestyle=":", label=mmix(_word("Applied "), "v_d", _word(" (after SVPWM and dead-time)")))
                ax.plot(t, s["vq_applied_arr"], linestyle=":", label=mmix(_word("Applied "), "v_q", _word(" (after SVPWM and dead-time)")))

            ax.set_title(mtext("Current-loop dq voltage: PI output vs. applied"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Voltage "), "v", _word(" [V]")))

        # elif key == "dq_voltage":
        #     ax.plot(t, s["vd_cmd_arr"], label="v_d")
        #     ax.plot(t, s["vq_cmd_arr"], label="v_q")
        #     ax.set_title("Commanded dq voltages")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Voltage (V)")

        elif key == "speed_sat":
            ax.plot(t, s["iq_ref_uns_arr"], linestyle="--", label=mmix(_word("Unsaturated "), "i_q^{\\mathrm{ref}}"))
            ax.plot(t, s["iq_ref_clamp_arr"], linestyle="--", label=mmix(_word("Clamped "), "i_q^{\\mathrm{ref}}"))
            ax.plot(t, s["i_q"], label=mmix(_word("Actual "), "i_q"))
            imax = self.read_raw_values(fast=True)["Imax"]
            ax.axhline(y=imax, linestyle="--", label=mmix(_word("Current limit "), "\\pm I_{\\mathrm{max}}"))
            ax.axhline(y=-imax, linestyle="--")
            ax.set_title(mtext("Speed-loop current reference saturation"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current "), "i_q", _word(" [A]")))

        elif key == "voltage_sat":
            ax.plot(t, s["v_mag_uns_arr"], linestyle="--", label=mmix(_word("Requested voltage magnitude "), "|v|", _word(" (before limiting)")))
            ax.plot(t, s["v_mag_cmd_arr"], label=mmix(_word("Applied voltage magnitude "), "|v|", _word(" (after limiting)")))
            if "Vmax" in s:
                ax.plot(t, np.full_like(t, s["Vmax"]), label=mmix(_word("DC-bus voltage limit "), "V_{\\mathrm{max}}"))
            ax.set_title(mtext("Voltage-vector magnitude saturation"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Voltage magnitude "), "|v|", _word(" [V]")))

        elif key == "pwm_compare":
            tz = s["t_zoom"]
            ax.plot(tz, s["carrier_zoom"], label=mtext("PWM triangular carrier"))
            ax.plot(tz, s["ma_zoom"], label=mmix(_word("Phase A modulation signal "), "m_a"))
            ax.plot(tz, s["mb_zoom"], label=mmix(_word("Phase B modulation signal "), "m_b"))
            ax.plot(tz, s["mc_zoom"], label=mmix(_word("Phase C modulation signal "), "m_c"))
            ax.set_title(mtext("SVPWM carrier and modulation signals"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mtext("Normalized amplitude"))

        # elif key == "switch_states":
        #     tz = s["t_zoom"]
        #     ax.step(tz, s["gA_up"], where="post", label="S_A_upper")
        #     ax.step(tz, s["gB_up"] + 1.2, where="post", label="S_B_upper")
        #     ax.step(tz, s["gC_up"] + 2.4, where="post", label="S_C_upper")
        #     ax.set_title("Upper-switch states")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Switch state")
        elif key == "switch_states":
            tz = s["t_zoom"]

            # switch traces
            ax.step(tz, s["gA_up"], where="post", label=mtext("Phase A upper switch"))
            ax.step(tz, s["gB_up"] + 1.2, where="post", label=mtext("Phase B upper switch"))
            ax.step(tz, s["gC_up"] + 2.4, where="post", label=mtext("Phase C upper switch"))

            # PWM timing
            f_pwm = self.read_raw_values(fast=True)["f_pwm"]
            T_pwm = 1.0 / max(f_pwm, 1e-12)

            t_start = float(tz[0])
            t_end = float(tz[-1])

            # first PWM boundary at or before the visible window
            first_boundary = np.floor(t_start / T_pwm) * T_pwm
            pwm_boundaries = np.arange(first_boundary, t_end + T_pwm, T_pwm)
            pwm_centers = pwm_boundaries + 0.5 * T_pwm

            # period boundaries: light dashed
            for tp in pwm_boundaries:
                if t_start <= tp <= t_end:
                    ax.axvline(tp, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

            # period centers: even lighter dotted
            for tp in pwm_centers:
                if t_start <= tp <= t_end:
                    ax.axvline(tp, color="gray", linestyle=":", linewidth=0.7, alpha=0.20)

            ax.set_title(mtext("Upper-switch gating states with PWM timing"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mtext("Gate state"))

            ax.set_yticks([0.5, 1.7, 2.9])
            ax.set_yticklabels([mtext("Phase A"), mtext("Phase B"), mtext("Phase C")])

        # elif key == "thermal":
            # ax.plot(t, s["T_s"], label="T_s")
            # ax.plot(t, s["T_m"], label="T_m")
            # ax.axhline(y=self.read_raw_values(fast=True)["T_warn_stator"], linestyle="--", label="T_warn_stator")
            # ax.axhline(y=self.read_raw_values(fast=True)["T_fault_stator"], linestyle="--", label="T_fault_stator")
            # ax.axhline(y=self.read_raw_values(fast=True)["T_warn_magnet"], linestyle="--", label="T_warn_magnet")
            # ax.axhline(y=self.read_raw_values(fast=True)["T_fault_magnet"], linestyle="--", label="T_fault_magnet")
            # ax.axhline(y=self.read_raw_values(fast=True)["T_demag"], linestyle="--", label="T_demag")
            # ax.set_title("Thermal states")
            # ax.set_xlabel("Time (s)")
            # ax.set_ylabel("Temperature (°C)")

            # temp ignored
        elif key == "thermal":
            params = self.read_raw_values(fast=True)

            ax.plot(t, s["T_s"], label=mmix(_word("Stator temperature "), "\\vartheta_s"))
            ax.plot(t, s["T_m"], label=mmix(_word("Magnet temperature "), "\\vartheta_m"))

            ax.axhline(
                y=params["T_warn_stator"],
                color="orange",
                linestyle="--",
                label=mtext("Stator warning limit"),
            )
            ax.axhline(
                y=params["T_fault_stator"],
                color="red",
                linestyle="--",
                label=mtext("Stator fault limit"),
            )
            ax.axhline(
                y=params["T_warn_magnet"],
                color="orange",
                linestyle="--",
                label=mtext("Magnet warning limit"),
            )
            ax.axhline(
                y=params["T_fault_magnet"],
                color="red",
                linestyle="--",
                label=mtext("Magnet fault limit"),
            )
            ax.axhline(
                y=params["T_demag"],
                color="darkred",
                linestyle="--",
                label=mtext("Demagnetization risk limit"),
            )

            ax.set_title(mtext("Stator and magnet temperature vs. time"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Temperature "), "\\vartheta", _word(" ["), "^{\\circ}", _word("C]")))

        elif key == "losses":
            run_include_iron_loss = bool(run_params.get("include_iron_loss", False))

            P_cu = s["P_cu_arr"]
            P_iron = s.get("P_iron_arr", np.zeros_like(t))
            P_total = P_cu + P_iron

            ax.plot(t, P_cu, label=mmix(_word("Copper loss "), "P_{\\mathrm{Cu}}"), linewidth=2.0)

            iron_label = mmix(_word("Iron (core) loss "), "P_{\\mathrm{Fe}}")
            if not run_include_iron_loss:
                iron_label = mmix(_word("Iron (core) loss "), "P_{\\mathrm{Fe}}", _word(" -- disabled (0 W)"))
            ax.plot(t, P_iron, label=iron_label, linewidth=2.0)

            ax.plot(
                t, P_total,
                label=mmix(_word("Total electromagnetic loss "), "P_{\\mathrm{Cu}}+P_{\\mathrm{Fe}}"),
                linestyle="--", color="black", linewidth=1.6,
            )

            ax.set_title(mtext("Copper and iron (core) losses vs. time"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Power "), "P", _word(" [W]")))

        elif key == "position_time":
            position_deg = np.mod(s["theta_m"] * 180.0 / np.pi, 360.0)

            ax.plot(t, position_deg, label=mmix(_word("Rotor position "), "\\theta_m"))

            ax.set_title(mtext("Rotor mechanical position vs. time"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Position "), "\\theta_m", _word(" [deg]")))
            ax.set_xlim(float(t[0]), float(t[-1]))
            ax.set_ylim(0.0, 400.0)

        elif key == "vehicle":
            ax.plot(t, s["v_vehicle"] * 3.6, label=mmix(_word("Vehicle speed "), "v", _word(" [km/h]")))
            ax.plot(t, s["mass_arr"], label=mmix(_word("Vehicle mass "), "m", _word(" [kg]")))
            ax.plot(t, s["slope_arr"], label=mmix(_word("Road slope "), "\\alpha", _word(" [deg]")))
            ax.plot(t, s["crr_arr"], label=mmix(_word("Rolling resistance coefficient "), "C_{\\mathrm{rr}}"))
            ax.set_title(mtext("Vehicle and load-condition schedules"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mtext("Mixed units"))

        # elif key == "mtpa_q_compare":
        #     ax.plot(t, s["iq_ref_pi_arr"], label="i_q from PI (before MTPA)")
        #     ax.plot(t, s["iq_mtpa_arr"], label="i_q after MTPA")
        #     ax.plot(t, s["iq_ref_arr"], label="i_q final (after MTPA + FW)")
        #     ax.set_title("q-axis current: PI vs MTPA vs final")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Current (A)")
        # elif key == "mtpa_q_compare":
        #     ax.plot(t, s["iq_ref_pi_arr"], label="q-axis current requested by speed controller")

        #     if run_use_mtpa and run_machine_type == "ipmsm":
        #         ax.plot(t, s["iq_mtpa_arr"], label="q-axis current after MTPA")

        #     ax.plot(t, s["iq_ref_arr"], label="final q-axis current reference after all limiting/FW")

        #     ax.set_title("q-axis current reference path")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Current (A)")
        elif key == "mtpa_q_compare":
            ax.plot(t, s["iq_ref_pi_arr"], label=mmix(_word("Speed-controller request "), "i_q^{\\mathrm{PI}}"))

            if run_use_mtpa and run_machine_type == "ipmsm":
                ax.plot(t, s["iq_mtpa_arr"], label=mmix(_word("After MTPA (before field weakening) "), "i_q^{\\mathrm{MTPA}}"))
            elif run_use_fw:
                ax.plot(t, s["iq_mtpa_arr"], label=mmix(_word("Before field weakening "), "i_q"))

            ax.plot(t, s["iq_ref_arr"], linestyle="--", label=mmix(_word("Final reference "), "i_q^{\\mathrm{ref}}"))

            if run_use_mtpa and run_use_fw:
                ax.set_title(mtext("q-axis current reference: MTPA + field weakening"))
            elif run_use_mtpa:
                ax.set_title(mtext("q-axis current reference: MTPA effect"))
            elif run_use_fw:
                ax.set_title(mtext("q-axis current reference: field weakening effect"))
            else:
                ax.set_title(mtext("q-axis current reference path"))

            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current "), "i_q", _word(" [A]")))

        # elif key == "mtpa_d_compare":
        #     ax.plot(t, np.zeros_like(t), label="i_d before MTPA (=0)")
        #     ax.plot(t, s["id_mtpa_arr"], label="i_d after MTPA")
        #     ax.plot(t, s["id_ref_arr"], label="i_d final (after FW)")
        #     ax.set_title("d-axis current introduced by MTPA / FW")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Current (A)")
        # elif key == "mtpa_d_compare":
        #     ax.plot(t, np.zeros_like(t), label="d-axis current before optimization/field weakening")

        #     if run_use_mtpa and run_machine_type == "ipmsm":
        #         ax.plot(t, s["id_mtpa_arr"], label="d-axis current introduced by MTPA")

        #     ax.plot(t, s["id_ref_arr"], label="final d-axis current reference after all limiting/FW")

        #     ax.set_title("d-axis current reference path")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Current (A)")
        elif key == "mtpa_d_compare":
            ax.plot(t, np.zeros_like(t), label=mmix(_word("Zero reference "), "i_d=0"))

            if run_use_mtpa and run_machine_type == "ipmsm":
                ax.plot(t, s["id_mtpa_arr"], label=mmix(_word("After MTPA (before field weakening) "), "i_d^{\\mathrm{MTPA}}"))
            elif run_use_fw:
                ax.plot(t, s["id_mtpa_arr"], label=mmix(_word("Before field weakening "), "i_d"))

            ax.plot(t, s["id_ref_arr"], linestyle="--", label=mmix(_word("Final reference "), "i_d^{\\mathrm{ref}}"))

            if run_use_mtpa and run_use_fw:
                ax.set_title(mtext("d-axis current reference: MTPA + field weakening"))
            elif run_use_mtpa:
                ax.set_title(mtext("d-axis current reference: MTPA effect"))
            elif run_use_fw:
                ax.set_title(mtext("d-axis current reference: field weakening effect"))
            else:
                ax.set_title(mtext("d-axis current reference path"))

            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current "), "i_d", _word(" [A]")))

        # elif key == "mtpa_is_compare":
        #     ax.plot(t, s["is_pi_arr"], label="|I_s| from PI-only")
        #     ax.plot(t, s["is_mtpa_arr"], label="|I_s| after MTPA")
        #     imax = self.read_raw_values(fast=True)["Imax"]
        #     ax.axhline(y=imax, linestyle="--", label="Imax")
        #     ax.set_title("Current magnitude: PI-only vs MTPA")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Current magnitude (A)")
        # elif key == "mtpa_is_compare":
        #     ax.plot(t, s["is_pi_arr"], label="current magnitude requested before MTPA")

        #     if run_use_mtpa and run_machine_type == "ipmsm":
        #         ax.plot(t, s["is_mtpa_arr"], label="current magnitude after MTPA")

        #     imax = self.last_params["Imax"] if self.last_params else self.read_raw_values(fast=True)["Imax"]
        #     ax.axhline(y=imax, linestyle="--", label="configured current limit Imax")

        #     ax.set_title("Stator current magnitude before and after MTPA")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Current magnitude (A)")
        elif key == "mtpa_is_compare":
            ax.plot(t, s["is_pi_arr"], label=mmix(_word("Speed-controller request "), "|i_s|^{\\mathrm{PI}}"))

            if run_use_mtpa and run_machine_type == "ipmsm":
                ax.plot(t, s["is_mtpa_arr"], label=mmix(_word("After MTPA (before field weakening) "), "|i_s|^{\\mathrm{MTPA}}"))
            elif run_use_fw:
                ax.plot(t, s["is_mtpa_arr"], label=mmix(_word("Before field weakening "), "|i_s|"))

            final_is_arr = np.hypot(s["id_ref_arr"], s["iq_ref_arr"])
            ax.plot(t, final_is_arr, linestyle="--", label=mmix(_word("Final reference "), "|i_s|^{\\mathrm{ref}}"))

            imax = self.last_params["Imax"] if self.last_params else self.read_raw_values(fast=True)["Imax"]
            ax.axhline(y=imax, linestyle="--", label=mmix(_word("Configured current limit "), "I_{\\mathrm{max}}"))

            if run_use_mtpa and run_use_fw:
                ax.set_title(mtext("Stator current magnitude: MTPA + field weakening"))
            elif run_use_mtpa:
                ax.set_title(mtext("Stator current magnitude: MTPA effect"))
            elif run_use_fw:
                ax.set_title(mtext("Stator current magnitude: field weakening effect"))
            else:
                ax.set_title(mtext("Stator current magnitude reference path"))

            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Current magnitude "), "|i_s|", _word(" [A]")))

        # elif key == "mtpa_torque_cmd":
        #     ax.plot(t, s["torque_cmd_arr"], linestyle="--", label="Torque command")
        #     ax.set_title("Torque command from speed PI")
        #     ax.set_xlabel("Time (s)")
        #     ax.set_ylabel("Torque (Nm)")

        elif key == "back_emf":
            ax.plot(t, s["ed_cross_arr"], label=mmix(_word("d-axis speed-coupling voltage "), "e_d=-\\omega_e L_q i_q"))
            ax.plot(t, s["eq_backemf_arr"], label=mmix(_word("q-axis speed/back-EMF voltage "), "e_q=\\omega_e(L_d i_d+\\varphi_f)"))
            ax.plot(t, s["epm_arr"], linestyle="--", label=mmix(_word("PM-only back-EMF component "), "e_{\\mathrm{pm}}=\\omega_e \\varphi_f"))
            ax.set_title(mtext("d/q speed-voltage and PM back-EMF components"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Voltage "), "e", _word(" [V]")))

        elif key == "inductance_model":
            ax.plot(s["id_grid_help"], s["Ld_curve"], label=mmix(_word("d-axis inductance "), "L_d(i_d)"))
            ax.plot(s["iq_grid_help"], s["Lq_curve"], label=mmix(_word("q-axis inductance "), "L_q(i_q)"))

            if "Ld0" in s:
                ax.axhline(s["Ld0"], linestyle="--", label=mmix(_word("d-axis inductance at low current "), "L_{d0}"))
            if "Lq0" in s:
                ax.axhline(s["Lq0"], linestyle="--", label=mmix(_word("q-axis inductance at low current "), "L_{q0}"))
            if "Ld_inf" in s:
                ax.axhline(s["Ld_inf"], linestyle=":", label=mmix(_word("d-axis saturated inductance limit "), "L_{d\\infty}"))
            if "Lq_inf" in s:
                ax.axhline(s["Lq_inf"], linestyle=":", label=mmix(_word("q-axis saturated inductance limit "), "L_{q\\infty}"))

            # ax.axvline(-s["I_Ld_sat"], linestyle="--", label="-I_Ld_sat")
            # ax.axvline(s["I_Lq_sat"], linestyle="--", label="I_Lq_sat")

            ax.set_title(mtext("Current-dependent d/q inductance curves"))
            ax.set_xlabel(mmix(_word("Current "), "i", _word(" [A]")))
            ax.set_ylabel(mmix(_word("Inductance "), "L", _word(" [H]")))

        elif key == "pi_gains":
            # All three current-loop PI gains together: K_pd and K_pq (proportional,
            # units V/A) are NOT constant -- they scale with the current-dependent
            # (saturating) dq inductances Ld_now/Lq_now, so they move whenever id/iq
            # move into saturation. K_i (integral, units V/(A*s)) is shared between
            # both axes and only depends on R_s (itself drifting with stator
            # temperature). K_i is numerically much larger than K_pd/K_pq (it carries
            # an extra factor of "per second"), so it is plotted on its own,
            # colour-matched secondary y-axis (ax.twinx()) rather than being crushed
            # flat on the same scale as the proportional gains -- that is the
            # "distinguish them properly" part. The two axes share one time axis and
            # one combined legend (see the twin-axis handling right after this
            # if/elif chain, and enable_legend_toggle()'s extra_axes support).
            ln_pd, = ax.plot(t, s["K_pd_arr"], color="tab:blue", label=mmix(_word("d-axis proportional gain "), "K_{p,d}"))
            ln_pq, = ax.plot(t, s["K_pq_arr"], color="tab:orange", linestyle="--", label=mmix(_word("q-axis proportional gain "), "K_{p,q}"))
            ax.set_title(mtext("Current-loop PI gains vs. time"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Proportional gain "), "K_p", _word(" [V/A]")), color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax.twinx()
            ax._twin_ax = ax2  # picked up by show_plot() for the combined legend/toggle
            ln_i, = ax2.plot(t, s["K_i_arr"], color="tab:green", linestyle=":", label=mmix(_word("Integral gain, shared d/q "), "K_i"))
            ax2.set_ylabel(mmix(_word("Integral gain "), "K_i", _word(" [V/(A"), "{\\cdot}", _word("s)]")), color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")
            ax2.grid(False)

        elif key == "speed_pi_gains":
            # Speed-loop (outer) PI gains, same "distinguish them properly"
            # twin-axis treatment as "pi_gains" above -- kept as a separate
            # plot rather than merged into that one because these carry yet
            # another pair of units (current per speed-error, not voltage per
            # current-error), so cramming all four gains onto one 2-axis plot
            # would stop being readable. K_ps (proportional, units A*s/rad,
            # i.e. A per rad/s of speed error) is NOT constant -- it scales
            # with the effective control inertia J_ctrl_now (grows with
            # vehicle mass when a load schedule is active) and inversely with
            # the torque constant K_T_now = 1.5*polepairs*phi_f (moves
            # whenever phi_f does -- flux weakening, temperature). K_is
            # (integral, units A/rad) is tied to K_ps by a fixed ratio
            # (omega_cs/5) and so moves in lockstep with it, but on a very
            # different numeric scale -- hence the secondary y-axis again.
            ln_ps, = ax.plot(t, s["K_ps_arr"], color="tab:blue", label=mmix(_word("Proportional gain "), "K_{p,s}"))
            ax.set_title(mtext("Speed-loop PI gains vs. time"))
            ax.set_xlabel(mmix(_word("Time "), "t", _word(" [s]")))
            ax.set_ylabel(mmix(_word("Proportional gain "), "K_{p,s}", _word(" [A"), "{\\cdot}", _word("s/rad]")), color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax.twinx()
            ax._twin_ax = ax2  # picked up by show_plot() for the combined legend/toggle
            ln_is, = ax2.plot(t, s["K_is_arr"], color="tab:green", linestyle=":", label=mmix(_word("Integral gain "), "K_{i,s}"))
            ax2.set_ylabel(mmix(_word("Integral gain "), "K_{i,s}", _word(" [A/rad]")), color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")
            ax2.grid(False)

        else:
            raise ValueError(f"Unknown plot key: {key}")

        time_based_keys = {
            "speed_time",
            "torque_time",
            "phase_currents",
            "id_tracking",
            "iq_tracking",
            "current_pi",
            "dq_voltage",
            "speed_sat",
            "voltage_sat",
            # temp ignored
            "thermal",
            "losses",
            "vehicle",
            "mtpa_q_compare",
            "mtpa_d_compare",
            "mtpa_is_compare",
            "mtpa_torque_cmd",
            "back_emf",
            "pi_gains",
            "speed_pi_gains",
        }

        if key in time_based_keys:
            ax.set_xlim(float(t[0]), float(t[-1]))
        elif key in {"pwm_compare", "switch_states"}:
            tz = s["t_zoom"]
            ax.set_xlim(float(tz[0]), float(tz[-1]))

        ax.grid(True)

        # Axis tick numbers rendered through mathtext (LatexScalarFormatter,
        # defined near the top of this file) so they match the LaTeX-style
        # (Computer-Modern) titles/labels/legends above instead of falling
        # back to the plain UI font. switch_states' y-axis is skipped -- it
        # already carries its own LaTeX-wrapped custom tick labels ("Phase A"
        # / "Phase B" / "Phase C" via mtext()) rather than plain numbers, and
        # overwriting its formatter here would replace those with numbers.
        ax.xaxis.set_major_formatter(LatexScalarFormatter())
        if key != "switch_states":
            ax.yaxis.set_major_formatter(LatexScalarFormatter())

        twin_ax = getattr(ax, "_twin_ax", None)
        if twin_ax is not None:
            twin_ax.yaxis.set_major_formatter(LatexScalarFormatter())
            # Combine both axes' handles into one legend on the primary axes --
            # a plain ax.legend() here would only see ax's own lines and silently
            # drop the twin axis's line (see "pi_gains" above).
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = twin_ax.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2)
        else:
            ax.legend()