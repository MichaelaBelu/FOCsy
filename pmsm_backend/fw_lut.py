from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from pmsm_backend.mtpa import MtpaLut


class FwLut:
    """
    Field-weakening "Regime B" lookup table for an IPMSM.

    ------------------------------------------------------------------
    WHERE THIS FITS IN THE THREE FIELD-WEAKENING REGIMES
    ------------------------------------------------------------------
    Above base speed there are really THREE distinct operating regimes,
    not one:

      Regime A (below base speed):
          The MTPA point (id_base, iq_base from MtpaLut) already fits
          inside the voltage-limit ellipse at this speed. No weakening
          needed. Handled directly in current_references_with_fw(),
          this LUT is not involved.

      Regime B (THIS FILE):
          The MTPA point no longer fits (voltage-limited), but the
          torque command can still be met EXACTLY using LESS than the
          full current limit Imax. Only the voltage constraint is
          active here -- the current-limit circle is not touched.

      Regime C (existing ipmsm_fw_currents_finite()):
          Even the cheapest (in current) way to hit torque_cmd under
          the voltage limit would need more than Imax. Both the
          voltage ellipse AND the current-limit circle are active
          simultaneously, and the machine can no longer deliver the
          full torque_cmd -- it delivers the best it can at |i_s|=Imax.
          ipmsm_fw_currents_finite() already computes this correctly;
          nothing about that formula needs to change.

    Before this file existed, current_references_with_fw() jumped
    straight from Regime A to Regime C the instant omega_r_e crossed
    omega_base -- i.e. it always assumed the current limit was active
    in field weakening, even at operating points that don't actually
    need the full current. This file adds the missing Regime B.

    ------------------------------------------------------------------
    THE PHYSICS: HOW REGIME B'S (i_d, i_q) IS DERIVED
    ------------------------------------------------------------------
    Two equations, two unknowns (i_d, i_q), for a given torque command
    T and electrical speed omega:

    (1) Torque equation (steady state, exact -- same law MtpaLut uses):

            T = 1.5 * p * (phi_f * i_q + (Ld(i_d) - Lq(i_q)) * i_d * i_q)

        Solved for i_q given a trial i_d (this is exactly what
        MtpaLut._solve_iq_for_torque() already does -- reused here
        rather than re-implemented):

            i_q = i_q(i_d)   such that T is met exactly

    (2) Voltage-limit equation (steady state, R_s dropped -- the same
        standard simplification used everywhere else in this file's
        sibling functions, spmsm_fw_currents_finite /
        ipmsm_fw_currents_finite):

            Vmax = omega * sqrt( (Ld(i_d)*i_d + phi_f)^2 + (Lq(i_q)*i_q)^2 )

        This is the voltage ELLIPSE boundary in the (i_d, i_q) plane;
        Regime B rides exactly ON this boundary (using the minimum
        extra current needed to get back within the voltage budget),
        not strictly inside it.

    Substituting i_q(i_d) from (1) into (2) leaves ONE equation in ONE
    unknown, i_d:

            g(i_d) = omega * sqrt( (Ld(i_d)*i_d + phi_f)^2
                                  + (Lq(i_q(i_d))*i_q(i_d))^2 )  -  Vmax  =  0

        Unlike the SPMSM case (where Ld=Lq makes i_q independent of
        i_d and g(i_d) collapses to a plain quadratic), for a salient
        IPMSM this is a nonlinear equation -- there is no clean closed
        form once i_q(i_d) is substituted in (it works out to a
        quartic in i_d in the unsaturated/constant-L case, and worse
        once Ld(i_d)/Lq(i_q) saturation is included). So it's solved
        numerically instead: NO Imax anywhere in this equation, which
        is exactly the point of Regime B.

    g(i_d) is bracketed and solved with brentq, exactly the same
    numerical tool MtpaLut already uses for its own inner torque solve
    -- nothing new is introduced here, this file leans on the same
    machinery. The bracket:

        - one end is i_d_mtpa (the Regime-A point): g(i_d_mtpa) > 0
          here is exactly the "voltage exceeded" condition that means
          we're above base speed in the first place.
        - the other end is wherever i_q(i_d) becomes infeasible under
          the Imax current-circle bound (MtpaLut._solve_iq_for_torque
          returns None past that point) -- i.e. the point where the
          torque-T hyperbola would leave the current-limit circle.

      ASSUMPTION (stated explicitly, not hidden): g(i_d) is assumed to
      decrease monotonically as i_d becomes more negative between
      those two ends -- physically, "more negative i_d weakens flux,
      which lowers required voltage." This was checked numerically for
      this machine's parameters (see the worked example below) and
      held throughout; it is the standard assumption for a
      non-deeply-saturated interior-PM machine, but would be worth
      re-checking if this LUT is ever reused for a machine with an
      unusual saturation curve.

      - If g(i_d) never reaches zero before that current-limit end
        (i.e. even at the most current Regime B is allowed to spend,
        the voltage is still too high) -- Regime B has NO feasible
        answer at this (T, omega). That cell is stored as NaN, which
        signals "fall through to Regime C" at lookup time.

    ------------------------------------------------------------------
    WHY A 2-D LOOKUP TABLE (NOT 1-D LIKE MtpaLut)
    ------------------------------------------------------------------
    MTPA only depends on torque -- the MTPA point for 5 Nm is the same
    point no matter the speed, so MtpaLut is a 1-D table indexed by
    torque alone. Regime B genuinely depends on BOTH torque and speed
    (the voltage-ellipse size depends on speed), so this table has to
    be indexed by the PAIR (torque, omega_r_e) -- a 2-D grid, with
    bilinear interpolation at lookup time instead of MtpaLut's simple
    1-D np.interp.

    ------------------------------------------------------------------
    STRUCTURE OF THE STORED TABLE
    ------------------------------------------------------------------
    After build(phi_f) runs, this object holds:

        self.torque_grid   : shape (Nt,)      -- 0 .. torque_max, Nm
        self.speed_grid     : shape (Nw,)      -- 0 .. omega_grid_max, rad/s (electrical)
        self.id_grid         : shape (Nt, Nw)   -- id_grid[i, j] = i_d at
                                                    (torque_grid[i], speed_grid[j])
        self.iq_grid         : shape (Nt, Nw)   -- matching i_q

    Reading id_grid/iq_grid column by column (fixed torque row i, all
    speed columns j) traces exactly the id(omega) curve you'd see on a
    classic "current vs. speed" field-weakening plot: flat at
    id_base/iq_base for speed <= omega_base(T_i) [Regime A], then i_d
    sliding smoothly negative once speed exceeds omega_base(T_i)
    [Regime B], eventually going to NaN once even Regime B can't keep
    up [-> caller should use Regime C, ipmsm_fw_currents_finite(),
    instead for those cells].

    ------------------------------------------------------------------
    WORKED EXAMPLE (real numbers, this machine's parameters)
    ------------------------------------------------------------------
    Using the IPMSM parameters from this project's own comparison
    script (compare_with_motulator.py's IPMSM_* constants):

        p = 2, R_s = 0.2 ohm, L_d = 0.0085 H, L_q = 0.0100 H,
        phi_f = 0.175 Wb, Imax = 20 A, Vdc = 400 V -> Vmax = 230.94 V

    At T = 3.0 Nm:  MTPA point is id_mtpa = -0.268 A, iq_mtpa = 5.701 A,
    which gives omega_base = 1269.7 rad/s (about 6062 rpm mechanical).

    Below 1269.7 rad/s: Regime A, use (-0.268, 5.701) unchanged.

    Above it, solving Regime B (verified numerically while writing this
    file):

        omega = 1600 rad/s : id* = -4.881 A, iq* = 5.485 A, |i_s| = 7.34 A
        omega = 1800 rad/s : id* = -6.894 A, iq* = 5.396 A, |i_s| = 8.75 A
        omega = 2000 rad/s : id* = -8.534 A, iq* = 5.325 A, |i_s| = 10.06 A

    Every one of those stays comfortably under Imax = 20 A -- meaning
    across this whole speed range, the OLD code (which jumped straight
    to the Imax-pinned Regime-C formula) was pinning i_s at 20 A when
    the machine only actually needed 7-10 A to hit the exact torque
    command. That gap is the ~40% id-magnitude discrepancy against
    motulator found earlier in this project's validation work -- this
    file is the fix for it.

    ------------------------------------------------------------------
    BUILT ONCE AT STARTUP, JUST LIKE MtpaLut
    ------------------------------------------------------------------
    build() is expensive relative to a single lookup() call (it runs a
    bracketed root-find for every one of Nt*Nw grid cells), so exactly
    like MtpaLut this is meant to be built ONCE when the simulator is
    constructed (see build_fw_object() / self.fw_lut in
    PMSMSimulation.__init__, mirroring self.mtpa_lut right next to it)
    and then only ever read from during the ODE solve via lookup().
    """

    def __init__(
        self,
        polepairs: int,
        i_max: float,
        v_max: float,
        mtpa_lut: MtpaLut,
        n_torque_points: int = 31,
        n_speed_points: int = 31,
        id_bracket_scan_points: int = 60,
        omega_grid_max: Optional[float] = None,
    ):
        """
        :param mtpa_lut: an ALREADY-BUILT MtpaLut for the same machine.
            Reused directly for i_d_mtpa/i_q_mtpa (the Regime-A/Regime-B
            boundary), Ld(i_d)/Lq(i_q) (same saturation law, no
            duplication), and the inner torque->i_q solve -- so this
            table always stays physically consistent with whatever
            MTPA curve the rest of the simulator is using.
        :param n_torque_points, n_speed_points: grid resolution. Kept
            modest by default (31x31 = 961 cells) because EACH cell
            costs one bracket scan (id_bracket_scan_points calls into
            MtpaLut's own brentq-based torque solve) plus one more
            brentq call to pin down i_d* -- i.e. building this table is
            roughly two orders of magnitude more expensive per point
            than MtpaLut's own build(), so raise these only if you've
            confirmed startup time is still acceptable.
        :param id_bracket_scan_points: how finely to scan from
            i_d_mtpa down toward -Imax when locating the bracket for
            g(i_d) = 0 (see class docstring). This does NOT need to be
            large -- it only has to be fine enough to not straddle the
            root and its neighboring current-limit cutoff in one step;
            the actual precision comes from the brentq call afterward.
        :param omega_grid_max: top of the speed axis (rad/s,
            electrical). If None, defaults to 1.5x the omega_base of
            the LOWEST nonzero torque grid point sampled to below 1 Nm
            or the torque grid's own minimum step, whichever is more
            conservative -- i.e. "a bit past the widest base-speed
            range this machine will realistically show." Pass this
            explicitly if you want the table to reach a specific known
            top speed instead.
        """
        self.p = int(polepairs)
        self.i_max = float(i_max)
        self.v_max = float(v_max)
        self.mtpa_lut = mtpa_lut

        self.n_torque_points = int(n_torque_points)
        self.n_speed_points = int(n_speed_points)
        self.id_bracket_scan_points = int(id_bracket_scan_points)
        self.omega_grid_max_setting = omega_grid_max

        self.torque_grid: Optional[np.ndarray] = None
        self.speed_grid: Optional[np.ndarray] = None
        self.id_grid: Optional[np.ndarray] = None
        self.iq_grid: Optional[np.ndarray] = None
        self.omega_grid_max: Optional[float] = None

    # -----------------------------------------------------------------
    # per-cell solve
    # -----------------------------------------------------------------
    def _voltage_needed(self, id_val: float, omega_r_e: float, torque_ref: float, phi_f: float
                         ) -> Optional[Tuple[float, float]]:
        """
        For a trial i_d, solve i_q from the torque equation (reusing
        MtpaLut's own brentq-based inner solve -- see the class
        docstring, equation (1)), then return (V_needed, i_q).
        Returns None if no feasible i_q exists for this i_d under the
        Imax current-circle bound -- i.e. this i_d has walked past
        where the torque-T hyperbola leaves the current-limit circle,
        which is exactly the natural stopping point for the bracket
        scan below.
        """
        iq_val = self.mtpa_lut._solve_iq_for_torque(torque_ref, id_val, phi_f)
        if iq_val is None:
            return None
        Ld = self.mtpa_lut.ld(id_val)
        Lq = self.mtpa_lut.lq(iq_val)
        v_needed = omega_r_e * np.hypot(Ld * id_val + phi_f, Lq * iq_val)
        return v_needed, iq_val

    def _solve_regime_b_point(
        self, torque_ref: float, omega_r_e: float, phi_f: float
    ) -> Tuple[Optional[float], Optional[float], float, float, float]:
        """
        Solve the Regime-B point for one (torque, speed) pair.

        :return: (i_d, i_q, id_mtpa, iq_mtpa, omega_base)
                 i_d/i_q are None if Regime B has no feasible answer
                 here (caller -- build(), or a live fallback lookup --
                 should use Regime C instead).
        """
        id_mtpa, iq_mtpa = self.mtpa_lut.solve_mtpa_point(torque_ref, phi_f)
        Ld_m = self.mtpa_lut.ld(id_mtpa)
        Lq_m = self.mtpa_lut.lq(iq_mtpa)
        omega_base = self.v_max / max(np.hypot(Ld_m * id_mtpa + phi_f, Lq_m * iq_mtpa), 1e-12)

        if omega_r_e <= omega_base:
            # Regime A: MTPA point already fits the voltage ellipse.
            return id_mtpa, iq_mtpa, id_mtpa, iq_mtpa, omega_base

        # ---- bracket scan: walk i_d from i_d_mtpa toward -Imax,
        #      tracking g(i_d) = V_needed(i_d) - Vmax, until either the
        #      sign flips (root bracketed -> Regime B feasible) or i_q
        #      becomes infeasible under Imax (Regime B not feasible).
        id_candidates = np.linspace(id_mtpa, -self.i_max, self.id_bracket_scan_points)

        prev_id = id_candidates[0]
        prev_res = self._voltage_needed(prev_id, omega_r_e, torque_ref, phi_f)
        if prev_res is None:
            # Shouldn't happen at i_d_mtpa itself, but guard anyway.
            return None, None, id_mtpa, iq_mtpa, omega_base
        prev_g = prev_res[0] - self.v_max

        for id_val in id_candidates[1:]:
            res = self._voltage_needed(id_val, omega_r_e, torque_ref, phi_f)
            if res is None:
                # Walked past where Imax can still deliver torque_ref
                # at all -> Regime B has no feasible point here.
                break
            g = res[0] - self.v_max
            if prev_g > 0.0 >= g:
                # Sign change bracketed between prev_id and id_val.
                def g_fn(x: float) -> float:
                    r = self._voltage_needed(x, omega_r_e, torque_ref, phi_f)
                    # r is guaranteed not None inside [prev_id, id_val]
                    # since both bracket ends were already evaluated.
                    return r[0] - self.v_max

                id_star = brentq(g_fn, prev_id, id_val, xtol=1e-6, rtol=1e-6, maxiter=100)
                _, iq_star = self._voltage_needed(id_star, omega_r_e, torque_ref, phi_f)
                return id_star, iq_star, id_mtpa, iq_mtpa, omega_base
            prev_id, prev_g = id_val, g

        # Never crossed zero before running out of current headroom:
        # even Imax isn't enough to bring the voltage down to Vmax
        # while still hitting torque_ref exactly -> Regime C territory.
        return None, None, id_mtpa, iq_mtpa, omega_base

    # -----------------------------------------------------------------
    # build the table (call once, at startup)
    # -----------------------------------------------------------------
    def build(self, phi_f: float) -> None:
        """
        Construct torque_grid / speed_grid / id_grid / iq_grid.
        Expensive (Nt*Nw bracketed root-finds) -- call once at startup,
        not inside the ODE loop.
        """
        torque_max = self.mtpa_lut.torque_max
        if torque_max is None:
            torque_max = self.mtpa_lut.estimate_torque_max(phi_f)

        self.torque_grid = np.linspace(0.0, torque_max, self.n_torque_points)

        # Auto top-of-speed-axis: 1.5x the omega_base at the SMALLEST
        # nonzero torque row (light load has the widest base-speed
        # range -- see the worked example in the class docstring, where
        # zero/light torque pushed omega_base well past the near-rated
        # value) plus a margin, so the grid comfortably covers Regime B
        # for every torque row without being wastefully large.
        if self.omega_grid_max_setting is not None:
            self.omega_grid_max = float(self.omega_grid_max_setting)
        else:
            t_probe = self.torque_grid[1] if len(self.torque_grid) > 1 else max(torque_max, 1e-6)
            _, _, _, _, wb_probe = self._solve_regime_b_point(t_probe, 0.0, phi_f)
            self.omega_grid_max = 1.5 * wb_probe

        self.speed_grid = np.linspace(0.0, self.omega_grid_max, self.n_speed_points)

        Nt, Nw = self.n_torque_points, self.n_speed_points
        self.id_grid = np.full((Nt, Nw), np.nan)
        self.iq_grid = np.full((Nt, Nw), np.nan)

        for i, T in enumerate(self.torque_grid):
            for j, w in enumerate(self.speed_grid):
                id_val, iq_val, _, _, _ = self._solve_regime_b_point(T, w, phi_f)
                if id_val is not None:
                    self.id_grid[i, j] = id_val
                    self.iq_grid[i, j] = iq_val
                # else: leave as NaN -> signals "use Regime C" at lookup time.

    # -----------------------------------------------------------------
    # lookup (call every ODE step -- must be cheap, no root-finding here)
    # -----------------------------------------------------------------
    def lookup(self, torque_abs: float, omega_r_e: float) -> Optional[Tuple[float, float]]:
        """
        Bilinear interpolation of (i_d, i_q) at (torque_abs, omega_r_e).
        torque_abs must already be abs(torque_cmd) -- this table only
        covers positive torque, matching MtpaLut's own convention
        (current_references_with_fw() applies the sign to i_q itself,
        same as it already does for the existing Regime-C formula).

        :return: (i_d, i_q), or None if this operating point falls in
            NaN territory (Regime B infeasible here -> caller should
            fall through to Regime C / ipmsm_fw_currents_finite()), or
            if this table hasn't been built() yet.
        """
        if self.torque_grid is None:
            return None

        T = float(np.clip(torque_abs, self.torque_grid[0], self.torque_grid[-1]))
        w = float(np.clip(omega_r_e, self.speed_grid[0], self.speed_grid[-1]))

        i1 = int(np.searchsorted(self.torque_grid, T, side="right"))
        i1 = min(max(i1, 1), len(self.torque_grid) - 1)
        i0 = i1 - 1
        j1 = int(np.searchsorted(self.speed_grid, w, side="right"))
        j1 = min(max(j1, 1), len(self.speed_grid) - 1)
        j0 = j1 - 1

        T0, T1 = self.torque_grid[i0], self.torque_grid[i1]
        w0, w1 = self.speed_grid[j0], self.speed_grid[j1]
        tt = 0.0 if T1 <= T0 else (T - T0) / (T1 - T0)
        ww = 0.0 if w1 <= w0 else (w - w0) / (w1 - w0)

        # bilinear weights for the 4 surrounding grid corners
        corners = [
            ((i0, j0), (1 - tt) * (1 - ww)),
            ((i1, j0), tt * (1 - ww)),
            ((i0, j1), (1 - tt) * ww),
            ((i1, j1), tt * ww),
        ]

        # NaN-aware interpolation: a corner can legitimately be NaN
        # (Regime B infeasible there). Rather than let one NaN corner
        # poison the whole interpolated result, drop NaN corners and
        # renormalize weights over whatever finite corners remain --
        # this keeps lookup() cheap (no re-solving at request time)
        # while staying correct right up to the Regime B/C boundary.
        # If ALL 4 corners are NaN, this point is squarely in Regime C
        # territory -> return None so the caller uses that formula.
        id_sum = 0.0
        iq_sum = 0.0
        weight_sum = 0.0
        for (ii, jj), wgt in corners:
            id_c = self.id_grid[ii, jj]
            iq_c = self.iq_grid[ii, jj]
            if np.isnan(id_c) or wgt <= 0.0:
                continue
            id_sum += wgt * id_c
            iq_sum += wgt * iq_c
            weight_sum += wgt

        if weight_sum <= 1e-12:
            return None

        return id_sum / weight_sum, iq_sum / weight_sum