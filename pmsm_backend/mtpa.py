from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import brentq


def proportional_grid_size(
    span: float,
    points_per_unit: float,
    min_points: int,
    max_points: int,
) -> int:
    """
    Convert a physical span into a reasonable number of grid points.

    Example:
        span = 100 Nm
        points_per_unit = 4 points/Nm
        -> about 401 points

    The result is clipped so the LUT does not become too coarse or too expensive.
    """
    span = abs(float(span))

    if span <= 0.0 or not np.isfinite(span):
        return int(min_points)

    n = int(np.ceil(span * points_per_unit)) + 1
    return int(np.clip(n, min_points, max_points))

class MtpaLut:
    """
    MTPA lookup table for an IPMSM.

    For each requested torque value, the LUT builder does this:
    
    1) Choose a candidate i_d from the interval [-Imax, 0]
       - In an IPMSM, MTPA usually uses negative d-axis current.
       - So we try many possible i_d values between full negative current and zero.
    
    2) Compute the allowable i_q range from the current circle
       - The stator current magnitude must satisfy:
             I_s^2 = i_d^2 + i_q^2 <= Imax^2
       - For a chosen i_d, the largest feasible i_q is:
             i_q_max = sqrt(Imax^2 - i_d^2)
       - So only i_q values between 0 and i_q_max are physically allowed.
    
    3) Solve for i_q so that the nonlinear torque equation matches the requested torque
       - For the chosen i_d, solve:
             T_e(i_d, i_q) = T_ref
       - In this model:
             T_e = 1.5 * p * (phi_f * i_q + (L_d(i_d) - L_q(i_q)) * i_d * i_q)
       - Because L_d and L_q change with current, this is a nonlinear equation.
       - So the code numerically finds the i_q that gives the desired torque.
    
       What goes into the Brentq solve:
       - Define a torque-error function:
             f(i_q) = T_e(i_d, i_q) - T_ref
       - The unknown is only i_q because i_d is fixed during this step.
       - Search interval:
             i_q in [0, i_q_max]
       - The solver checks:
             f(0) and f(i_q_max)
         If they have opposite signs, then a root lies between them.
       - That root is the i_q value for which:
             f(i_q) = 0
         which means:
             T_e(i_d, i_q) = T_ref
    
       How brentq works, in short:
       - brentq is a 1D root-finding method.
       - It does not minimize directly.
       - It finds where a scalar function crosses zero.
       - Here it finds the i_q for which the torque error becomes zero.
       - It is robust because it starts from an interval that brackets the root.
       - It repeatedly shrinks that interval until the root is found accurately.
    
    4) Compute the total current magnitude
       - For every feasible pair (i_d, i_q), compute:
             I_s = sqrt(i_d^2 + i_q^2)
    
    5) Keep the pair with the minimum current magnitude
       - Among all feasible (i_d, i_q) pairs that produce the requested torque,
         choose the one with the smallest I_s.
       - That is the MTPA point:
             maximum torque per ampere
       - Benefits:
             lower copper loss
             less inverter stress
             better current utilization
    
    Important note:
    - brentq is only used in step 3 to solve the inner equation for i_q.
    - The overall MTPA search is still done by scanning many candidate i_d values.
    - So the complete optimization is:
         outer search over i_d
         inner root solve over i_q using brentq
         final selection by minimum current magnitude
    """

    # def __init__(self, polepairs: int, i_max: float, n_points: int = 401, id_samples: int = 1200):
        # '''
        # :param n_points: If n_points=401, then the LUT is built at 401 torque values from 0 to maximum achievable torque.
        # :param id_samples: If id_samples = 1200, then it tries 1200 candidate d-axis currents for each torque point. (between -Imax and 0)
        # '''
        # self.p = int(polepairs)
        # self.i_max = float(i_max)
        # self.n_points = int(n_points)
        # self.id_samples = int(id_samples)

        # self.Ld0 = None
        # self.Lq0 = None
        # self.Ld_inf = None
        # self.Lq_inf = None
        # self.I_Ld_sat = None
        # self.I_Lq_sat = None

        # self.torque_grid = None
        # self.id_grid = None
        # self.iq_grid = None
        # self.torque_max = None
    
    def __init__(
        self,
        polepairs: int,
        i_max: float,
        n_points: Optional[int] = None,
        id_samples: Optional[int] = None,
        torque_points_per_nm: float = 4.0,
        id_points_per_amp: float = 10.0,
        min_torque_points: int = 121,
        max_torque_points: int = 1201,
        min_id_samples: int = 200,
        max_id_samples: int = 2000,
    ):
        """
        MTPA LUT grid settings.

        If n_points is None:
            the torque grid size is chosen automatically from torque_max.

        If id_samples is None:
            the d-axis search grid size is chosen automatically from Imax.

        Defaults:
            torque_points_per_nm = 4.0  -> about 0.25 Nm spacing
            id_points_per_amp    = 10.0 -> about 0.1 A spacing
        """
        self.p = int(polepairs)
        self.i_max = float(i_max)

        # May be None. If None, build() will choose automatically.
        self.n_points = None if n_points is None else int(n_points)
        self.id_samples = None if id_samples is None else int(id_samples)

        # Automatic sizing settings
        self.torque_points_per_nm = float(torque_points_per_nm)
        self.id_points_per_amp = float(id_points_per_amp)

        self.min_torque_points = int(min_torque_points)
        self.max_torque_points = int(max_torque_points)
        self.min_id_samples = int(min_id_samples)
        self.max_id_samples = int(max_id_samples)

        self.Ld0 = None
        self.Lq0 = None
        self.Ld_inf = None
        self.Lq_inf = None
        self.I_Ld_sat = None
        self.I_Lq_sat = None

        self.torque_grid = None
        self.id_grid = None
        self.iq_grid = None
        self.torque_max = None

    def configure_inductance_model(self, Ld0: float, Lq0: float, Ld_inf: float, Lq_inf: float, I_Ld_sat: float, I_Lq_sat: float) -> None:
        """
        This is a bounded monotonic saturation law.

        - at very small current, inductance is near Ld0, Lq0
        - at high current, inductance approaches Ld_inf, Lq_inf

        - I_Ld_sat and I_Lq_sat are current-scale parameters that control how quickly inductance drops as saturation develops.
            - if I_Ld_sat is small, then L_d drops quickly as current increases.
            - if it is large, then L_d changes more slowly.

        Saturation-law parameters:
            Ld(id) = Ld_inf + (Ld0 - Ld_inf)/(1 + |id|/I_Ld_sat)
            Lq(iq) = Lq_inf + (Lq0 - Lq_inf)/(1 + |iq|/I_Lq_sat)
        """
        self.Ld0 = float(Ld0)
        self.Lq0 = float(Lq0)
        self.Ld_inf = float(Ld_inf)
        self.Lq_inf = float(Lq_inf)
        self.I_Ld_sat = float(I_Ld_sat)
        self.I_Lq_sat = float(I_Lq_sat)

        if not (self.Ld0 > 0 and self.Lq0 > 0 and self.Ld_inf > 0 and self.Lq_inf > 0):
            raise ValueError("All inductances must be positive.")
        if not (self.Ld0 >= self.Ld_inf and self.Lq0 >= self.Lq_inf):
            # This enforces the assumption that saturation reduces inductance.
            raise ValueError("Need Ld0 >= Ld_inf and Lq0 >= Lq_inf.")
        if not (self.I_Ld_sat > 0 and self.I_Lq_sat > 0):
            # Current scales must be positive because they are denominators and represent real current magnitudes.
            raise ValueError("Saturation current scales must be positive.")

    def ld(self, id_val: float) -> float:
        """
        Equation: Ld(id) = Ld_inf + (Ld0 - Ld_inf)/(1 + |id|/I_Ld_sat)
        """
        x = abs(float(id_val))
        return self.Ld_inf + (self.Ld0 - self.Ld_inf) / (1.0 + x / self.I_Ld_sat)

    def lq(self, iq_val: float) -> float:
        """
        Equation: Lq(iq) = Lq_inf + (Lq0 - Lq_inf)/(1 + |iq|/I_Lq_sat)
        """
        x = abs(float(iq_val))
        return self.Lq_inf + (self.Lq0 - self.Lq_inf) / (1.0 + x / self.I_Lq_sat)

    def torque(self, id_val: float, iq_val: float, phi_f: float) -> float:
        """
        Equation: Te = 1.5*p*(phi_f*iq + (Ld(id)-Lq(iq))*id*iq)
        """
        Ld_now = self.ld(id_val)
        Lq_now = self.lq(iq_val)
        return 1.5 * self.p * (phi_f * iq_val + (Ld_now - Lq_now) * id_val * iq_val)

    def _solve_iq_for_torque(self, torque_ref: float, id_val: float, phi_f: float) -> Optional[float]:
        # _ = this is a helper/private method, it is intended to be used inside the class, it is not part of the public API
        iq_lim = np.sqrt(max(self.i_max ** 2 - id_val ** 2, 0.0))
        if iq_lim <= 1e-12:
            return None

        def f(iq_val: float) -> float:
            return self.torque(id_val, iq_val, phi_f) - torque_ref

        '''
        This checks torque error at:
        iq = 0
        iq = iq_lim
        The root will be searched in this interval
        '''
        f0 = f(0.0)
        f1 = f(iq_lim)

        if abs(f0) < 1e-14:
            # That mostly happens for zero torque.
            return 0.0
        if f0 * f1 > 0.0:
            # This means the function does not cross zero between 0 and iq_lim. (too high i_q or unfeasible)
            # You only call brentq when the root is trapped between two points with opposite sign.
            return None

        # It finds the iq such that torque exactly matches the reference for that fixed id
        return brentq(f, 0.0, iq_lim, xtol=1e-7, rtol=1e-5, maxiter=100)

    def solve_mtpa_point(self, torque_ref: float, phi_f: float) -> Tuple[float, float]:
        '''
        :return: best = (id_val, iq_val)
        '''
        if torque_ref <= 0.0:
            return 0.0, 0.0

        best = None
        best_is = np.inf  # best_is will hold the smallest current magnitude found so far.

        for id_val in np.linspace(-self.i_max, 0.0, self.id_samples):
            iq_val = self._solve_iq_for_torque(torque_ref, id_val, phi_f)
            if iq_val is None:
                # If no feasible iq exists for that id, skip it.
                continue

            is_val = np.hypot(id_val, iq_val)  # sqrt(id_val**2 + iq_val**2) >> minimising this
            if is_val < best_is:
                best_is = is_val
                best = (id_val, iq_val)

        if best is None:
            # This means the requested torque cannot be produced inside the current circle.
            raise RuntimeError(
                f"No feasible MTPA point for torque_ref={torque_ref:.6f} Nm."
            )

        return best

    def estimate_torque_max(self, phi_f: float) -> float:
        '''
        - This estimates the largest torque achievable under the current limit.
        :return: searching along the circle boundary is a good way to estimate torque_max
        '''
        best_torque = 0.0
        # for id_val in np.linspace(-self.i_max, 0.0, self.id_samples):
        id_samples_for_estimate = self.id_samples
        if id_samples_for_estimate is None:
            id_samples_for_estimate = proportional_grid_size(
                span=self.i_max,
                points_per_unit=self.id_points_per_amp,
                min_points=self.min_id_samples,
                max_points=self.max_id_samples,
            )

        for id_val in np.linspace(-self.i_max, 0.0, id_samples_for_estimate):
            iq_lim = np.sqrt(max(self.i_max ** 2 - id_val ** 2, 0.0))
            tq = self.torque(id_val, iq_lim, phi_f)
            if tq > best_torque:
                best_torque = tq
        return best_torque

    # def build(self, phi_f: float) -> None:
    #     '''
    #     - Stored LUT arrays are created here = This is the startup step that actually constructs the lookup table.
    #     '''
    #     self.torque_max = self.estimate_torque_max(phi_f)
    #     self.torque_grid = np.linspace(0.0, self.torque_max, self.n_points)
    #     self.id_grid = np.zeros_like(self.torque_grid)
    #     self.iq_grid = np.zeros_like(self.torque_grid)

    #     for k, tq in enumerate(self.torque_grid):
    #         id_star, iq_star = self.solve_mtpa_point(tq, phi_f)
    #         self.id_grid[k] = id_star
    #         self.iq_grid[k] = iq_star

    def build(self, phi_f: float) -> None:
        '''
        - Stored LUT arrays are created here.
        - If n_points or id_samples are None, they are automatically chosen.
        '''
        # Choose id_samples first, because estimate_torque_max() uses it.
        if self.id_samples is None:
            self.id_samples = proportional_grid_size(
                span=self.i_max,
                points_per_unit=self.id_points_per_amp,
                min_points=self.min_id_samples,
                max_points=self.max_id_samples,
            )

        # Estimate max torque using the chosen id grid.
        self.torque_max = self.estimate_torque_max(phi_f)

        # Choose torque-grid size from the torque span.
        if self.n_points is None:
            self.n_points = proportional_grid_size(
                span=self.torque_max,
                points_per_unit=self.torque_points_per_nm,
                min_points=self.min_torque_points,
                max_points=self.max_torque_points,
            )

        self.torque_grid = np.linspace(0.0, self.torque_max, self.n_points)
        self.id_grid = np.zeros_like(self.torque_grid)
        self.iq_grid = np.zeros_like(self.torque_grid)

        for k, tq in enumerate(self.torque_grid):
            id_star, iq_star = self.solve_mtpa_point(tq, phi_f)
            self.id_grid[k] = id_star
            self.iq_grid[k] = iq_star

    def lookup(self, torque_cmd: float) -> Tuple[float, float]:
        '''
        - Given a requested torque command, return the corresponding optimal (i_d, i_q) using interpolation.
        - So the stored LUT covers only positive torque, and negative torque is handled symmetrically.
        :return: id_star, iq_star
        '''
        if self.torque_grid is None:
            # This prevents using the LUT before calling build(...)
            raise RuntimeError("MTPA LUT not built.")

        sgn = np.sign(torque_cmd) if torque_cmd != 0.0 else 1.0
        tq = np.clip(abs(torque_cmd), self.torque_grid[0], self.torque_grid[-1])

        id_star = np.interp(tq, self.torque_grid, self.id_grid)
        iq_star = np.interp(tq, self.torque_grid, self.iq_grid)

        return id_star, sgn * iq_star