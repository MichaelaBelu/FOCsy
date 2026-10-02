# This is the backend public API used by the GUI.
#
# No structural change was needed here for the iron-loss feature: this function
# is already a generic params-dict -> PMSMConfig -> PMSMSimulation pipeline, so
# the new iron-loss keys (include_iron_loss, k_h_iron, k_e_iron, beta_iron,
# core_volume_m3, B_iron_rated) just need to be present in `params`, same as
# every other field. See example_iron_loss_usage.py for a full example call.

from __future__ import annotations

from typing import Any, Dict

from pmsm_backend.config import config_from_params
from pmsm_backend.simulation import PMSMSimulation


def run_pmsm_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config_from_params(params)
    sim = PMSMSimulation(cfg)
    return sim.run()