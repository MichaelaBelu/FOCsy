from pmsm_backend.runner import run_pmsm_simulation
from pmsm_gui.settings import DEFAULTS, rpm_to_rad_s

params = dict(DEFAULTS)
params["machine_type"] = "ipmsm"
params["use_mtpa"] = True
params["use_field_weakening"] = True
params["t_steps"] = [0.0]
params["omega_steps"] = [rpm_to_rad_s(3000.0)]

result = run_pmsm_simulation(params)

print(result.keys())
