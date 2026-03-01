#boolean_models/scripts/run_param_sweep.py
#
# Run Full MaBoSS pipeline: Perturbation + Param Sweep

import maboss
from pathlib import Path

from src.config import load_sim_config, load_sweep_config
from src.paths import *
from boolean_models.scripts import run_perturbations, run_sweeps

sim_cfg = load_sim_config()
sweep_cfg = load_sweep_config()

MODELS_BND = DATA_DIR / sim_cfg['model']['bnd']
MODELS_CFG = DATA_DIR / sim_cfg['model']['cfg_v3']
base_model = maboss.load(str(MODELS_BND), str(MODELS_CFG))
base_model.param['max_time'] = 10.0
base_model.param['sample_count'] = 5000


if __name__ == "__main__":
    perb_df = run_perturbations(base_model, RESULTS_DIR, sim_cfg)
    sweep_df = run_sweeps(base_model, RESULTS_DIR, sweep_cfg, sim_cfg)

