#src/config.py

import yaml
from .paths import CONFIG_DIR

def load_bm_sim_cfg():
    with open(CONFIG_DIR / "bm_sim.yaml") as f:
        return yaml.safe_load(f)

def load_bm_sweep_cfg():
    with open(CONFIG_DIR / "bm_sweep.yaml") as f:
        return yaml.safe_load(f)
    
def load_abm_sim_cfg(): 
    with open(CONFIG_DIR / "abm_sim.yaml") as f:
        return yaml.safe_load(f)
    
def load_abm_sweep_cfg(): 
    with open(CONFIG_DIR / "abm_sweep.yaml") as f:
        return yaml.safe_load(f)