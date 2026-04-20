# src/utils/config_utils.py
#
# Reusable helpers for config loading and manipulation 

import copy
import yaml
from src.paths import CONFIG_DIR

# ------------------------------------------------------------------
# Config Loaders
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Require: Config Access Assist
# ------------------------------------------------------------------
def require(cfg, *keys):
    """
    Safely restrive a nested config value. 
    Raises KeyError with a clear path if any key is missing. 

    cfg: dictionary
    keys: separated sequence of dict keys
    """
    value = cfg

    # Traverse dictionary with keys
    for i, key in enumerate(keys):
        # Raise error if key is missing
        if not isinstance(value, dict) or key not in value: 
            path = '.'.join(keys[:i+1])
            raise KeyError(f"Missing required config key: {path}")
        # Index into dictionary
        value = value[key]
        
    return value
