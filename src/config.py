import yaml
from .paths import CONFIG_DIR

def load_sim_config():
    with open(CONFIG_DIR / "rho_sim_config.yaml") as f:
        return yaml.safe_load(f)

def load_sweep_config():
    with open(CONFIG_DIR / "parameter_sweep_config.yaml") as f:
        return yaml.safe_load(f)
    
def load_spatial_config(): 
    with open(CONFIG_DIR / "spatial_config.yaml") as f:
        return yaml.safe_load(f)