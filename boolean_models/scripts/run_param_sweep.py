#boolean_models/scripts/run_param_sweep.py
import maboss
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys
import yaml
import numpy as np

from boolean_models.analysis import compute_delta

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "rho_sim_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# --------------------------------------------------
# Result directories
# --------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / config['paths']['results_base']

# Result subdirectories
PARAM_DIR = RESULTS_DIR / config['paths']['subdirs']['param_sweep']

# --------------------------------------------------
# Model definition files
# --------------------------------------------------
MODELS_BND = PROJECT_ROOT / config['paths']['model_bnd']
MODELS_CFG = PROJECT_ROOT / config['paths']['model_cfg']

# --------------------------------------------------
# Global variable
# --------------------------------------------------
EPS = config['analysis']['thresholds']['eps']
PERBS_DICT = config.get('perturbations') # get mutations directly from config


