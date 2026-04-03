# src/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Core directories
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Boolean model specific
BM_DIR = PROJECT_ROOT / "boolean_models"

BM_RESULTS_DIR = RESULTS_DIR / "boolean_models"
BM_FIG_DIR = FIGURES_DIR / "boolean_models"

PARAM_FIG_DIR = BM_FIG_DIR / "param_sweep_figs"
PERB_FIG_DIR = BM_FIG_DIR / "perturbation_figs"

# ABM specific
ABM_SIM_RES_DIR = RESULTS_DIR / "abm_sim"
ABM_SWEEP_RES_DIR = RESULTS_DIR / "abm_sweep"

HILL_RES_DIR = RESULTS_DIR / "analyse_hill"

# Russ Experimental Data
EXP_DIR = PROJECT_ROOT / "russ_results" / "Elongation"