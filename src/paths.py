# src/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# Core Directories
# ------------------------------------------------------------------
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ------------------------------------------------------------------
# Boolean Model Directories
# ------------------------------------------------------------------
BM_DIR = PROJECT_ROOT / "boolean_models"

BM_RESULTS_DIR = RESULTS_DIR / "boolean_models"

# --- Figures ---
PARAM_FIG_DIR = FIGURES_DIR / "bm_sweep_figs"
PERB_FIG_DIR = FIGURES_DIR / "bm_sim_figs"

# ------------------------------------------------------------------
# LUT Directory
# ------------------------------------------------------------------
RESULTS_DIR / "lut_csv"

# ------------------------------------------------------------------
# ABM Directories
# ------------------------------------------------------------------
ABM_RESULTS_DIR = RESULTS_DIR / "abm"

ABM_RESULTS_ARCH = ABM_RESULTS_DIR / "archive"
ABM_SIM_RES_DIR = ABM_RESULTS_DIR / "sim" 
ABM_SWEEP_RES_DIR = ABM_RESULTS_DIR / "sweep"

# --- Figures ---
ABM_SIM_FIG_DIR = FIGURES_DIR / "abm_sim_figs" 
ABM_SWEEP_FIG_DIR = FIGURES_DIR / "abm_sweep_figs"

# ------------------------------------------------------------------
# Additional Directories
# ------------------------------------------------------------------

# Russ Experimental Data
EXP_DIR = PROJECT_ROOT / "russ_results" / "Elongation"
ANALYSIS_DIR = PROJECT_ROOT / "data_analysis"