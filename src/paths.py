# src/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# Core Directories
# ------------------------------------------------------------------
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ------------------------------------------------------------------
# Model Definition Files
# ------------------------------------------------------------------
MABOSS_DIR = MODELS_DIR / "maboss"

# ------------------------------------------------------------------
# Data Directories
# ------------------------------------------------------------------
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXP_DIR = RAW_DATA_DIR / "russ" / "Elongation"

# ------------------------------------------------------------------
# Boolean Model Directories
# ------------------------------------------------------------------
BM_RESULTS_DIR = RESULTS_DIR / "boolean_model"
BM_SIM_RES_DIR = BM_RESULTS_DIR / "sim"
BM_SWEEP_RES_DIR = BM_RESULTS_DIR / "sweep"

BM_FIG_DIR = FIGURES_DIR / "boolean_model"
BM_SIM_FIG_DIR = BM_FIG_DIR / "sim"
BM_SWEEP_FIG_DIR = BM_FIG_DIR / "sweep"

# ------------------------------------------------------------------
# ABM Directories
# ------------------------------------------------------------------
LUT_DIR = RESULTS_DIR / "lut"

ABM_RESULTS_DIR = RESULTS_DIR / "abm"
ABM_RESULTS_ARCH = ABM_RESULTS_DIR / "archive"
ABM_SIM_RES_DIR = ABM_RESULTS_DIR / "sim"
ABM_SWEEP_RES_DIR = ABM_RESULTS_DIR / "sweep"

ABM_FIG_DIR = FIGURES_DIR / "abm"
ABM_FIG_ARCH = ABM_FIG_DIR / "archive"
ABM_SIM_FIG_DIR = ABM_FIG_DIR / "sim"
ABM_SWEEP_FIG_DIR = ABM_FIG_DIR / "sweep"

# ------------------------------------------------------------------
# Comparison / Report Figures
# ------------------------------------------------------------------
COMPARISON_FIG_DIR = FIGURES_DIR / "comparison"
REPORT_FIG_DIR = FIGURES_DIR / "report"