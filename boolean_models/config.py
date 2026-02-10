from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

MODELS_DIR = PROJECT_ROOT / "boolean_models" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "boolean_models"

MODEL_FILE = MODELS_DIR / "rho.bnd"
CFG_FILE   = MODELS_DIR / "rho_base.cfg"

PERB_NODES = ['DSP', 'TJP1', 'JCAD']
