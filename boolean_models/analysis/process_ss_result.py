#process_ss_result.py
import maboss
import pandas as pd
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
MODELS_DIR = PROJECT_ROOT / "boolean_models" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "boolean_models" / "rho_model"

ss_dir = RESULTS_DIR / "steady_state"


ss_dict = {}
for file_path in ss_dir.iterdir(): 
    name = file_path.stem.replace("steady_state_", "")
    df = pd.read_csv(file_path, index_col=[0])
    ss_dict[name] = df


pheno_df = ss_dict['phenotype']
balance_df = ss_dict['balance']
pheno_df_long = df.melt(id_vars="scenario", 
                  value_vars=["Failed", "Hyper", "Normal"], 
                  var_name="phenotype", 
                  value_name="value")
pheno_df_final = pheno_df_long[pheno_df_long["value"] == 1].drop(columns="value").reset_index(drop=True)

merged = pheno_df_final.merge(
    balance_df[['scenario', 'delta']],
    on="scenario",
    how="left"
)
merged

