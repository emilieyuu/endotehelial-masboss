import maboss
import pandas as pd
from pathlib import Path
import shutil

from boolean_models.analysis import (
    generate_ko_models,
    compute_delta,
    classify_phenotype,
    save_df_to_csv
)

# ==================================================
# PATH CONFIGURATION
# ==================================================
# Define all paths relative to this file for reproducebility
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
MODELS_DIR = PROJECT_ROOT / "boolean_models" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "boolean_models" / "rho_model"


# --------------------------------------------------
# Model definition files
# --------------------------------------------------
model_file = MODELS_DIR / "rho.bnd"
cfg_file   = MODELS_DIR / "rho_base.cfg"

# --------------------------------------------------
# Specific result directories
# --------------------------------------------------
raw_dir = RESULTS_DIR / "raw"
processed_dir = RESULTS_DIR / "processed"
ss_dir = RESULTS_DIR / "steady_state"

# --------------------------------------------------
# Perturbation (KO) nodes
# --------------------------------------------------
# Treated as external / upstream inputs
PERB_NODES = ['DSP', 'TJP1', 'JCAD']

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def extract_steady_state(dict):
    processed_dfs = []
    for name, df in dict.items():
        tail_df = (df.tail(1).copy().assign(scenario=name))
        processed_dfs.append(tail_df)

    return pd.concat(processed_dfs, ignore_index=True)
    
# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    """
    Run MaBoSS simulations for WT and all KO perturbations, store raw trajectories, and compute RhoA/RhoC balance.
    """
    # Load base WT model
    base_model = maboss.load(str(model_file), str(cfg_file))
    print("DEUBG: Loaded base MaBoSS model")

    # Generate KO Perturbation
    ko_scenarios = generate_ko_models(base_model, PERB_NODES)
    print(f"DEUBG: Generated {len(ko_scenarios)} perturbation scenarios")

    # Build result directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("DEUBG: Built result directories.")

    # Dictionary storing node probability + delta trajectories
    perb_dict = {}
    phenotype_dict = {}

    # Run simulations. 
    for name, model in ko_scenarios.items():
        print(f"DEBUG: Running scenario: {name}")

        # Run MaBoSS
        res = model.run()

        # Basic sanity check
        prob_df = res.get_nodes_probtraj().rename_axis('t').reset_index()
        save_df_to_csv(prob_df, raw_dir, f"{name}_nodes_probraj.csv")
        
        # Compute Rho balance
        balance_df = prob_df.copy()  
        balance_df["delta"] = compute_delta(balance_df)
        save_df_to_csv(balance_df, processed_dir, f"{name}_balance.csv")

        perb_dict[name] = balance_df

        # Compute raw phenotypes
        phenotype_df = classify_phenotype(balance_df)  
        save_df_to_csv(phenotype_df, processed_dir, f"{name}_phenotype.csv")

        phenotype_dict[name] = phenotype_df

    print("DEBUG: All simulations completed successfully")

    # Compute and save combined steady state data
    balance_concat_df = extract_steady_state(perb_dict)
    pheno_concat_df = extract_steady_state(phenotype_dict)

    save_df_to_csv(balance_concat_df, ss_dir, f"steady_state_balance.csv")
    save_df_to_csv(pheno_concat_df, ss_dir, f"steady_state_phenotype.csv")
    print("DEBUG: Processed steady state data saved sucessfully")

if __name__ == "__main__":
    main()