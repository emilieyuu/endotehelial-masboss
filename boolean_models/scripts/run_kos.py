import maboss
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from boolean_models.analysis import (
    generate_ko_models,
    compute_delta,
    classify_phenotype,
)

# ==================================================
# PATH CONFIGURATION
# ==================================================
# Define all paths relative to this file for reproducebility
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
MODELS_DIR = PROJECT_ROOT / "boolean_models" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "boolean_models"


# --------------------------------------------------
# Model definition files
# --------------------------------------------------
model_file = MODELS_DIR / "rho.bnd"
cfg_file   = MODELS_DIR / "rho_base.cfg"

# --------------------------------------------------
# Perturbation (KO) nodes
# --------------------------------------------------
# Treated as external / upstream inputs
PERB_NODES = ['DSP', 'TJP1', 'JCAD']

def save_sim_details(result, result_directory, perb_name):
    """
    Save raw simulation outputs for a single perturbation. 

    :param result (MaBoSS.Result): Completed MaBoSS simulation object.
    :param result_directory (Path): Directory where outputs should be written.
    :param perb_name (str): Name of the perturbation scenario.
    """
    # Node activation probability trajectories
    nodes_probtraj = result.get_nodes_probtraj().rename_axis('t').reset_index()    
    nodes_probtraj.to_csv(
        result_directory / f"{perb_name}_nodes_probtraj.csv",
        index=False
    )

    # Full Boolean state probability trajectories
    states_probtraj = result.get_states_probtraj().rename_axis('t').reset_index()    
    states_probtraj.to_csv(
        result_directory / f"{perb_name}_states_probtraj.csv",
        index=False
    )

def main():
    """
    Run MaBoSS simulations for WT and all KO perturbations, store raw trajectories, and compute RhoA/RhoC balance.
    """

    # Load base WT model. 
    base_model = maboss.load(str(model_file), str(cfg_file))
    print("DEUBG: Loaded base MaBoSS model")

    # Generate KO Perturbation.
    ko_scenarios = generate_ko_models(base_model, PERB_NODES)
    print("DEUBG: Generated {len(ko_scenarios)} perturbation scenarios")

    # Dictionary storing node probability + delta trajectories.
    perb_dfs = {}

    # Run simulations. 
    for name, model in ko_scenarios.items():
        print(f"DEBUG: Running scenario: {name}")

        res_dir = RESULTS_DIR / "rho_model" / name
        
        # Ensure clean output directory
        if res_dir.exists():
            shutil.rmtree(res_dir)
        res_dir.mkdir(parents=True, exist_ok=True)

        # Run MaBoSS
        res = model.run()

        # Basic sanity check
        prob_df = res.get_nodes_probtraj()
        assert "RhoA" in prob_df.columns, "RhoA missing from output"
        assert "RhoC" in prob_df.columns, "RhoC missing from output"

        # Save raw simulation details
        save_sim_details(res, res_dir, name)
        
        # Compute Rho balance
        balance_df = prob_df.copy()
        balance_df["delta"] = prob_df["RhoC"] - prob_df["RhoA"]

        perb_dfs[name] = balance_df
        
    print("DEBUG: All simulations completed successfully")

if __name__ == "__main__":
    main()