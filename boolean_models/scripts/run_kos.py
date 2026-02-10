import maboss
import pandas as pd
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

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def save_sim_raw_details(result, result_directory, perb_name):
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

def save_sim_processed(results_dict, num_rows=5, out_path=None):
    if not isinstance(results_dict, dict) or not results_dict:
        raise ValueError("results_dict must be a non-empty dict")

    processed_dfs = []
    for name, df in results_dict.items():
        if not isinstance(df, pd.DataFrame) or len(df) < num_rows:
            raise ValueError((f"'{name}' must be a DataFrame of length at least {num_rows}"))
        
        tail_df = ( df.tail(num_rows).copy().assign(scenario=name))
        processed_dfs.append(tail_df)

    concat_df = pd.concat(processed_dfs, ignore_index=True)

    if out_path is not None:
        concat_df.to_csv(out_path, index=False)

    return concat_df
    
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
    print("DEUBG: Generated {len(ko_scenarios)} perturbation scenarios")

    # Dictionary storing node probability + delta trajectories
    perb_dict = {}
    phenotype_dict = {}

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
        save_sim_raw_details(res, res_dir, name)
        
        # Compute Rho balance
        balance_df = prob_df.copy().rename_axis('t').reset_index()   
        balance_df["delta"] = compute_delta(balance_df)
        perb_dict[name] = balance_df

        # Compute raw phenotypes
        phenotype_df = classify_phenotype(balance_df).rename_axis('t').reset_index()   
        phenotype_dict[name] = phenotype_df

        # Compute raw phenotypes
        print(f"DEBUG: Writing intermediate data to file for: {name}")
        balance_df.to_csv(res_dir / f"{name}_phenotype.csv")
        phenotype_df.to_csv(res_dir / f"{name}_phenotype.csv")
    
    print("DEBUG: Saving combined processed reults.")
    save_sim_processed(perb_dict, num_rows=1, out_path=RESULTS_DIR / "rho_model" / "steady_state_balance.csv")
    save_sim_processed(phenotype_dict, num_rows=1, out_path=RESULTS_DIR / "rho_model" / "steady_state_phenotype.csv")

    print("DEBUG: All simulations completed successfully")

if __name__ == "__main__":
    main()