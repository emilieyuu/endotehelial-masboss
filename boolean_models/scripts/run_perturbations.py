import maboss
import pandas as pd
from pathlib import Path
import shutil
import yaml

from boolean_models.analysis import (
    compute_delta,
    classify_phenotype,
    save_df_to_csv, 
    generate_ko_model
)

# --------------------------------------------------
# Run perturbation
# --------------------------------------------------
def run_perturbations(base_model, result_dir, config):
    """
    Run MaBoSS simulations for WT and all KO perturbations, RhoA/RhoC balance.
    """
    # Get dictionary of perturbations from config
    perbs_dict = config['perturbations']

    # Initiate list to store perturbation results
    perbs = []

    # Run simulations. 
    for name, mutation in perbs_dict.items():
        print(f"DEBUG: Running perturbation: {name}")

        # Create model of KO scenario
        m = generate_ko_model(base_model, mutation)

        # Run MaBoSS
        res = m.run()
        prob_df = res.get_nodes_probtraj().rename_axis('t').reset_index()
        prob_df['perturbation'] = name

        # Compute Rho balance (delta)
        balance_df = prob_df.copy()  
        balance_df["delta"] = compute_delta(balance_df, config)
        
        # Classify respective phenotypes
        phenotype_df = balance_df.copy()
        phenotype_df['phenotype'] = balance_df['delta'].apply(lambda x: classify_phenotype(x, config))

        perbs.append(phenotype_df)

    print("DEBUG: All simulations completed successfully")

    full_perb_df = pd.concat(perbs, ignore_index=True)
    save_df_to_csv(full_perb_df, result_dir, "perturbation_timeseries")

    return full_perb_df

if __name__ == "__main__":
    perb_dict = run_perturbations()