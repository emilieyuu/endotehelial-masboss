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
# Helpers
# --------------------------------------------------
# def extract_steady_state(dict):
#     processed_dfs = []
#     for name, df in dict.items():
#         tail_df = (df.tail(1).copy().assign(scenario=name))
#         processed_dfs.append(tail_df)

#     return pd.concat(processed_dfs, ignore_index=True)
# --------------------------------------------------
# Run perturbation
# --------------------------------------------------
def run_perturbations(model, result_dir, config):
    """
    Run MaBoSS simulations for WT and all KO perturbations, RhoA/RhoC balance.
    """
    perbs_dict = config.get('perturbations')

    # Load base WT model
    base_model = model #maboss.load(str(MODELS_BND), str(MODELS_CFG))
    print("DEUBG: Loaded base MaBoSS model")

    # Build result directories
    # for d in [RAW_DIR, PROCESSED_DIR, SS_DIR]:
    #     d.mkdir(parents=True, exist_ok=True)
    # print("DEUBG: Built result directories.")

    # Dictionary storing node probability + delta trajectories
    # perb_dict = {}
    # phenotype_dict = {}
    perbs = []

    # Run simulations. 
    for name, mutation in perbs_dict.items():
        print(f"DEBUG: Running scenario: {name}")

        # Create model of KO scenario
        m = generate_ko_model(base_model, mutation)

        # Run MaBoSS
        res = m.run()
        prob_df = res.get_nodes_probtraj().rename_axis('t').reset_index()
        prob_df['perturbation'] = name
    

        # Compute Rho balance
        balance_df = prob_df.copy()  
        balance_df["delta"] = compute_delta(balance_df, config)
        
        phenotype_df = balance_df.copy()
        phenotype_df['phenotype'] = balance_df['delta'].apply(lambda x: classify_phenotype(x, config))

        perbs.append(phenotype_df)
        # save_df_to_csv(balance_df, PROCESSED_DIR, f"{name}_balance.csv")

        # # perb_dict[name] = balance_df
        
        # phenotype_df = classify_phenotype(balance_df)
        # phenotype_dict[name] = phenotype_df
        # #print(phenotype_df)

        # ss_df = res.get_last_nodes_probtraj()
        # ss_df['scenario'] = name
        # perbs_ss.append(ss_df)

    print("DEBUG: All simulations completed successfully")

    full_perb_df = pd.concat(perbs, ignore_index=True)
    save_df_to_csv(full_perb_df, result_dir, "perturbation_timeseries.csv")

    # # Compute and save combined steady state data
    # ss_concat_df = pd.concat(perbs_ss)
    # save_df_to_csv(ss_concat_df, PROCESSED_DIR, f"steady_state_node_probtraj.csv")

    # #Temp for testing
    # balance_concat_df = extract_steady_state(perb_dict)
    # save_df_to_csv(balance_concat_df, PROCESSED_DIR, f"steady_state_balance.csv")

    # pheno_concat_df = extract_steady_state(phenotype_dict)
    # save_df_to_csv(pheno_concat_df, PROCESSED_DIR, f"steady_state_phenotype.csv")

    # print("DEBUG: Processed steady state data saved sucessfully")
    
    return full_perb_df

if __name__ == "__main__":
    perb_dict = run_perturbations()