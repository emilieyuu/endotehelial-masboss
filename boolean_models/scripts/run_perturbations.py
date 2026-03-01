import maboss
import pandas as pd

from boolean_models.analysis import (
    compute_delta,
    classify_phenotype,
    save_df_to_csv, 
    generate_ko_model
)

# --------------------------------------------------
# Run perturbation
# --------------------------------------------------
def run_perturbations(base_model, cfg, result_dir=None):
    """
    Run MaBoSS simulations for all pertubations.

    param base_model (maboss.simulation): MasBoSS model to modify for each perb. 
    param cfg (dict-like): config containing simualtion details.
    param result_dir (Path): Optionally save sim result as csv. 

    return full_perb_df (DataFrame): Full timeseries for each perturbation. 
    return ss_df (DataFrame): Final steady state probabilities only. 
    """
    # Get dictionary of perturbations from config
    perbs_dict = cfg['perturbations']

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
        balance_df["delta"] = compute_delta(balance_df, cfg)
        
        # Classify respective phenotypes
        phenotype_df = balance_df.copy()
        phenotype_df['phenotype'] = balance_df['delta'].apply(lambda x: classify_phenotype(x, cfg))

        perbs.append(phenotype_df)

    print("DEBUG: All simulations completed successfully")

    # DataFrame timeseries of all perturbations
    full_perb_df = pd.concat(perbs, ignore_index=True)

    # DataFrame of steadystate node probabilities only
    ss_mask = full_perb_df.groupby('perturbation')['t'].idxmax()
    ss_df = full_perb_df.loc[ss_mask].reset_index(drop=True)


    if result_dir is not None: 
        save_df_to_csv(full_perb_df, result_dir, "perturbation_timeseries")
        save_df_to_csv(ss_df, result_dir, "perturbation_steady_state")

    return full_perb_df, ss_df