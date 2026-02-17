#boolean_models/scripts/run_param_sweep.py
import maboss
import pandas as pd
from pathlib import Path
import yaml
import numpy as np

from boolean_models.analysis import (
    compute_delta,
    classify_phenotype,
    save_df_to_csv, 
    generate_ko_model
)


def run_param_sweep(base_model, result_dir, config):
    # Extract variables from config
    perb_cfg = config['perturbations']
    sweep_cfg = config['sensitivity_analysis']
    param_cfg = sweep_cfg['parameter_ranges']
    groups = sweep_cfg['groups']

    # Initiate list to store parameter sweep results
    result = []

    for perb, mutation in perb_cfg.items():
        # Create model of KO scenario
        model = generate_ko_model(base_model, mutation)

        # Identify which parameters to sweep based on the profile
        profile = sweep_cfg['sweep_profiles'][perb]
        
        # Run parameter sweep for each parameter (for each group)
        print(f"DEGUG: Starting sweep for perturbation {perb}")
        for group, features in groups.items(): 
            # For each parameter
            for p in features['parameters']:
                name = f"${group}_{p}"

                # Skip parameters in non-exhaustive KO scenarios
                if profile != "exhaustive" and name not in profile: 
                    continue
                
                # Setup sweeping ranges
                values = np.arange(param_cfg[p]['range'][0], param_cfg[p]['range'][1], param_cfg[p]['step'])

                print(f"DEGUG: Performing sweep for parameter {name} with values: {values}")

                # Perform sweep
                for val in values: 
                    m = model.copy()
                    m.update_parameters(**{name: val})

                    res = m.run()
                    ss_df = res.get_last_nodes_probtraj()
                    ss_df['param_value'] = val
                    ss_df['param_name'] = name
                    ss_df['perturbation'] = perb

                    delta_df = ss_df.copy()
                    delta_df['delta'] = compute_delta(delta_df, config)

                    phenotype_df = delta_df.copy()
                    phenotype_df['phenotype'] = delta_df['delta'].apply(lambda x: classify_phenotype(x, config))

                    result.append(phenotype_df)

        print(f"DEBUG: Sweep of parameters for {perb} completed. ")

    combined_df = pd.concat(result, ignore_index=True)
    save_df_to_csv(combined_df, result_dir, "full_1D_param_sweep")

    return combined_df
