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
)


def run_param_sweep(model, result_dir, config, perturbation='WT'):
    # Extract variables from config
    sweep_cfg = config['sensitivity_analysis']
    param_cfg = sweep_cfg['parameter_ranges']
    groups = sweep_cfg['groups']

    # Identify which parameters to sweep based on the profile
    profile = sweep_cfg['sweep_profiles'][perturbation]

    # Initiate list to store parameter sweep results
    result = []
    
    # Run parameter sweep for each parameter (for each group)
    print(f"DEGUG: Starting sweep for perturbation {perturbation}")
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

                delta_df = ss_df.copy()
                delta_df['delta'] = compute_delta(delta_df, config)

                phenotype_df = delta_df.copy()
                phenotype_df['phenotype'] = delta_df['delta'].apply(lambda x: classify_phenotype(x, config))

                result.append(phenotype_df)

    print(f"DEBUG: Sweep of parameters for {perturbation} completed. ")

    combined_df = pd.concat(result, ignore_index=True)
    save_df_to_csv(combined_df, result_dir, f"{perturbation}_param_sweep")

    return combined_df
