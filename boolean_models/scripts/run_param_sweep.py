#boolean_models/scripts/run_param_sweep.py
import maboss
import pandas as pd
import numpy as np
import itertools

from boolean_models.analysis import (
    compute_delta,
    classify_phenotype,
    save_df_to_csv, 
    generate_ko_model
)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def build_ranges(sweep_config):
    # Extract group, range and step info
    ranges = sweep_config['ranges']
    groups = sweep_config['groups']

    # Initiate list to store parameter and values
    param_dict = {}

    for group in groups:
        for p, info in ranges.items(): 
            # Construct parameter name
            p_name = f"${group}_{p}"

            # Setup sweeping values
            values = np.arange(info['start'], info['stop'], info['step'])
            param_dict[p_name] = values
            
    return param_dict


# --------------------------------------------------
# Experiment Runs
# --------------------------------------------------
def run_1d_sweeps(base_model, exp_config, perb_config, param_values):
    """
    Run all 1D experiments, and return a combined DataFrame with results. 
    
    :param base_model: Description
    :param exp_config: Description
    :param perb_config: Description
    :param param_values: Description
    """
    # Initiate list to store results
    results = []

    for perb, params in exp_config.items():
        # Create model for KO scenaro
        perb_model = generate_ko_model(base_model, perb_config[perb])

        # Resolve "all" to parameter list
        if params == 'all': 
            params = param_values.keys()

        print(f"DEBUG: Starting 1D sweep for perturbation: {perb} with parameter {params}")

        for p in params:
            # Get values to sweep for parameter
            values = param_values[p]

            for v in values: 
                # Setup model with adjusted parameter values
                m_temp = perb_model.copy()
                m_temp.update_parameters(**{p: v})

                # Run model and extract steady state probabilities
                res = m_temp.run()
                ss_df = res.get_last_nodes_probtraj()

                # Append info columns
                ss_df[['p_name', 'p_value', 'perb']] = pd.DataFrame([[p, v, perb]], index=ss_df.index)

                results.append(ss_df)

        print("DEBUG: All 1D sweeps completed")
    
    # Return dataframe of concatenated results for all 1D experiments
    return pd.concat(results)


def run_2d_sweep(base_model, exp, perb_config, param_values):
    """
    Run all 2D experiments, and return a combined DataFrame with results.
    
    :param base_model: Description
    :param exp_config: Description
    :param perb_config: Description
    :param param_values: Description
    """
    # Initiate list to store results
    results = []
    
    # Extract experiment information
    exp_name = exp['name']
    perbs = exp['perturbations']
    p1, p2 = exp['parameters']

    print(f"DEBUG: Starting 2D sweep experiment {exp_name} with parameters {p1, p2}")

    for perb in perbs:
        # Create model for KO scenaro
        perb_model = generate_ko_model(base_model, perb_config[perb])

        # Setup sweeping ranges
        v1_list = param_values[p1]
        v2_list = param_values[p2]
        
        print(f"DEBUG: Computed values: {p1}: {v1_list}, {p2}: {v2_list}")
        
        # Run model for each combination of parameter values (cartesian product)
        for v1, v2 in itertools.product(v1_list, v2_list):
            
            # Setup model with adjusted parameter values
            m_temp = perb_model.copy()
            m_temp.update_parameters(**{p1: v1, p2: v2})

            # Run model and extract steady state probabilities
            res = m_temp.run()
            ss_df = res.get_last_nodes_probtraj()

            # Append info columns
            ss_df[[p1, p2, 'perb', 'exp_name']] = pd.DataFrame([[v1, v2, perb, exp_name]], index=ss_df.index)

            results.append(ss_df)
            
        print(f"DEBUG: Completed {exp_name} sweep for perturbation: {perb}")

    return pd.concat(results)

# --------------------------------------------------
# Full Combined Param Sweep
# --------------------------------------------------
def run_sweeps(base_model, result_dir, sweep_config, sim_config):
    perb_config = sim_config['perturbations']
    param_values = build_ranges(sweep_config)
    exp_1d_cfg = sweep_config['1D_experiments']
    exp_2d_cfg = sweep_config['2D_experiments']

    sweep_results = {}
    sweep_results["1D_sweeps"] = run_1d_sweeps(base_model, exp_1d_cfg, perb_config, param_values)

    for exp in exp_2d_cfg:
        sweep_results[f"2D_{exp['name']}"] = run_2d_sweep(base_model, exp, perb_config, param_values)
    
    for name, df in sweep_results.items():
        df['delta'] = compute_delta(df, sim_config)
        df['phenotype'] = df['delta'].apply(lambda x: classify_phenotype(x, sim_config))
        save_df_to_csv(df, result_dir, name)

    return sweep_results

