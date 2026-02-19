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
def build_ranges(sweep_config, resolution="fine"):
    # Extract group, range and step info
    ranges = sweep_config['ranges']
    groups = sweep_config['groups']

    # Initiate list to store parameter and values
    param_dict = {}

    for group in groups:
        for p, info in ranges.items(): 
            # Construct parameter name
            p_name = f"${group}_{p}"

            # Determine the step size based on resolution
            step = info['step']
            if resolution == "coarse":
                step *= 2

            # Generate values
            values = np.arange(info['start'], info['stop'], step)
            param_dict[p_name] = values
            
    return param_dict


# --------------------------------------------------
# Experiment Runs
# --------------------------------------------------
def run_1d_sweep_single(base_model, exp, perb_config, param_values):
    """
    Run all 1D experiments, and return a combined DataFrame with results. 
    """
    # Initiate list to store results
    results = []
    exp_name = exp['name']
    params = exp['parameters']
    perturbations = exp['perturbations']

    # Resolve "all" to parameter list
    if params == 'all': 
        params = param_values.keys()

    for perb in perturbations: 
        print(f"DEBUG: Starting {exp_name} sweep for perturbation: {perb} with {params}")

        # Create model for KO scenaro
        perb_model = generate_ko_model(base_model, perb_config[perb])

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

                # Unified Metadata columns
                ss_df['p1_name'] = p
                ss_df['p1_value'] = v
                ss_df['p2_name'] = "N/A" # Keeps DF shape consistent with 2D
                ss_df['p2_value'] = np.nan
                ss_df['perturbation'] = perb
                ss_df['exp_name'] = exp_name

                results.append(ss_df)

        print(f"DEBUG: Completed {exp_name} sweep for perturbation: {perb}")
    
    # Return dataframe of concatenated results for all 1D experiments
    return pd.concat(results)


def run_2d_sweep_single(base_model, exp, perb_config, param_values):
    """
    Run all 2D experiments, and return a combined DataFrame with results.
    """
    # Initiate list to store results
    results = []
    
    # Extract experiment information
    exp_name = exp['name']
    perbs = exp['perturbations']
    p1, p2 = exp['parameters']

    for perb in perbs:
        print(f"DEBUG: Starting {exp_name} sweep for perturbation: {perb}")

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

            # Metadata columns
            ss_df['p1_name'], ss_df['p1_value'] = p1, v1
            ss_df['p2_name'], ss_df['p2_value'] = p2, v2
            ss_df['perturbation'] = perb
            ss_df['exp_name'] = exp_name

            results.append(ss_df)
            
        print(f"DEBUG: Completed {exp_name} sweep for perturbation: {perb}")

    return pd.concat(results)

# --------------------------------------------------
# Full Combined Param Sweep
# --------------------------------------------------
def run_sweeps(base_model, result_dir, sweep_config, sim_config, target_exp=None):
    """
    Orchestrates the running of experiments. 
    If target_experiments are provided (list), only those experiment runs.
    """
    perb_config = sim_config['perturbations']
    all_exps = sweep_config['experiments']
    #exp_2d_cfg = sweep_config['2D_experiments']

    # Filter if we only want one specific run
    if target_exp:
        all_exps = [e for e in all_exps if e['name'] in target_exp]
        if not all_exps:
            print(f"ERROR: One of {target_exp} not found in config.")
            return
        
    sweep_results = []
    for exp in all_exps:
        # Determine resolution from config
        res_type = exp['resolution']
        param_values = build_ranges(sweep_config, resolution=res_type)

        print(f"\n>>> DEBUG: Initialising: {exp['name']} ({exp['type']})")

        try:
            # Choose the runner based on type
            if exp['type'] == "1D":
                df = run_1d_sweep_single(base_model, exp, perb_config, param_values)
            elif exp['type'] == "2D":
                df = run_2d_sweep_single(base_model, exp, perb_config, param_values)
            else:
                print(f"Unknown type {exp['type']} for {exp['name']}")
                continue

            # Post-processing (Delta & Phenotype)
            df['type'] = exp['type']
            df['delta'] = compute_delta(df, sim_config)
            df['phenotype'] = df['delta'].apply(lambda x: classify_phenotype(x, sim_config))
            sweep_results.append(df)
            
            # Save individual experiment result
            save_df_to_csv(df, result_dir, f"{exp['type']}_{exp['name']}")
            print(f">>> DEBUG: {exp['name']} successfully saved to {result_dir}")

        except Exception as e:
            print(f"ERROR: Failed experiment {exp['name']}: {str(e)}")

    full_sweep_df = pd.concat(sweep_results)
    save_df_to_csv(full_sweep_df, result_dir, f"param_sweep_full")
    print(f"DEBUG: Combined parameter sweep experiment data saved to {result_dir}")

    return full_sweep_df

