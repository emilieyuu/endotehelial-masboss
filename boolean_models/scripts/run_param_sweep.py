#boolean_models/scripts/run_param_sweep.py
import maboss
import pandas as pd
from pathlib import Path
import yaml
import numpy as np

from boolean_models.analysis import compute_delta

def run_param_sweep_single(model, perturbation, config, sweep_cfg):
    base_model = model

    result = []
    
    for param_cfg in sweep_cfg:
        name = param_cfg['parameter']
        print(f"DEBUG: Running sweep for parameter: {name}")

        values = np.arange(param_cfg['range'][0], param_cfg['range'][1], param_cfg['steps'])

        for val in values: 
            m = base_model.copy()
            m.update_parameters(**{name: val})
            #print(m.param[name])

            res = m.run()
            ss_df = res.get_last_nodes_probtraj()

            ss_df['delta'] = compute_delta(ss_df, config)
            ss_df['param_value'] = val
            ss_df['param_name'] = name

            result.append(ss_df)

    combined_df = pd.concat(result, ignore_index=True)

    return combined_df
    save_df_to_csv(combined_df, PARAM_DIR, f"{perturbation}_param_sweep.csv")
