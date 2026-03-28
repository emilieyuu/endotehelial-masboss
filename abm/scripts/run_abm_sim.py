# abm/scripts/run_abm_sim.py

import pandas as pd
import numpy as np
import copy 
from datetime import datetime

from abm.flow_field import FlowField
from abm.endothelial_cell import EndothelialCell
from src.utils import save_df_to_csv

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_perb_cfg(cfg_base, knockouts):
    """
    Apply knockout overrides to a config copy. 

    knockouts: dict of {protein: {key: value}}.
    """
    cfg = copy.deepcopy(cfg_base)

    for protein, override in knockouts.items():
        cfg['hill_params'][protein].update(override)

    return cfg

def get_spring_states(cell, exp_dict) -> list:
    """
    Return state of all springs as a list of dicts.
    Suitable for building a DataFrame across conditions and timesteps.

    condition: optional label (e.g. 'WT', 'DSP-KO')
    step:      optional step number
    time:  optional time in minutes
    """
    rows = []
    for s in cell.springs:
        state = s.get_state()
        rows.append({**exp_dict, **state})
    return pd.DataFrame(rows)

# --------------------------------------------------
# Single Perturbation Sim
# --------------------------------------------------

def run_abm_sim_single(cfg, lut, perturbation, n_steps):
    dt = cfg['sim'].get('dt', 0.1)
    n_nodes = cfg['sim'].get('n_nodes', 16)
    radius = cfg['sim'].get('radius', 12)
    f_magnitude = cfg['mechanics'].get('f_magnitude', 12)

    flow = FlowField(magnitude=f_magnitude)
    cell = EndothelialCell( 
        cell_id=0, centroid=np.array([0.0, 0.0]),
        lut=lut, cfg=cfg,
        n_nodes=n_nodes, radius=radius,
        flow_direction=flow.direction
    )

    cell_rows = []
    spring_dfs = []

    for step in range(n_steps):
        cell.step(flow, dt=dt)
        if step % 50 == 0 or step == n_steps - 1:
            time = round(step * dt, 1)

            exp_dict = {'step': step, 'time': time, 'perturbation': perturbation}
            state = cell.get_state()

            cell_rows.append({**exp_dict, **state})
            spring_dfs.append(get_spring_states(cell, exp_dict))

    cell_df = pd.DataFrame(cell_rows)
    spring_df = pd.concat(spring_dfs, ignore_index=True)

    return {
        'perb': perturbation,
        'cell_df': cell_df,
        'spring_df': spring_df,
        'cell_final': cell.get_state(),
        'spring_final': spring_dfs[-1],
        'cell': cell
    }

# --------------------------------------------------
# Full Perturbation Sim
# --------------------------------------------------
def run_abm_sim(cfg, lut, n_steps=None, result_dir=None):
    """
    Run all perturbations

    Returns:
        timeseries_df:  full cell history across all perturbations
        ss_df:          final state only per perturbation
        spring_ts_df:   full spring history across all perturbations
        spring_ss_df:   final spring state only per perturbation
    """
    perbs = cfg['perturbations']
    n_steps = n_steps or cfg['sim']['n_steps']

    cell_histories   = []
    spring_histories = []
    ss_rows          = []
    spring_ss_rows   = []

    for name, perb, in perbs.items(): 
        #print(f">>> INFO: Running abm simulation perturbation: {name} ({n_steps} steps).")

        perb_cfg = get_perb_cfg(cfg_base=cfg, knockouts=perbs[name])
        result = run_abm_sim_single(perb_cfg, lut, name, n_steps)

        cell_histories.append(result['cell_df'])
        spring_histories.append(result['spring_df'])
        ss_rows.append({'perturbation': name, **result['cell_final']})
        spring_ss_rows.append(result['spring_final'])


   # print(f">>> INFO: All perturbations completed successfully.")

    timeseries_df  = pd.concat(cell_histories,   ignore_index=True)
    spring_ts_df   = pd.concat(spring_histories, ignore_index=True)
    ss_df          = pd.DataFrame(ss_rows)
    spring_ss_df   = pd.concat(spring_ss_rows,   ignore_index=True)

    if result_dir is not None:
        save_df_to_csv(timeseries_df, result_dir, "abm_timeseries", ts=True)
        save_df_to_csv(ss_df, result_dir, "abm_ss", ts=True)
        save_df_to_csv(spring_ts_df, result_dir, "abm_spring_timeseries", ts=True)
        save_df_to_csv(spring_ss_df, result_dir, "abm_spring_ss", ts=True)
        print(f">>> INFO: Results saved to {result_dir}")

    return timeseries_df, ss_df, spring_ts_df, spring_ss_df