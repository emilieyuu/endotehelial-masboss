# abm/scripts/run_abm_sim.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def plot_cell(cell, title=''):
    """
    Minimal cell plot — nodes and springs on axes only.
    Call standalone or via run_sim(plot=True).
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Springs
    for s in cell.springs:
        p1, p2 = s.node_1.pos, s.node_2.pos
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                'b-', linewidth=1.2, alpha=0.6)

    # Nodes coloured by role
    colors = {'upstream': 'red', 'downstream': 'orange', 'lateral': 'steelblue'}
    for n in cell.nodes:
        ax.scatter(*n.pos, c=colors[n.role], s=50, zorder=4)
        ax.text(n.pos[0] + 0.02, n.pos[1] + 0.02,
                str(n.id), fontsize=8, color='black', zorder=6)

    ax.scatter(*cell.centroid, marker='x', c='gray', s=60, zorder=5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.4, linestyle='--')
    ax.set_title(title or 'cell shape')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Single Perturbation Sim
# --------------------------------------------------

def run_abm_sim_single(cfg, lut, n_steps, perturbation = 'WT', plot=False):
    """
    Run full step-pipeline for a single cell under a single perturbation. 
    """
    # Extract simulation parameters from config
    dt = cfg['integration'].get('dt', 0.1)
    n_nodes = cfg['cell_geometry'].get('n_nodes', 16)
    radius = cfg['cell_geometry'].get('radius', 12)

    f_magnitude = cfg['flow'].get('f_magnitude', 10)
    f_direction = np.array(cfg['flow'].get('f_direction', [1.0, 0.0]))
    drag_frac = cfg['flow'].get('f_drag_fraction', 0.1)

    # Initiate flowfield and cell
    flow = FlowField(magnitude=f_magnitude, drag_frac=drag_frac, direction=f_direction)
    cell = EndothelialCell( 
        cell_id=0, centroid=np.array([0.0, 0.0]),
        lut=lut, cfg=cfg,
        n_nodes=n_nodes, radius=radius,
        flow_direction=flow.direction
    )


    # Initiate lists to store results
    cell_rows = []
    diagnostic_rows = []
    spring_dfs = []

    # Time loop
    for step in range(n_steps):
        cell.step(flow, dt=dt)
        if step % 50 == 0 or step == n_steps - 1:
            time = round(step * dt, 1)

            exp_dict = {'step': step, 'time': time, 'perturbation': perturbation}
            state = cell.get_state()
            diags = cell.get_diagnostics()

            cell_rows.append({**exp_dict, **state})
            diagnostic_rows.append({**exp_dict, **diags})
            spring_dfs.append(get_spring_states(cell, exp_dict))

    cell_df = pd.DataFrame(cell_rows)
    diags_df = pd.DataFrame(diagnostic_rows)
    spring_df = pd.concat(spring_dfs, ignore_index= True)
    sf_state = cell.stress_fibre.get_state()
    print(sf_state)
    # Optionally plot cell at initiation
    if plot:
        plot_cell(cell)

    return {
        'perb': perturbation,
        'cell_df': cell_df,
        'spring_df': spring_df,
        'diagnostics': diags_df,
        'cell_final': cell.get_state(),
        'diagnostics_final': cell.get_diagnostics(),
        'springs_final': get_spring_states(cell, {}),
       'sf_final': cell.stress_fibre.get_state(),
        'cell': cell
    }

# --------------------------------------------------
# Full Perturbation Sim
# --------------------------------------------------
def run_abm_sim(cfg, lut, n_steps=None, result_dir=None, plot=False):
    """
    Run all perturbations

    Returns:
        timeseries_df:  full cell history across all perturbations
        ss_df:          final state only per perturbation
    """
    perbs = cfg['perturbations']
    n_steps = n_steps or cfg['sim']['n_steps']

    cell_histories = []
    diag_histories = []
    ss_rows = []
    diag_rows = []

    for name, perb, in perbs.items(): 
        print(f">>> INFO: Running abm simulation perturbation: {name} ({n_steps} steps).")

        perb_cfg = get_perb_cfg(cfg_base=cfg, knockouts=perbs[name])
        result = run_abm_sim_single(perb_cfg, lut, n_steps, name, plot)

        cell_histories.append(result['cell_df'])
        diag_histories.append(result['diagnostics'])

        ss_state = result['cell_final']

        ss_rows.append({'perturbation': name, **result['cell_final']})
        diag_rows.append({'perturbation': name, **result['diagnostics_final']})

        print(f"{name:<18} {result['cell_final']['ar']:>6.3f} {result['cell_final']['orientation']:>8.1f}° "
                f"{result['cell_final']['a_sf']:>6.3f} ")

   # print(f">>> INFO: All perturbations completed successfully.")

    timeseries_df  = pd.concat(cell_histories,   ignore_index=True)
    diagnostics_ts_df = pd.concat(diag_histories, ignore_index=True)
    ss_df = pd.DataFrame(ss_rows)
    diagnostics_ss_df = pd.DataFrame(diag_rows)

    if result_dir is not None:
        save_df_to_csv(timeseries_df, result_dir, "abm_timeseries", ts=True)
        save_df_to_csv(ss_df, result_dir, "abm_ss", ts=True)
        save_df_to_csv(diagnostics_ts_df, result_dir, "abm_diagnostics_timeseries", ts=True)
        save_df_to_csv(diagnostics_ss_df, result_dir, "abm_dianostics_ss", ts=True)
        print(f">>> INFO: Results saved to {result_dir}")

    return timeseries_df, ss_df, diagnostics_ts_df, diagnostics_ss_df