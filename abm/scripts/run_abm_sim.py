# abm/scripts/run_abm_sim.py
#
# Simulation for the EC elongation ABM.
#
# run_abm_sim_single — one perturbation, returns DataFrames
# run_abm_sim — all perturbations, aggregates and optionally saves

import copy 
import time
import pandas as pd
import matplotlib.pyplot as plt

from abm.flow_field import FlowField
from abm.cell_agent import CellAgent
from abm.analysis.cell_measurement import measure_cell, measure_springs, measure_nodes
from src.utils import save_df_to_csv, require

# ------------------------------------------------------------------
# Config Helpers
# ------------------------------------------------------------------
def perturbation_cfg(cfg_base, knockouts):
    """
    Return a deep copy of cfg with knockout overrides applied.

    knockouts: dict of {protein_name: {key: value}}, e.g.
        {'DSP': {'knocked_out': True}}
    """
    cfg = copy.deepcopy(cfg_base)
    hill = require(cfg, 'hill_params')
    for protein, override in knockouts.items():
        require(hill, protein).update(override)
    return cfg

def override_cfg(cfg_base, overrides):
    """
    Return a deep copy of cfg with nested overrides applied.

    overrides: nested dict. Leaf values replace the corresponding
    values in cfg_base.
    """
    cfg = copy.deepcopy(cfg_base)
    
    def _apply(target, src):
        for key, val in src.items():
            if isinstance(val, dict):
                _apply(require(target, key), val)
            else:
                require(target, key)   # verify the key exists
                target[key] = val
    _apply(cfg, overrides)
    return cfg

# ------------------------------------------------------------------
# Per-step Recording
# ------------------------------------------------------------------
def _record_step(cell, exp_dict, log_detail):
    """
    Build the three measurement records for one logged timestep.

    exp_dict: experimental information columns
    log_detail: whether or not to return spring/node rows

    Returns: (cell_row, spring_rows, node_rows). 
    """
    cell_row = {**exp_dict, **measure_cell(cell)}

    if not log_detail:
        return cell_row, [], []

    spring_rows = [{**exp_dict, **r} for r in measure_springs(cell)]
    node_rows   = [{**exp_dict, **r} for r in measure_nodes(cell)]
    return cell_row, spring_rows, node_rows

# ------------------------------------------------------------------
# Single Perturbation Sim
# ------------------------------------------------------------------
def run_abm_sim_single(cfg, lut, perturbation = 'WT', plot=False):
    """
    Run a single perturbation through the full timestep pipeline.

    Records measure_cell every step and measure_springs/measure_nodes 
    every detail_interval steps.

    Returns a result dict containing three DataFrames and the final
    cell object.
    """
    # Extract experiment parameters from config
    sim_cfg = require(cfg, 'simulation')
    dt = require(sim_cfg, 'dt')
    n_steps = require(sim_cfg, 'n_steps')
    detail_interval = require(sim_cfg, 'detail_log_interval')

    # Initiate flowfield and cell
    flow = FlowField(cfg)
    cell = CellAgent( 
        cell_id=0, flow_axis=flow.direction,
        lut=lut, cfg=cfg
    )

    # Initiate lists to store results
    cell_rows, spring_rows, node_rows = [], [], []

    # Time loop
    print(f">>> INFO: Running perturbation: {perturbation} for {n_steps} steps.")
    t_start = time.perf_counter()
    for step in range(n_steps):
        cell.step(flow, dt=dt)

        t = round(step * dt, 2)
        exp_dict = {'step': step, 'time': t, 'perturbation': perturbation}

        # Cell-level: every step. Detail dumps: every detail_interval
        # Final step steady-state always captured.
        log_detail = (step % detail_interval == 0) or (step == n_steps - 1)
        c_row, s_rows, n_rows = _record_step(cell, exp_dict, log_detail)
        cell_rows.append(c_row)
        spring_rows.extend(s_rows)
        node_rows.extend(n_rows)

    # Steady-state snapshots — captured from the final cell geometry
    ss_exp_dict = {'step': n_steps - 1, 'time': round((n_steps - 1) * dt, 2),
                'perturbation': perturbation}
    cell_ss   = {**ss_exp_dict, **measure_cell(cell)}
    spring_ss = [{**ss_exp_dict, **r} for r in measure_springs(cell)]
    node_ss   = [{**ss_exp_dict, **r} for r in measure_nodes(cell)]

    # Debug
    if plot:
        plot_cell(cell, title=f"{perturbation} (final)")

    runtime = time.perf_counter() - t_start
    print(f" {perturbation:} ar={cell_ss['ar']:.3f} | "
      f"rho_balance={cell_ss['rho_balance']:.3f} | "
      f"rhoa={cell_ss['rhoa_mean']:.3f} | rhoc={cell_ss['rhoc_mean']:.3f} | "
      f"t={runtime:.1f}s")

    return {
        'perturbation': perturbation,
        'cell_df':      pd.DataFrame(cell_rows),
        'spring_df':    pd.DataFrame(spring_rows),
        'node_df':      pd.DataFrame(node_rows),
        'cell_ss':      cell_ss,      
        'spring_ss':    spring_ss,   
        'node_ss':      node_ss,     
        'cell':         cell,
    }

# ------------------------------------------------------------------
# Full Perturbation Sim
# ------------------------------------------------------------------
def run_abm_sim(cfg, lut, result_dir=None, plot=False):
    """
    Run every perturbation defined in cfg['perturbations'].

    Saves four CSVs if result_dir is provided:
        abm_cell_timeseries - per-step cell-level snapshot
        abm_spring_timeseries - sparse per-spring snapshots
        abm_node_timeseries - sparse per-node snapshots
        abm_steady_state - final cell-level row per perturbation

    Returns the four DataFrames.
    """
    perturbations = require(cfg, 'perturbations')

    cell_dfs, spring_dfs, node_dfs = [], [], []
    ss_rows = []

    for name, knockouts in perturbations.items():
        perb_cfg = perturbation_cfg(cfg_base=cfg, knockouts=knockouts)
        result = run_abm_sim_single(perb_cfg, lut, perturbation=name, plot=plot)

        cell_dfs.append(result['cell_df'])
        spring_dfs.append(result['spring_df'])
        node_dfs.append(result['node_df'])

        # Steady-state row = the last cell row of this perturbation.
        ss_rows.append(result['cell_df'].iloc[-1].to_dict())
        
    cell_ts_df   = pd.concat(cell_dfs,   ignore_index=True)
    spring_ts_df = pd.concat(spring_dfs, ignore_index=True)
    node_ts_df   = pd.concat(node_dfs,   ignore_index=True)
    ss_df        = pd.DataFrame(ss_rows)

    if result_dir is not None:
        save_df_to_csv(cell_ts_df,   result_dir, "abm_cell_timeseries",   ts=True)
        save_df_to_csv(spring_ts_df, result_dir, "abm_spring_timeseries", ts=True)
        save_df_to_csv(node_ts_df,   result_dir, "abm_node_timeseries",   ts=True)
        save_df_to_csv(ss_df,        result_dir, "abm_steady_state",      ts=True)
        print(f">>> INFO: Results saved to {result_dir}")

    return cell_ts_df, spring_ts_df, node_ts_df, ss_df

# ------------------------------------------------------------------

def plot_cell(cell, title=''):
    """
    Minimal cell plot — nodes and springs on axes only.
    Nodes are coloured by current polar/lateral classification,
    recomputed from the cell's geometry at plot time.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Build polar set once for O(1) membership lookup during node draw.
    polar_set = set(id(n) for n in cell.polar_nodes)

    # Springs — uniform colour (differentiation by polar/lateral at
    # plot time is possible but rarely useful; keep it simple).
    for s in cell.springs:
        p1, p2 = s.node_1.pos, s.node_2.pos
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                'b-', linewidth=1.2, alpha=0.6)

    # Nodes coloured by current polar/lateral classification.
    for n in cell.nodes:
        colour = 'red' if id(n) in polar_set else 'steelblue'
        ax.scatter(*n.pos, c=colour, s=50, zorder=4)
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