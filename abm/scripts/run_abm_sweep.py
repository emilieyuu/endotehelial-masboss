# abm/scripts/run_abm_sweep.py
import itertools
import copy
import numpy as np

from abm.scripts.run_abm_sim import run_abm_sim

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def build_param_combos(sweep_cfg):
    """
    Build all parameter combinations from abm sweep config. 

    Returns list of dicts, one per combination. 
    """
    param_ranges = sweep_cfg['param_ranges']
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    combinations = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    print(f">>> INFO: Built {len(combinations)} parameter combinations from {len(keys)} parameters: {keys}")
    return combinations

def get_combo_cfg(sim_cfg_base, param_combo):
    """
    Apply combination to a config copy. 

    param_combo: dict of {mechanic_param: value}

    Returns modified config copy
    """

    cfg = copy.deepcopy(sim_cfg_base)
    for param, value in param_combo.items(): 
        cfg['mechanics'][param] = value
    return cfg

def score_combination(ss_df, sweep_cfg):
    """
    Score one parameter combination against experiental targets. 

    Stage 1: Hard ordering constraints. 
    Stage 2: MAE AR ratios relatieve to WT 

    Returns (score, status) tuple. 
    """
    # Read targets and constraints from config
    exp_targets = sweep_cfg['exp_targets']
    order_consts = sweep_cfg['ordering_constraints']

    # Build ar lookup from ss_df
    ar = dict(zip(ss_df['perturbation'], ss_df['ar']))

    # Stage 1: Ordering Constraint
    for lower, upper in order_consts: 
        if lower not in ar or upper not in ar:
            return float('-inf'), f"missing_perturbation_{lower}_or_{upper}"
        if ar[lower] >= ar[upper]:
            return float('-inf'), f"ordering_failed_{lower}_geq_{upper}"
        
    # Stage 2: MAE of ratios relative to WT
    if 'WT' not in ar or 'WT' not in exp_targets:
        return float('-inf'), 'missing_WT'
    
    errors = []
    for perb in exp_targets: 
        if perb not in ar: 
            # Ensure only pertubations included in both are used
            continue

        model_ratio = ar[perb] / ar['WT']
        exp_ratio = exp_targets[perb] / exp_targets['WT']
        err = abs(model_ratio - exp_ratio)
        errors.append(err)

    mae = float(np.mean(errors))
    score = -mae # higher = better
    return score, 'ok'

def run_sweep_single(sim_cfg_base, lut, param_combo, sweep_cfg):
    """
    Run all perturbation for one parameter combination (2d)

    Returns one row dict of: parameters, AR per condition, score, status
    """

    # Apply this combination's parameters to config
    cfg_copy = get_combo_cfg(sim_cfg_base, param_combo)

    # Run all perturbations
    n_steps = sweep_cfg['sweep']['n_steps']
    _, ss_df, _, _ = run_abm_sim(cfg_copy, lut, n_steps=n_steps)

    # Score
    score, status = score_combination(ss_df, sweep_cfg)

    # Get WT AR for ratio computation
    wt_ar = ss_df.loc[ss_df['perturbation'] == 'WT', 'ar'].values
    wt_ar = float(wt_ar[0]) if len(wt_ar) > 0 else None

    # Build result row — parameters + AR per condition + score
    row = {**param_combo, 'score': score, 'status': status}
    for _, r in ss_df.iterrows():
        row[f"{r['perturbation']}_ar"] = r['ar']

    return row


        