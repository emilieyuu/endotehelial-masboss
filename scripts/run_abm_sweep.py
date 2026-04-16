# abm/scripts/run_abm_sweep.py
import itertools
import copy
import numpy as np
import pandas as pd

from abm.scripts.run_abm_sim import run_abm_sim
from abm.analysis.phenotype import classify_abm_phenotype
from src.utils import save_df_to_csv

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
    rows = []
    for _, r in ss_df.iterrows():
        ar_ratio  = round(r['ar'] / wt_ar, 4) if wt_ar else None
        phenotype = classify_abm_phenotype(
            r['rho_balance'], r['mean_k_active'],
            r['mean_lsf_ratio'], ar_ratio, sim_cfg_base
        )
        rows.append({
            **param_combo,
            'perturbation':   r['perturbation'],
            'ar':             r['ar'],
            'ar_ratio':       ar_ratio,
            'phenotype':      phenotype,
            'rho_balance':    r['rho_balance'],
            'mean_k_active':  r['mean_k_active'],
            'mean_lsf_ratio': r['mean_lsf_ratio'],
            'mean_t_sf':      r['mean_t_sf'],
            'combo_score':    round(score, 6) if score != float('-inf') else score,
            'combo_status':   status,
        })

    return pd.DataFrame(rows)

def run_param_sweep(cfg_base, lut, sweep_cfg, result_dir=None):
    """
    Run full parameter sweep over all combinations and perturbations.
    Mirrors run_sweeps() from MaBoSS sweep.

    Returns sweep_df in long format — one row per perturbation per combination.
    Sorted by combo_score descending within each perturbation.
    """
    combinations = build_param_combos(sweep_cfg)
    total        = len(combinations)
    all_dfs      = []

    print(f">>> INFO: Starting ABM parameter sweep")
    print(f">>> INFO: {total} combinations × "
          f"{len(cfg_base['perturbations'])} perturbations = "
          f"{total * len(cfg_base['perturbations'])} runs")
    print(f">>> INFO: {sweep_cfg['sweep']['n_steps']} steps per run\n")

    for i, combo in enumerate(combinations):
        print(f">>> INFO: [{i+1}/{total}] {combo}")

        try:
            combo_df = run_sweep_single(cfg_base, lut, combo, sweep_cfg)
            all_dfs.append(combo_df)

            # Quick progress summary — WT row only
            wt = combo_df[combo_df['perturbation'] == 'WT']
            if not wt.empty:
                print(f"    WT: ar={wt['ar'].values[0]:.3f}  "
                      f"score={wt['combo_score'].values[0]:.4f}  "
                      f"status={wt['combo_status'].values[0]}")

        except Exception as e:
            print(f">>> ERROR: Combination {i+1} failed: {e}")
            continue

    if not all_dfs:
        print(">>> ERROR: No combinations completed successfully.")
        return pd.DataFrame()

    sweep_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by combo_score descending
    sweep_df = sweep_df.sort_values('combo_score', ascending=False
                                    ).reset_index(drop=True)

    # Save
    if result_dir is not None:
        save_df_to_csv(sweep_df, result_dir,
                       'abm_param_sweep', timestamp=True)

    # Print top 5 by WT AR
    params   = list(sweep_cfg['param_ranges'].keys())
    top5     = (sweep_df[sweep_df['perturbation'] == 'WT']
                .sort_values('combo_score', ascending=False)
                .head(5))

    print(f"\n>>> INFO: Sweep complete — top 5 combinations:")
    print(f"{'#':>3} " +
          " ".join(f"{p:>8}" for p in params) +
          f" {'WT_ar':>7} {'score':>8}")
    print("-" * (3 + 9 * len(params) + 17))

    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        param_str = " ".join(f"{row[p]:>8.1f}" for p in params)
        print(f"{rank:>3} {param_str} {row['ar']:>7.3f} "
              f"{row['combo_score']:>8.4f}")

    return sweep_df


        