# abm/analysis/analyse_hill.py
#
# Hill function parameter analysis for DSP, TJP1, JCAD recruitment. 

import numpy as np
import pandas as pd

from abm.signalling import get_protein_recruitment
from abm.rho_lookup_table import RhoLookupTable
from src.utils import save_df_to_csv

def compute_recruitment_curves(cfg, f_values):
    """
    Compute protein recruitment over a range of input values (tau).

    DSP and JCAD use f_normal (tensile shear).
    TJP1 uses f_total (uniform shear magniude).

    Return DataFrame: f_input, DSP, TJP1, JCAD
    """
    rows = []
    for f in f_values: 
        rows.append({
            'f_input': f, 
            'DSP': round(get_protein_recruitment(cfg, f, 'DSP'), 3),
            'TJP1': round(get_protein_recruitment(cfg, f, 'TJP1'), 3),
            'JCAD': round(get_protein_recruitment(cfg, f, 'JCAD'), 3),
        })

    return pd.DataFrame(rows)

def compute_spatial_recruitment(cfg, recr_dir, f_dir, f_magnitude, n_nodes=16):
    """
    Compute recruitment at each node of the cell. 

    Returns DataFrame: node_id, angle_deg, f_normal, f_total, 
        DSP, TJP1, JCAD, P_RhoA, P_RhoC, balance
    """

    lut = RhoLookupTable(cfg, recr_dir)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) + np.pi/2
    rows = [] 

    for i, angle in enumerate (angles):
        normal   = np.array([np.cos(angle), np.sin(angle)])
        f_normal = f_magnitude * abs(np.dot(normal, f_dir))

        dsp  = get_protein_recruitment(cfg, f_normal,    'DSP')
        tjp1 = get_protein_recruitment(cfg, f_magnitude, 'TJP1')
        jcad = get_protein_recruitment(cfg, f_normal,    'JCAD')
        rhoa, rhoc = lut.query(dsp, tjp1, jcad)

        rows.append({
            'node_id':   i,
            'angle_deg': round(np.degrees(angle) % 360, 1),
            'f_normal':  round(f_normal,    4),
            'f_total':   round(f_magnitude, 4),
            'DSP':       round(dsp,  4),
            'TJP1':      round(tjp1, 4),
            'JCAD':      round(jcad, 4),
            'P_RhoA':    round(rhoa, 4),
            'P_RhoC':    round(rhoc, 4),
            'balance':   round(rhoa - rhoc, 4),
        })

    return pd.DataFrame(rows)

def compute_ko_table(cfg, recr_dir, f_magnitude):
    """
    Compute P_RhoA and P_RhoC at pole nodes for each KO condition.
    """
    lut = RhoLookupTable(cfg, recr_dir)

    dsp_wt  = get_protein_recruitment(cfg, f_magnitude, 'DSP')
    tjp1_wt = get_protein_recruitment(cfg, f_magnitude, 'TJP1')
    jcad_wt = get_protein_recruitment(cfg, f_magnitude, 'JCAD')

    conditions = {
        'WT':            (dsp_wt,  tjp1_wt, jcad_wt),
        'DSP_KO':        (0.0,     tjp1_wt, jcad_wt),
        'TJP1_KO':       (dsp_wt,  0.0,     jcad_wt),
        'JCAD_KO':       (dsp_wt,  tjp1_wt, 0.0    ),
        'TJP1_JCAD_DKO': (dsp_wt,  0.0,     0.0    ),
        'DSP_JCAD_DKO':  (0.0,     tjp1_wt, 0.0    ),
    }

    rows = []
    for name, (d, t, j) in conditions.items():
        rhoa, rhoc = lut.query(d, t, j)
        rows.append({
            'condition':   name,
            'DSP':         round(d,    4),
            'TJP1':        round(t,    4),
            'JCAD':        round(j,    4),
            'P_RhoA':      round(rhoa, 4),
            'P_RhoC':      round(rhoc, 4),
            'rho_balance': round(rhoa - rhoc, 4),
        })

    return pd.DataFrame(rows)

def compute_balance_table(cfg, f_magnitude):
    """
    Recruitment and Rho balance at three representative boundary locations.

    pole:  f_normal = f_magnitude  (normal ∥ flow, full tensile load)
    mid:   f_normal = f_magnitude/2 (45° from flow axis)
    flank: f_normal = 0             (normal ⊥ flow, no tensile load)

    TJP1 always sees f_total = f_magnitude regardless of location.

    Returns DataFrame: location, f_normal, f_total,
                       DSP, TJP1, JCAD, P_RhoA, P_RhoC, balance
    """
    lut  = RhoLookupTable(cfg)
    rows = []

    for label, fn in [
        ('pole',  f_magnitude),
        ('mid',   f_magnitude / 2),
        ('flank', 0.0),
    ]:
        dsp  = get_protein_recruitment(cfg, fn,          'DSP')
        tjp1 = get_protein_recruitment(cfg, f_magnitude, 'TJP1')
        jcad = get_protein_recruitment(cfg, fn,          'JCAD')
        rhoa, rhoc = lut.query(dsp, tjp1, jcad)

        rows.append({
            'location': label,
            'f_normal': round(fn,          4),
            'f_total':  round(f_magnitude, 4),
            'DSP':      round(dsp,  4),
            'TJP1':     round(tjp1, 4),
            'JCAD':     round(jcad, 4),
            'P_RhoA':   round(rhoa, 4),
            'P_RhoC':   round(rhoc, 4),
            'balance':  round(rhoa - rhoc, 4),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Summary printer
# ------------------------------------------------------------------

def print_hill_summary(cfg, f_magnitude):
    """
    Print all four tables to stdout for quick inspection.
    """
    print(f"\n{'='*60}")
    print(f"Hill Parameter Summary")
    print(f"{'='*60}")
    print(f"f_magnitude = {f_magnitude}")
    print(f"K = f_magnitude / 5 = {f_magnitude/5:.2f}")
    for protein in ['DSP', 'TJP1', 'JCAD']:
        hp = cfg['hill_params'][protein]
        print(f"  {protein:<6}: K={hp['K']}  n={hp['n']}  "
              f"p_max={hp['p_max']}")

    print(f"\n--- Balance table (spatial gradient) ---")
    balance_df = compute_balance_table(cfg, f_magnitude)
    print(balance_df.to_string(index=False))

    print(f"\n--- KO Rho activity at poles ---")
    ko_df = compute_ko_table(cfg, f_magnitude)
    print(ko_df.to_string(index=False))

    print(f"\n--- Spatial recruitment per node ---")
    spatial_df = compute_spatial_recruitment(cfg, f_magnitude)
    print(spatial_df[['node_id', 'angle_deg', 'f_normal',
                       'DSP', 'TJP1', 'JCAD',
                       'P_RhoA', 'P_RhoC', 'balance']].to_string(index=False))

    return {
        'balance':  balance_df,
        'ko':       ko_df,
        'spatial':  spatial_df,
    }


def run_hill_analysis(cfg, output_dir=None):
    """
    Run full Hill analysis — compute all tables and optionally save CSVs.

    Returns dict of DataFrames for use in notebooks or further analysis.
    """
    f_magnitude = cfg['mechanics']['f_magnitude']
    f_values    = np.linspace(0, f_magnitude * 1.5, 300)

    results = {
        'curves':  compute_recruitment_curves(cfg, f_values),
        'spatial': compute_spatial_recruitment(cfg, f_magnitude),
        'ko':      compute_ko_table(cfg, f_magnitude),
        'balance': compute_balance_table(cfg, f_magnitude),
    }

    print_hill_summary(cfg, f_magnitude)

    if output_dir is not None:
        for name, df in results.items():
            save_df_to_csv(df, output_dir, f"hill_{name}", False)

    return results
