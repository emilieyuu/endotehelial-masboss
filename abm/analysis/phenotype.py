# abm/analysis/phenotype.py
#
# Phenotype classification and AR ratio computation for ABM results. 

import pandas as pd

def classify_abm_phenotype(rho_balance, mean_k_active, mean_lsf_ratio, 
                           ar_ratio, cfg):
    """
    Classiy ABM sell phenotype from steady-state mechanical features. 

    
    """
    thresholds = cfg['analysis']['thresholds']
    labels     = cfg['analysis']['phenotypes']

    t_balance   = thresholds.get('balance',   0.25)
    t_k         = thresholds.get('k_active',  1.3)
    t_lsf       = thresholds.get('lsf',       0.95)
    t_ar_hyper  = thresholds.get('ar_hyper',  1.2)
    t_ar_failed = thresholds.get('ar_failed', 0.85)

    # Primary — mechanical outputs
    fibres_assembled = mean_lsf_ratio < t_lsf
    cortex_stiffened = mean_k_active  > t_k
    no_fibres        = mean_lsf_ratio >= 0.999  # lsf never dropped — gate confirmed

    ar_above_wt = ar_ratio is not None and ar_ratio > t_ar_hyper
    ar_below_wt = ar_ratio is not None and ar_ratio < t_ar_failed

    # Secondary — Rho balance confirms mechanism
    rhoc_dominant = rho_balance >  t_balance
    rhoa_dominant = rho_balance < -t_balance

    # Classification
    if fibres_assembled and ar_above_wt:
        return labels['hyper']

    elif cortex_stiffened and no_fibres and ar_below_wt:
        return labels['failed']

    # Borderline — one mechanical signal present, use balance to resolve
    elif fibres_assembled and rhoc_dominant:
        return labels['hyper']   # fibres present, RhoC confirms

    elif cortex_stiffened and rhoa_dominant:
        return labels['failed']  # stiffening present, RhoA confirms

    else:
        return labels['normal']
    
def compute_ar_ratios(df, ref='WT'):
    """
    Add ar_ratio column to DataFrame – AR relative to reference condition. 
    Computed per parameter combination (groupe by sweep params).
    """

    non_param_cols = {'perturbation', 'ar', 'ar_ratio', 'phenotype',
                      'rho_balance', 'mean_k_active', 'mean_lsf_ratio',
                      'mean_t_sf', 'combo_score', 'combo_status'}
    param_cols = [c for c in df.columns if c not in non_param_cols]

    if not param_cols:
        # Single run — no grouping needed
        ref_ar = df.loc[df['perturbation'] == ref, 'ar']
        if len(ref_ar) == 0:
            df['ar_ratio'] = None
            return df
        ref_val = float(ref_ar.iloc[0])
        df['ar_ratio'] = df['ar'].apply(
            lambda x: round(x / ref_val, 4) if ref_val > 0 else None
        )
        return df

    # Sweep run — compute ratio within each parameter combination
    ref_ar_map = (
        df[df['perturbation'] == ref]
        .set_index(param_cols)['ar']
        .to_dict()
    )

    def get_ratio(row):
        key = tuple(row[p] for p in param_cols)
        ref = ref_ar_map.get(key)
        return round(row['ar'] / ref, 4) if ref and ref > 0 else None

    df = df.copy()
    df['ar_ratio'] = df.apply(get_ratio, axis=1)
    return df

