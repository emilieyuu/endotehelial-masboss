#abm/signalling.py

from abm.mechanics import hill

def get_protein_recruitment(cfg, tau, protein, perturbation='WT'):
    """
    Look up Hill parameters for protein and return recruitment probability.
    """
    params = cfg['hill_params'][protein]

    if params.get('knocked_out', False):
        return 0.0 # No recruitment if protein is knocked out regardless of tau.

    f_mag = cfg['flow']['f_magnitude']
    K = f_mag / params['K_divisor']

    p_raw = hill(tau, K, params['n'])

    # Scale to physiological range from MaBoSS
    p_max = params.get('p_max', 1.0) 
    return p_raw * p_max