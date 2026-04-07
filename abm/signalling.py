# abm/signalling.py
#
# Signalling functions: map mechanical inputs to protein recruitment.

from abm.mechanics import hill

def get_protein_recruitment(cfg, tau, protein):
    """
   Compute Hill-function recruitment for a junction protein.
    
    tau: mechanical input (f_normal for DSP/JCAD, f_total for TJP1)
    protein: 'DSP', 'TJP1', or 'JCAD'
    K = f_magnitude / K_divisor (scales with shear magnitude)
    Returns: recruitment level in [0, p_max]
    """
    params = cfg['hill_params'][protein]

    if params.get('knocked_out', False):
        return 0.0 # No recruitment if protein is knocked out.

    K = params.get('K', 5)
    n = params.get('n', 2)
    
    p_max = params.get('p_max', 0.67) 
    p_raw = hill(tau, K, n)
    return p_raw * p_max