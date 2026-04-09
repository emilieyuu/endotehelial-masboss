# abm/signalling.py
#
# Signalling functions: map mechanical inputs to protein recruitment.

def hill(tau, K, n):
    """
    Hill activation function. 
    Maps mechanical stimulus to protein recruitment probability. 

    tau: stimulus magnitude
    K: half-activation threshold
    n: Hill coefficient (switch sharpness)
    """
    # No recruitment under compression. 
    if tau <= 0: 
        return 0.0 
    
    return tau**n / (K**n + tau**n)

def get_protein_recruitment(cfg, tau, protein):
    """
    Compute Hill-function recruitment for a junction protein.
    
    tau: mechanical input (f_normal for DSP/JCAD, f_total for TJP1)
    protein: 'DSP', 'TJP1', or 'JCAD'
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