# abm/helpers/signalling.py
#
# Signalling functions: map mechanical inputs to protein recruitment.
from src.utils import require

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
    
    tau: mechanical stimulus magnitude (tensile for DSP, shear for TJP1/JCAD)
    protein: 'DSP', 'TJP1', or 'JCAD'
    Returns: recruitment level in [0, p_max], or 0.0 if the protein is knocked out
    """
    params = require(cfg, 'hill_params', protein)

    if params.get('knocked_out', False):
        return 0.0 # KO: clamp recruitment to zero

    K = require(params, 'K')
    n = require(params, 'n')
    p_max = require(params, 'p_max')

    return hill(tau, K, n) * p_max