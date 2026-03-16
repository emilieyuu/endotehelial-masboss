import numpy as np

def hill(tau, K, n):
    """
    Hill activation function. Maps mechanical stimulus (tau) to 
    protein recruitment probability. 

    tau: Mechnical stimulus – a tension magnitude
    K: Half Activation Threshold – the tau at which recruitment = 0.5.
    n: Hill coefficient – controls spring sharpness.
    """
    if tau <= 0: 
        return 0.0 # No recruitment under compression. 
    
    return tau**n / (K**n + tau**n)

def get_recruitment(cfg, tau, protein, perturbation='WT'):
    """
    Look up Hill parameters for protein and return recruitment probability.
    """
    params = cfg['hill_params'][protein]

    if params.get('knocked_out', False):
        print(">>> DEBUG: {protein} is knocked out, recruitment is 0")
        return 0.0 # No recruitment if protein is knocked out regardless of tau.

    p_raw = hill(tau, params['K'], params['n'])
    # Scale to physiological range from MaBoSS
    p_max = params.get('p_max', 1.0)   # set per-protein in config
    return p_raw * p_max

def calculate_bilinear_tension(l_current, l_rest, k_tensile, kc_ratio):
    """
    Bilinear elastic law for single spring component.

    l_current: Current physical length of the spring
    l_rest: Rest length – length when spring is stress free
    k_tensile: Stiffness in the stretched regime
    kc_ratio: Compressive stiffness as a fraction of tensile stiffness. 
              0.1 means 10x softer in compression
    """

    extension = l_current - l_rest # current_length - rest_length

    if extension > 0:
        # Stretching Regime: tension > 0, pulls nodes together
        return k_tensile * extension 
    else:
        # Compression Regime: tension < 0, pushes nodes aåart 
        k_comp = k_tensile * kc_ratio
        return k_comp * extension 