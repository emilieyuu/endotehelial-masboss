import numpy as np

def hill(tau, K, n):
    """
    S: Sensitivity input (sheae nmagnitude / force)
    K: Half activation thershold
    n: Hill coefficiet
    """
    if tau <= 0: 
        return 0.0
    
    return tau**n / (K**n + tau**n)

def get_recruitment(cfg, tau, protein):
    """
    Tension-Based Hill recruitment for protein. 
    """
    hill_params = cfg['hill_params'][protein]

    K = hill_params['K']
    n = hill_params['n']

    return hill(tau, K ,n)

def calculate_bilinear_tension(l, l0, kt, kc_ratio=0.1):
    """
    l: current_length
    l0: rest_length
    kt: tensile stiffness (from RhoA/RhoC mapping)
    kc_ratio: 0.1 (compressive stiffness is 10% of tensile)
    """

    extension = l - l0 # current_length - rest_length
    if extension > 0:
        # Tension regime (Stretching) - gives positice tension
        return kt * extension 
    else:
        # Compression regime (Squishing) - gives smaller restoring force (10% stiffness)
        kc = kt * kc_ratio
        return kc * extension # This will be a negative value (pushing force)