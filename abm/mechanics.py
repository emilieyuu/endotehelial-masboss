# abm/mechanics.py
#
# Pure mathematical funcions.
#
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

def bilinear_tension(l_current, l_rest, k_tensile, kc_ratio):
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
    
def activated_bilinear(l_current, l_rest, k, kc_ratio, a):
    """
    Bilinear elastic law for single spring component.

    l_current: Current physical length of the spring
    l_rest: Rest length – length when spring is stress free
    k: Stiffness in the stretched regime
    kc_ratio: Compressive stiffness as a fraction of tensile stiffness. 
              0.1 means 10x softer in compression
    a: Activation 
    """

    extension = l_current - l_rest # current_length - rest_length

    if extension > 0:
        # Stretching Regime: tension > 0, pulls nodes together
        k_eff = k
    else:
        # Compression Regime: tension < 0, pushes nodes aåart 
        k_eff = k * kc_ratio
    return k_eff * extension * a
    
def weighted_poisson(tension, nu, weight, width):
    """
    Compute lateral squeeze from axial tension using Poisson coupling
    """
    magnitude = -nu * tension * weight
    return magnitude * np.sign(width)
    