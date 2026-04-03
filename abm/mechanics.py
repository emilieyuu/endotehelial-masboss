# abm/mechanics.py
#
# Pure mathematical functions used by mechanical classes.

def hookes_law(l, l0, k):
    """
    Hooke's Law: tension-only spring. 
    Returns positive tension when stretched (L > L0).
    Returns zero when compressed (cable goes slack).
    """
    extension = l - l0
    if extension <= 0: 
        return 0
    return extension * k

def bilinear_tension(l, l0, k, kc_ratio=0.1):
    """
    Bilinear spring: stiff in tension, soft in compression.
    Stretched (L > L0):  T = k × (L - L0), pulls nodes together
    Compressed (L < L0): T = k × kc × (L0 - L), pushes nodes apart (weak)
    
    Returns signed tension: positive = pull, negative = push.
    """

    extension = l - l0 
    if extension > 0:
        return k * extension 
    else:
        return k * kc_ratio * extension 
    
def hill(tau, K, n):
    """
    Hill activation function. 
    Maps mechanical stimulus to protein recruitment probability. 

    tau: stimulus magnitude
    K: half-activation threshold
    n: Hill coefficient (switch sharpness)
    """
    if tau <= 0: 
        return 0.0 # No recruitment under compression. 
    return tau**n / (K**n + tau**n)

    
    