# abm/helpers/mechanics.py
#
# Pure mathematical functions used by mechanical classes.

def bilinear_tension(l, l0, k, kc_ratio):
    """
    Bilinear spring: stiff in tension, soft in compression.
    Stretched (L > L0):  T = k × (L - L0), pulls nodes together
    Compressed (L < L0): T = k × kc × (L0 - L), pushes nodes apart (weak)
    
    Returns: signed axial force.
    """
    extension = l - l0 
    if extension > 0:
        return k * extension 
    else:
        return k * kc_ratio * extension 

def relax_toward(current, target, dt, tau): 
    """
    First order relaxation: (dx/dt = target - current) / tau

    Used for first-order relaxation of contractility towards target. 
    dt: model timestep
    tau: time constant, determines how fast to move toward target
    """
    return current + (dt / tau) * (target - current)

    
    