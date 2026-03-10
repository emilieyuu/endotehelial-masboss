import numpy as np 
def hill(tau, K, n):
    """
    S: Sensitivity input (sheae nmagnitude / force)
    K: Half activation thershold
    n: Hill coefficiet
    """
    return tau**n / (K**n + tau**n)

def get_mechanical_input(shear, normal_force, protein):
        """
        Select the relevant force component for each junction protein

        DSP:  tangential shear    → lateral tensile loading
        TJP1: normal force        → upstream pressure loading  
        JCAD: combined magnitude  → overall junction load
        """
        if protein == "DSP":
            return abs(shear)
        elif protein == "TJP1":
            return abs(normal_force)
        elif protein == "JCAD":
            # Overall junction tension — magnitude of total force vector
            return np.sqrt(shear**2 + normal_force**2) / np.sqrt(2)