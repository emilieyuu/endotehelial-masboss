# abm/spring.py
#
# Represents a spring between two membrane nodes: the Junction. 
# Each junction acts as a self-contained mechanosensing unit.
# Mechanical input of junction proteins is spring tension. 
#
# Has: 
# 1. Spring Properties: rest_length, k_base, length, tension
# 2. Alignment to flow: alignment
# 3. Junction proteins: DSP, TJP1, JCAD
# 4. RhoA/RhoC
# 5. Differentiation between cortical/sf: k_cortical, L_cortical, L_sf

import numpy as np
###
# Helpers
###
def hill(tau, K, n):
    """
    S: Sensitivity input (sheae nmagnitude / force)
    K: Half activation thershold
    n: Hill coefficiet
    """
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

###
# Spring Class
### 
class Spring: 
    """
    Represents junction between two adjacent membrane nodes.
    """

    def __init__(self, node_1, node_2, rest_length, k_base=1.0):
        # Reference to MembraneNodes
        self.node_1 = node_1
        self.node_2 = node_2

        self.rest_length = rest_length # spring rest length
        self.k_base = k_base # base constant for bilinear law calculations

        # Spring Geometry
        self.length = rest_length
        self.tension = 0.0 # follows bilinear law
        self.unit_vec = np.zeros(2) # unit vector, represent spring orientation (node_1 + node_2)
        self.alignment = 0.0 # alignment of junction to flow

        # Junction Protein States
        self.DSP, self.TJP1, self.JCAD = 0, 0, 0
 
        # Rho Activation Proability
        self.P_RhoA, self.P_RhoC = 0.0, 0.0

        # Cortical Spring Mechanics ( RhoA -> actomyosin -> stiffer )
        self.k_cortical = k_base
        self.L_cortical = rest_length

        # # Stress Fibre Meachnics (RhoC -> stress fibres)
        # self.k_sf = k_base * 2.0
        # self.L_sf = rest_length

    
    def update_geometry(self, flow_direction):
        """
        Compute length, extension, tension and flow alignment from current node positions. 
        """
        # Compute length of junction
        diff = self.node_1.pos - self.node_2.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        # Update length
        self.length = length
        self.unit_vec = diff / self.length # calclate unit vector representation of junction
        self.alignment = abs(np.dot(self.unit_vec, flow_direction))

        # Update spring tension ( Katie spring equation)
        self.tension = calculate_bilinear_tension(self.length, self.rest_length, self.k_cortical)

    def calculate_forces(self):
        """
        Calculates the force vector and applies it to the two connected nodes.
        """
        # The scalar tension (magnitude) was calculated in update_geometry
        # We turn it into a vector. Tension > 0 pulls nodes together.
        force_vec = self.tension * self.unit_vec
        
        # Apply equal and opposite forces
        self.node_1.apply_force(-force_vec) # Pulling node 1 toward node 2
        self.node_2.apply_force(force_vec)
    

    def get_spring_mechanical_inputs(self):
        """
        Convert junction mechanical state to protein states. 

        DSP: tensile tension × alignment (lateral junctions, parallel to flow, pulled apart by shear)
        TJP1: tensile tension × (1 - alignment) (perpendicular junctions, upstream face, loaded by flow)
        JCAD: tensile tension magnitude (scaffold at any loaded junction, regardless of orientation)
        """
        tensile = max(self.tension, 0) 
        tau_dsp = tensile * self.alignment # senses shear drag on lateral junctions
        tau_tjp1 = tensile * (1 - self.alignment) # senses compression at upstream face
        tau_jcad = tensile # senses crowding at upstream face

        print(f">>> DEBUG: Computed mechnical input tau_dsp: {tau_dsp}, tau_tjp1: {tau_tjp1}, tau_jcad: {tau_jcad}")
        return tau_dsp, tau_tjp1, tau_jcad
    
    def update_signalling(self, cfg, perturbation='WT'):
        """
        Convert junction proteins to Rho activity.
        """

        tau_dsp, tau_tjp1, tau_jcad = self.get_spring_mechanical_inputs()

        # Calculate Hill-based probabilities for protein recruitment 
        p_dsp = get_recruitment(cfg, tau_dsp, 'DSP')
        p_tjp1 = get_recruitment(cfg, tau_tjp1, 'TJP1')
        p_jcad = get_recruitment(cfg, tau_jcad, 'JCAD')

        # Get RhoA / RhoC probabilities from Lookup table
        # Get p_RhoA and p_RhoC from lookup table
        return p_dsp, p_tjp1, p_jcad

    def update_stiffness(self):
        """
        Translate Rho activity to mechanical spring parameters

        RhoA → contractility → cortical spring stiffer + shorter rest length
        RhoC → stress fibre prestretch → L_sf < rest_length
        """

        # Cortical spring — RhoA driven
        self.k_cortical = self.k_base * (1.0 + 3.0 * self.P_RhoA)
        self.L_cortical = self.rest_length * (1.0 - 0.4 * self.P_RhoA)
        self.L_cortical = max(self.L_cortical, 0.3 * self.rest_length)


    def __repr__(self):
        return (f"Spring({self.node_1.id}→{self.node_2.id} | "
                f"L={self.length:.3f} L0={self.rest_length:.3f} | "
                f"T={self.tension:.3f} align={self.alignment:.3f} | "
                f"DSP={self.DSP} TJP1={self.TJP1} JCAD={self.JCAD} | "
                f"RhoA={self.P_RhoA:.3f} RhoC={self.P_RhoC:.3f})")

