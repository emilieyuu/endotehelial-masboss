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
from abm.abm_helpers import hill, get_recruitment, calculate_bilinear_tension


###
# Spring Class
### 
class Spring: 
    """
    Represents junction between two adjacent membrane nodes.
    """

    def __init__(self, spring_id: int, node_1, node_2, 
                 rest_length: float, lut, cfg, k_base=1.0):
        
        self.lut = lut # rho lookup table
        self.cfg = cfg
        
        # Spring General Properties
        self.id = spring_id
        self.node_1 = node_1
        self.node_2 = node_2

        # Spring Initial Mechanical (rest)
        self.L_rest = rest_length # spring rest length
        self.k_base = k_base # stiffness of actin filaments

        # Cortical Spring Mechanics (current)
        self.L_cortical = rest_length
        self.k_cortical = k_base # active stiffness at any moment (changes with RhoA)

        # Spring Geometry – Recomputed at each step
        self.length = rest_length
        self.tension = 0.0 # follows bilinear law
        self.unit_vec = np.zeros(2) # unit vector, represent spring orientation (node_1 + node_2)
        self.alignment = 0.0 # alignment of junction to flow (1 = parallel, 0 = perpendicular)

        # Junction Protein States
        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
 
        # Rho Activation Proability
        self.P_RhoA, self.P_RhoC = 0.0, 0.0

    
    def update_geometry(self, flow_direction):
        """
        Recompute length, unit vector, flow alignment and tension. 

        unit_vec points from node 1 to node 2
        """
        # Compute length of junction
        diff = self.node_2.pos - self.node_1.pos # unit vector point from node 1 to 2
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        # Update length
        self.length = length
        self.unit_vec = diff / self.length # calclate unit vector representation of junction
        self.alignment = abs(np.dot(self.unit_vec, flow_direction))
        print(f""">>> DEBUG: Calcualted Geometry for Spring {self.id}
              diff: {diff.round(2)} | length: {length.round(2)} | unit vect: {self.unit_vec.round(2)} | alignment: {self.alignment:.2f}""")

        # Update spring tension 
        kc_ratio = self.cfg['mechanics']['kc_ratio']
        self.tension = calculate_bilinear_tension(
            self.length, self.L_cortical, 
            self.k_cortical, kc_ratio)

    
    def update_signalling(self, perturbation='WT'):
        """
        Convert mechanical state to Rho activity via Junction Proteins
        """
        tensile = max(self.tension, 0) 

        tau_dsp = tensile * self.alignment # senses shear drag on lateral junctions
        tau_tjp1 = tensile * (1 - self.alignment) # senses compression at upstream face
        tau_jcad = tensile # senses crowding at upstream face

        print(f">>> DEBUG: Computed mechnical input tau_dsp: {tau_dsp:.2f}, tau_tjp1: {tau_tjp1:.2f}, tau_jcad: {tau_jcad:.2f}")

        # Calculate Hill-based probabilities for protein recruitment 
        self.DSP = get_recruitment(self.cfg, tau_dsp, 'DSP')
        self.TJP1 = get_recruitment(self.cfg, tau_tjp1, 'TJP1')
        self.JCAD = get_recruitment(self.cfg, tau_jcad, 'JCAD')
        print(f""">>>INFO: Protein Recruitment for spring {self.id} is
               DSP: {self.DSP}, TJP1: {self.TJP1}, JCAD: {self.JCAD}""")

        # Get RhoA / RhoC probabilities from Lookup table
        self.P_RhoA, self.P_RhoC = self.lut.query(self.DSP, self.TJP1, self.JCAD)
        print(f""">>>INFO: Rho Activation for sprinG {self.id} is 
              RhoA: {self.P_RhoA:.2f}, RhoC: {self.P_RhoC:.2f}""")


    def update_stiffness(self, dt=0.1):
        """
        Translate Rho activity to mechanical spring parameters

        RhoA → contractility → cortical spring stiffer + shorter rest length
        RhoC → stress fibre prestretch → L_sf < rest_length
        """
        spring_cfg = self.cfg['mechanics']
        tau = spring_cfg['tau_remodel'] # constant time for remodelling

        # How much Rho is active ABOVE the resting competition baseline
        # Contract if Rho is above resting level
        rhoA_rest = self.lut.rhoA_rest
        rhoC_rest = self.lut.rhoC_rest
        delta_rhoA = max(self.P_RhoA - rhoA_rest, 0.0)
        delta_rhoC = max(self.P_RhoC - rhoC_rest, 0.0)

        # Calculate target (Where spring wants to be)
        # RhoA stiffens, RhoC + alignment thins the cell
        k_target = self.k_base * (1.0 + spring_cfg['rhoa_k_gain'] * delta_rhoA)
        print(f"k_target: {k_target}")

        # L_target shrings with RhoA and RhoC * alignment (lateral thinning)
        l_shrink_rhoa = spring_cfg['rhoa_l_shrink'] * delta_rhoA
        l_shrink_rhoc = spring_cfg['rhoc_l_shrink'] * delta_rhoC * self.alignment
        print(f"l_shrink_rhoa: {l_shrink_rhoa}, l_shrink_rhoc: {l_shrink_rhoc}")

        L_target = self.L_rest * (1.0 - l_shrink_rhoa - l_shrink_rhoc)
        L_target = max(L_target, self.L_rest * 0.4) # Physical limit
        print(f"L_target: {L_target}")

        # Relax toward targets (First-order lag)
        alpha = dt / tau
        self.k_cortical += alpha * (k_target - self.k_cortical)
        self.L_cortical += alpha * (L_target - self.L_cortical)
        print(f"k_cortical: {self.k_cortical}, L_cortical: {self.L_cortical}")

    def calculate_forces(self):
        """
        Calculates the force vector and applies it to the two connected nodes.
        """
        if self.length < 1e-10:
            return
        
        f = calculate_bilinear_tension(self.length, self.L_cortical, self.k_cortical,
                                    self.cfg['mechanics']['kc_ratio'])
        
        # The scalar tension (magnitude) was calculated in update_geometry
        # We turn it into a vector. Tension > 0 pulls nodes together.
        force_vec = f * self.unit_vec
        
        # Apply equal and opposite forces
        self.node_1.apply_force(force_vec) # Pushes node 1 away from node 2
        self.node_2.apply_force(-force_vec)


    def __repr__(self):
        return (f"Spring({self.node_1.id}→{self.node_2.id} | "
                f"L={self.length:.3f} L0={self.L_rest:.3f} | "
                f"T={self.tension:.3f} align={self.alignment:.3f} | "
                f"DSP={self.DSP} TJP1={self.TJP1} JCAD={self.JCAD} | "
                f"RhoA={self.P_RhoA:.3f} RhoC={self.P_RhoC:.3f})")

