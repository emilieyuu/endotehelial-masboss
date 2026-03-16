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
# 5. Differentiation between cortical/sf: k_active, L_rest_active, L_sf

import numpy as np
from abm.abm_helpers import hill, get_recruitment, calculate_bilinear_tension


###
# Spring Class
### 
class Spring: 
    """
    Represents junction between two adjacent membrane nodes.
    Tracks Length, Stiffness, Proteins and Rho. 
    """
    def __init__(self, spring_id: int, node_1, node_2, 
                 rest_length: float, lut, cfg, k_base=1.0):
        
        self.lut = lut # rho lookup table
        self.cfg = cfg
        
        # Spring General Properties
        self.id = spring_id
        self.node_1 = node_1
        self.node_2 = node_2

        # REFERNCE: Length and stiffness upon initialisation
        self.L_init = rest_length 
        self.k_base = k_base 

        # DYNAMIC: Remodeled length and stiffness
        self.L_rest_active = rest_length 
        self.k_active = k_base 

        # INSTANT GEOMETRY: Physical state this frame
        self.L_current = rest_length
        self.tension = 0.0 # follows bilinear law
        self.unit_vec = np.zeros(2) # unit vector, represent spring orientation (node_1 + node_2)
        self.alignment = 0.0 # alignment of junction to flow (1 = parallel, 0 = perpendicular)

        # Junction Protein and Rho States 
        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
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
        self.L_current = length
        self.unit_vec = diff / self.L_current # calclate unit vector representation of junction
        self.alignment = abs(np.dot(self.unit_vec, flow_direction))

        # Update spring tension 
        kc_ratio = self.cfg['mechanics']['kc_ratio']
        self.tension = calculate_bilinear_tension(
            self.L_current, self.L_rest_active, 
            self.k_active, kc_ratio)

    
    def update_signalling(self, perturbation='WT'):
        """
        Convert mechanical state to Rho activity via Junction Proteins
        """
        tensile = max(self.tension, 0) 

        # CHANGE: DSP/JCAD response field
        tau_dsp = tensile # responds to raw junction load
        tau_tjp1 = abs(tensile) #tensile * (1 - self.alignment) # senses compression at upstream face
        tau_jcad = tensile * self.alignment # shear-flow amplifier, most active at lateral junctions

        # Calculate Hill-based probabilities for protein recruitment 
        self.DSP = get_recruitment(self.cfg, tau_dsp, 'DSP')
        self.TJP1 = get_recruitment(self.cfg, tau_tjp1, 'TJP1')
        self.JCAD = get_recruitment(self.cfg, tau_jcad, 'JCAD')

        # Get RhoA / RhoC probabilities from Lookup table
        self.P_RhoA, self.P_RhoC = self.lut.query(self.DSP, self.TJP1, self.JCAD)


    def update_stiffness(self, dt=0.1):
        """
        Translate Rho activity to mechanical spring parameters

        RhoA → contractility → cortical spring stiffer + shorter rest length
        RhoC → stress fibre prestretch → L_sf < rest_length
        """
        spring_cfg = self.cfg['mechanics']
        tau = spring_cfg['tau_remodel'] # constant time for remodelling

        # Relative Rho Activation above baseline
        delta_rhoA = max(self.P_RhoA - self.lut.rhoA_rest, 0.0)
        delta_rhoC = max(self.P_RhoC - self.lut.rhoC_rest, 0.0)

        # Calculate target stiffness (where spring wants to be)
        k_target = self.k_base * (1.0 + spring_cfg['rhoa_k_gain'] * delta_rhoA)

        # Calculate target length relative to initial length
        l_shrink_rhoa = spring_cfg['rhoa_l_shrink'] * delta_rhoA
        # RhoC shortening is alignment-weighted AND capped to prevent runaway
        l_shrink_rhoc = spring_cfg['rhoc_l_shrink'] * delta_rhoC * (1.0 - self.alignment)        
        
        L_target = self.L_init * (1.0 - l_shrink_rhoa - l_shrink_rhoc)
        L_target = max(L_target, self.L_init * 0.4) # Physical limit

        # Relax toward targets (First-order lag)
        alpha = dt / tau
        self.k_active += alpha * (k_target - self.k_active)
        self.L_rest_active += alpha * (L_target - self.L_rest_active)

    def calculate_forces(self):
        """
        Calculates the force vector and applies it to the two connected nodes.
        """
        if self.L_current < 1e-10:return
        force_vec = self.tension * self.unit_vec
        
        # Apply equal and opposite forces
        # CHANGE: tension should pull nodes together not tear them part
        self.node_1.apply_force(force_vec) # Negative
        self.node_2.apply_force(-force_vec) # Positive

    def get_spring_summary(self):
        """
        Returns a structured dictionary of the spring's current state.
        Use this for logging, CSV export, or detailed debugging.
        """
        return {
            "id": self.id,
            "geometry": {
                "L_curr": round(self.L_current, 3),
                "L_active": round(self.L_rest_active, 3),
                "tension": round(self.tension, 4),
                "align": round(self.alignment, 3)
            },
            "proteins": {
                "DSP": round(self.DSP, 3),
                "TJP1": round(self.TJP1, 3),
                "JCAD": round(self.JCAD, 3)
            },
            "signaling": {
                "RhoA": round(self.P_RhoA, 3),
                "RhoC": round(self.P_RhoC, 3)
            },
            "mechanics": {
                "k_active": round(self.k_active, 3)
            }
        }

    def __repr__(self):
        s = self.get_spring_summary()
        return (f"Spring {self.id} | L={s['geometry']['L_curr']} | "
                f"T={s['geometry']['tension']} | RhoA/C={s['signaling']['RhoA']}/{s['signaling']['RhoC']}")

