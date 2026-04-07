# abm/membrane_node.py
#
#

import numpy as np
from abm.signalling import get_protein_recruitment

class MembraneNode:

    def __init__(self, node_id, position, lut, cfg):
        self.id = node_id
        self.pos = np.array(position, dtype=float)

        # --- Mechanics ---
        self.force = np.zeros(2)
        self.role = 'lateral'

        # --- Load channels ---
        self.f_tensile_load = 0.0
        self.f_total_load = 0.0
        self.tensile_load = 0.0 # accumulated tension (SF + cortex)
        self.shear_total = 0.0 # from flow – total magnitude

        # --- Sigalling ---
        self.lut = lut
        self.cfg = cfg

        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
        self.P_RhoA, self.P_RhoC = 0.0,  0.0
    
    # ------------------------------------------------------------------
    # Load & Force Reset 
    # ------------------------------------------------------------------
    def reset_loads(self):
        """
        Reset signalling loads. 
        Called at the start of each timestep.
        """
        self.tensile_load = 0.0
        self.shear_total = 0.0

    def reset_force(self):
        """
        Reset signalling loads. 
        Called at the start of each timestep.
        """
        self.force[:] = 0.0

    # ------------------------------------------------------------------
    # Force Accumulation & Integration
    # ------------------------------------------------------------------
    def apply_force(self, force):
        """
        Accumulate mechanical force.  
        """
        force = np.asarray(force)
        self.force += force

    def integrate_step(self, dt, gamma, max_displacement=0.5):
        """
        Overdamped integration: dx = (F / gamma) × dt.
        Displacement clamped to max_displacement for numerical stability.
        """
        displacement = (self.force / gamma) * dt
        d_norm = np.linalg.norm(displacement)

        if d_norm > max_displacement:
            displacement = displacement / d_norm * max_displacement

        self.pos += displacement 

    # ------------------------------------------------------------------
    # Siganlling
    # ------------------------------------------------------------------
    def update_signalling(self):
        """
        Compute junction protein recruitment from accumulated loads

        DSP:  tensile loading 
        TJP1: total shear magnitude 
        JCAD: total shear magnitude 
        
        Proteins → LUT → RhoA, RhoC.
        """
        # Clamp inputs
        tau_dsp  = max(self.tensile_load, 0.0)  
        tau_tjp1 = max(self.shear_total, 0.0) 
        tau_jcad = max(self.shear_total, 0.0) 

        # Junction protein recruitment
        self.DSP  = get_protein_recruitment(self.cfg, tau_dsp, 'DSP')
        self.TJP1 = get_protein_recruitment(self.cfg, tau_tjp1, 'TJP1')
        self.JCAD = get_protein_recruitment(self.cfg, tau_jcad, 'JCAD')

        # RhoA/RhoC activation
        self.P_RhoA, self.P_RhoC = self.lut.query(self.DSP, self.TJP1, self.JCAD)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------      
    def __repr__(self):
        return (
            f"MembraneNode(id={self.id} | role={self.role} | "
            f"pos={self.pos.round(2)} | "
            f"f_n={self.f_normal:.3f} | f_t={self.f_total:.3f} | "
            f"DSP={self.DSP:.3f} | TJP1={self.TJP1:.3f}) | JCAD={self.JCAD:.3f}) "
            f"P_RhoA={self.P_RhoA:.3f} P_RhoC={self.P_RhoC:.3f})"
        )