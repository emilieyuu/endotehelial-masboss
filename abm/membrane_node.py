# abm/membrane_node.py
#
# A "geometric" point on the cell membrane, containing a "state". 
#
# Responsibilities:
#   1. Mechanics
#   2. Load Accumulation
#   3. Signalling

import numpy as np
from abm.signalling import get_protein_recruitment

class MembraneNode:

    def __init__(self, node_id, position, lut, cfg):
        self.id = node_id
        self.pos = np.array(position, dtype=float)

        # --- Mechanics ---
        self.force = np.zeros(2)

        # --- Load channels --
        self.tensile_load = 0.0 # accumulated tension (magnitude + SF + cortex) 
        self.shear_total = 0.0 # magnitude of flow (shear stress)

        # --- Sigalling ---
        self.lut = lut
        self.cfg = cfg

        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
        self.P_RhoA, self.P_RhoC = 0.0,  0.0
        
    # ------------------------------------------------------------------
    # 1. Load & Force Reset 
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
        Reset force accumulation.  
        Called at the start of each timestep.
        """
        self.force[:] = 0.0

    # ------------------------------------------------------------------
    # Load & Force Accumulation 
    # ------------------------------------------------------------------
    def add_tensile_load(self, load):
        """
        Accumulate tensile loading for DSP recruitment
        """
        self.tensile_load += load

    def set_shear(self, shear):
        """
        Set shear magnitude felt by nodes (from flow)
        """
        self.shear_total = shear

    def apply_force(self, force):
        """
        Accumulate mechanical force.  
        """
        force = np.asarray(force)
        self.force += force

    # ------------------------------------------------------------------
    # 2. Force Integration
    # ------------------------------------------------------------------
    def integrate_step(self, dt, gamma, max_displacement=0.5):
        """
        Converts net force to displacement.

        Overdamped integration: dx = (F / gamma) × dt.
        Displacement clamped to max_displacement for numerical stability.
        """
        displacement = (self.force / gamma) * dt
        d_norm = np.linalg.norm(displacement)

        if d_norm > max_displacement:
            displacement = displacement / d_norm * max_displacement

        self.pos += displacement 

    # ------------------------------------------------------------------
    # 3. Siganlling
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
    # Getters
    # ------------------------------------------------------------------ 
    def get_rhoa(self): 
        return self.P_RhoA
    
    def get_rhoc(self):
        return self.P_RhoC
    
    def get_force(self):
        return self.force

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------  
    def get_state(self):
        return {
            'id': self.id, 
            'position': (self.pos[0].round(2), self.pos[1].round(2)), 
            'tensile_load': self.tensile_load, 
            'shear_total': self.shear_total, 
            'DSP': self.DSP, 
            'TJP1': self.TJP1, 
            'JCAD': self.JCAD, 
            'RhoA': self.P_RhoA, 
            'RhoC': self.P_RhoC
        }    
    
    def __repr__(self):
        return (
            f"MembraneNode(id={self.id} | pos={self.pos.round(2)} | "
            f"DSP={self.DSP:.3f} | TJP1={self.TJP1:.3f} | JCAD={self.JCAD:.3f} | "
            f"RhoA={self.P_RhoA:.3f} | RhoC={self.P_RhoC:.3f})"
        )