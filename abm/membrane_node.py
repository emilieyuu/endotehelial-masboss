# abm_v2/membrane_node.py
import numpy as np
from abm.signalling import get_protein_recruitment

class MembraneNode:

    def __init__(self, node_id, position, lut, cfg):
        self.id = node_id
        self.pos = np.array(position, dtype=float)
        self.force = np.zeros(2)
        self.role = 'lateral'
        self.lut = lut
        self.cfg = cfg

        # Shear Inputs (set by EndothelialCell._apply_shear())
        self.f_normal = 0.0 # tensile component
        self.f_total = 0.0 # weighted magnitude for TJP1

        # Signalling state, computed by update_signalling()
        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
        self.P_RhoA, self.P_RhoC = 0.0,  0.0

    def update_signalling(self):
        """
        Compute junction protein recruitment local shear components.

        DSP:  tensile loading (f_normal) — pole-enriched
        TJP1: total shear magnitude (f_total) — near-uniform
        JCAD: tensile loading (f_normal) — pole-enriched
        
        Proteins → LUT → RhoA, RhoC stored for Spring and Cell.
        """
        # Get shear input
        tau_dsp  = max(self.f_normal, 0.0)  
        tau_tjp1 = max(self.f_total, 0.0) 
        tau_jcad = max(self.f_normal, 0.0) 

        # Compute junction protein recruitment
        self.DSP  = get_protein_recruitment(self.cfg, tau_dsp, 'DSP')
        self.TJP1 = get_protein_recruitment(self.cfg, tau_tjp1, 'TJP1')
        self.JCAD = get_protein_recruitment(self.cfg, tau_jcad, 'JCAD')

        # Compute RhoA/RhoC activation
        self.P_RhoA, self.P_RhoC = self.lut.query(self.DSP, self.TJP1, self.JCAD)

    def apply_force(self, force):
        """
        Accumulate force contribution to this node.  
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
        self.force = np.zeros(2) 
            
    def __repr__(self):
        return (
            f"MembraneNode(id={self.id} | role={self.role} | "
            f"pos={self.pos.round(2)} | "
            f"f_n={self.f_normal:.3f} | f_t={self.f_total:.3f} | "
            f"DSP={self.DSP:.3f} | TJP1={self.TJP1:.3f}) | JCAD={self.JCAD:.3f}) "
            f"P_RhoA={self.P_RhoA:.3f} P_RhoC={self.P_RhoC:.3f})"
        )