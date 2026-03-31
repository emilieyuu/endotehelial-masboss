# abm_v2/membrane_node.py
import numpy as np
from abm.signalling import get_protein_recruitment

class MembraneNode:

    def __init__(self, node_id, position, lut, cfg):
        self.id = node_id
        self.pos = np.array(position, dtype=float)
        self.force = np.zeros(2)
        self.role = 'lateral'
        self.lut   = lut
        self.cfg   = cfg

        # Shear input, set by EndothelialCell at each step
        self.f_normal = 0.0 
        self.f_total = 0.0

        # Signalling state, computed by update_signalling()
        self.DSP    = 0.0
        self.TJP1   = 0.0
        self.JCAD   = 0.0
        self.P_RhoA = 0.0
        self.P_RhoC = 0.0

    def update_signalling(self):
        """
        Compute junction protein recruitment and Rho activity
        from local normal shear component.

        f_normal: tensile shear at this node = tau × |n̂ · ê_x|. 
        Highest at poles, zero at flanks.

        DSP, TJP1, JCAD: all isotropic for now (no alignment weighting)
        Output: P_RhoA, P_RhoC stored for reading by Spring and Cell
        """
        tau_dsp  = max(self.f_normal, 0.0)   # tensile component — local
        tau_tjp1 = self.f_total # full shear magnitude — global
        tau_jcad = max(self.f_normal, 0.0) # same as DSP 

        self.DSP  = get_protein_recruitment(self.cfg, tau_dsp, 'DSP')
        self.TJP1 = get_protein_recruitment(self.cfg, tau_tjp1, 'TJP1')
        self.JCAD = get_protein_recruitment(self.cfg, tau_jcad, 'JCAD')

        self.P_RhoA, self.P_RhoC = self.lut.query(
            self.DSP, self.TJP1, self.JCAD
        )

    def apply_force(self, force):
        """
        Accumulate force contribution to this node.  
        """
        force = np.asarray(force)
        if not np.all(np.isfinite(force)):
            raise ValueError(
                f"Node {self.id} received non-finite force: {force}"
            )
        self.force += force

    def integrate_step(self, dt: float, gamma: float, max_displacement: float = 1.0):
        """
        Integrate position forward one timestep. 

        dt: mechanical timestep. 
        gamma: viscous drag coefficient – higher = slower movement. 
        max_displacement: soft cap on movement per step. 
            Prevents nodes teleporting if forces spike during exploration. 
            Should be ≈ 10% of rest length, pass dynamically. 
        """
        displacement = (self.force / gamma) * dt
        d_norm = np.linalg.norm(displacement)

        if d_norm > max_displacement:
            displacement = displacement / d_norm * max_displacement

        self.pos += displacement # update node position
        self.force = np.zeros(2) # reset for next time step
    
    def __repr__(self):
        return (
            f"MembraneNode(id={self.id} | role={self.role} | "
            f"pos={self.pos.round(2)} | "
            f"f_normal={self.f_normal:.3f} | "
            f"P_RhoA={self.P_RhoA:.3f} P_RhoC={self.P_RhoC:.3f})"
        )