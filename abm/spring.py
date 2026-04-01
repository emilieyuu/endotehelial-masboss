# abm/spring.py

import numpy as np
from abm.mechanics import activated_bilinear

class Spring:
    """
    Cortical junction between two adjacent membrane nodes.
    """

    def __init__(self, spring_id, node_1, node_2,
                 rest_length, side, cfg):

        self.id = spring_id
        self.node_1, self.node_2 = node_1, node_2
        self.mech = cfg['mechanics']

        # Cortical properties
        self.L_cortex = rest_length # Length at initiation – constant
        self.a_cortex = 0.95 # RhoA activated tension gain
        self.t_cortex = 0.0 # Tension (force) of spring

        # Geometry
        self.L_current = rest_length # Current (activated length)
        self.unit_vec = np.zeros(2) # Unit vector difference between nodes
        self.alignment = 0.0 # cos angle to flow
        self.side = side # Pole or Flank depending on role of nodes

    # ------------------------------------------------------------------
    # 1. Update Geometry and Tension (From previous timestep)
    # ------------------------------------------------------------------
    def update_geometry_and_tension(self, flow_direction):
        """
        Recompute length, alignment and tension at the beginning of each timestep. 

        Tension using bilinear law: 
            - Stretched (L > L0): T = k * (L - L0) * a
            - Compressed (L0 > L): T = k * 0.1 * (L - L0) * a
        """
        # --- Geometry ---
        diff = self.node_2.pos - self.node_1.pos

        length = np.linalg.norm(diff)
        if length < 1e-10:
            return
        
        self.L_current = length
        self.unit_vec = diff / length
        self.alignment = abs(np.dot(self.unit_vec, flow_direction))
        
        # --- Tension ---
        # Tensile and Compressive stiffness from config
        k_cortex = self.mech['k_cortex']
        kc_ratio = self.mech['kc_ratio']
        
        # Calculate tension using activated bilinear law
        self.t_cortex = activated_bilinear(
            l_current=self.L_current, l_rest=self.L_cortex,
            k=k_cortex, kc_ratio=kc_ratio, a=self.a_cortex
        ) 

    # ------------------------------------------------------------------
    # 2. Force application
    # ------------------------------------------------------------------
    def apply_forces(self):
        """
        Apply equal and opposite cortical forces on connected nodes.
        Pulls nodes together in the tensile regime. 
        Pushes nodes apart in the compressive regime. 
        """
        # if self.L_current < 1e-10:
        #     return
        
        # Force as tension (magnitude) * direction
        force_vec = self.t_cortex * self.unit_vec

        self.node_1.apply_force(force_vec)
        self.node_2.apply_force(-force_vec)
    
    # ------------------------------------------------------------------
    # Update Cortex Activation (after signalling)
    # ------------------------------------------------------------------
    def update_activation(self):
        """
        Updates cortex activation factor directly from mean 
        RhoA concentration between nodes. 

        Uses direct RhoA scaling because cortex always has some stiffness. 
            -> Directly represent "stiffness above baseline", as baseline is never 0. 
        """
        # Compute mean RhoA of connecting nodes
        mean_rhoa = 0.5 * (self.node_1.P_RhoA + self.node_2.P_RhoA)

        # Compute activation directly of RhoA level
        rhoa_k_gain = self.mech.get('rhoa_k_gain', 5.0)
        self.a_cortex = 1.0 + rhoa_k_gain * mean_rhoa

        # alpha = dt / mech['tau_remodel']
        # self.a_cortex += alpha * (a_cortex_target - self.a_cortex)
        # self.a_cortex = float(np.clip(self.a_cortex, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        # threshold  = self.cfg['cell_geometry'].get('polar_threshold', 0.85)
        # population = 'lateral' if self._init_alignment > threshold else 'polar'

        return {
            'id': self.id,
            'side': self.side,
            'L': round(self.L_current, 4),
            'L0': round(self.L_cortex, 4),
            'activation': round(self.a_cortex, 4),
            'tension': round(self.t_cortex, 4),
            'alignment': round(self.alignment, 3)
        }

    def __repr__(self):
        return (
            f"Spring(id={self.id} | "
            f"side={self.side} | "
            f"L={self.L_current:.3f} L0={self.L_cortex:.3f} | "
            f"activation={self.a_cortex:.3f} | "
            f"T={self.t_cortex:.4f})"
        )