# abm/spring.py
#
# Cortical spring between adjacent membran nodes. 
# Stiffness modulated by RhoA: k_active = k_cortex + (rhoa_gain * mean_rhoa)
# Tension from bilinear law: stiff in tension, soft in compression

import numpy as np
from abm.mechanics import bilinear_tension

class Spring:
    """
    Cortical junction between two adjacent membrane nodes.
    """

    def __init__(self, spring_id, node_1, node_2, rest_length, cfg):
        self.id = spring_id
        self.node_1, self.node_2 = node_1, node_2
        self.mech = cfg['mechanics']

        # Cortical properties
        self.k_cortex = self.mech.get('k_cortex', 1.0) # Basal stiffness
        self.L_cortex = rest_length # Initiation length
        self.k_active = self.k_cortex # Effective stiffness, modulated by RhoA
        self.t_cortex = 0.0 # Tension of spring

        # Geometry
        self.L_current = rest_length # Current/Activated length
        self.unit_vec = np.zeros(2) # Unit vector difference between nodes
        self.alignment = 0.0 # cos angle to flow
        if self.node_1.role in ('upstream', 'downstream') or self.node_2.role in ('upstream', 'downstream'):
            self.side = 'polar'
        else: 
            self.side = 'flank'

    # ------------------------------------------------------------------
    # 1. Update Geometry and Tension
    # ------------------------------------------------------------------
    def update_cortex_geometry_tension(self, flow_direction):
        """
        Recompute spring geometry and tensions from current node positions. 
        Uses k_active set by update_stiffness() at previous step. 
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
        # Bilinear tension using current k_active
        kc_ratio = self.mech.get('kc_ratio', 0.1)
        self.t_cortex = bilinear_tension(
            l=self.L_current, l0=self.L_cortex, 
            k=self.k_active, kc_ratio=kc_ratio
        ) 

    # ------------------------------------------------------------------
    # 2. Load Accumulation
    # ------------------------------------------------------------------
    def accumulate_cortex_loads(self):
        """
        Compute spring signalling load contribution
        """
        load = max(self.t_cortex, 0.0)  # tensile only

        self.node_1.tensile_load += load
        self.node_2.tensile_load += load

    # ------------------------------------------------------------------
    # 3. Force Application
    # ------------------------------------------------------------------
    def apply_cortex_forces(self):
        """
        Apply equal and opposite mechanical forces on connected nodes.
        Pulls nodes together in the tensile regime. 
        Pushes nodes apart in the compressive regime. 
        """
        force_vec = self.t_cortex * self.unit_vec

        self.node_1.apply_force(force_vec)
        self.node_2.apply_force(-force_vec)

    # ------------------------------------------------------------------
    # 4. Update Cortex Stiffness (after signalling)
    # ------------------------------------------------------------------
    def update_cortex_stiffness(self):
        """
        Updates cortex stiffness from local RhoA.
        """
        # Compute mean RhoA of connecting nodes
        local_rhoa = 0.5 * (self.node_1.P_RhoA + self.node_2.P_RhoA)
   
        # Compute activation directly of RhoA level
        rhoa_gain = self.mech.get('rhoa_gain', 4.0) 
        rhoa_signal = rhoa_gain * max(local_rhoa, 0.0) 

        # Instant stiffness update (operates on seconds timescale)
        self.k_active = self.k_cortex + rhoa_signal # k_cortex as stiffness baseline

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            'id': self.id,
            'side': self.side,
            'extension': round(self.L_current - self.L_cortex, 4),
            'stiffness': round(self.k_active, 4),
            'tension': round(self.t_cortex, 4),
            'alignment': round(self.alignment, 3)
        }

    def __repr__(self):
        return (
            f"Spring(id={self.id} | side={self.side} | "
            f"L={self.L_current:.3f} L0={self.L_cortex:.3f} | "
            f"k_active={self.k_active:.3f} | T={self.t_cortex:.4f})"
        )