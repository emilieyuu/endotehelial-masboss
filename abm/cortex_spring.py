# abm/spring.py
#
# Cortical spring between adjacent membrane nodes. 
# 
# Responsibilities: 
#   1. Holds state (rest length, stiffness, activation, side, alignment)
#   2. Computes bilinear tension from current geometry
#   3. Applies equal-and-opposite forces on endpoint nodes
#   4. Adds tension to tensile load accumulator
#   5. Updates activation and stiffness based on mean RhoA from endpoint nodes

import numpy as np
from abm.mechanics import bilinear_tension

class CortexSpring:
    """
    Cortical junction between two adjacent membrane nodes.
    """
    def __init__(self, spring_id, node_1, node_2, rest_length, cfg):

        self.id = spring_id
        self.node_1 = node_1
        self.node_2 = node_2
        self.mech = cfg['mechanics']

        # --- Cortical properties ---
        # Activation
        self.a_cortex_base = self.mech.get('a_cortex_base', 0.95) # Pretension
        self.a_cortex_range = self.mech.get('a_cortex_range', 0.2) # Contraction recruitment capacity
        self.a_cortex = self.a_cortex_base # Iniate at base

        # Stiffness
        self.k_cortex_base = self.mech.get('k_cortex_base', 1.0) # Basal contractile stiffness
        self.k_cortex_range = self.mech.get('k_cortex_range', 1.0) # Stiffness increase capacity
        self.k_cortex = self.k_cortex_base # Initiate at base
        self.kc_ratio = self.mech.get('kc_ratio', 0.1) # Compressive stiffness 

        # Rest Length and Tension
        self.L_cortex = rest_length 
        self.t_cortex = 0.0 

        # --- Geometry ---
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
        self.t_cortex = bilinear_tension(
            l=self.L_current, l0=self.L_cortex * self.a_cortex, 
            k=self.k_cortex, kc_ratio=self.kc_ratio
        ) 

    # ------------------------------------------------------------------
    # 2. Load Accumulation
    # ------------------------------------------------------------------
    def accumulate_cortex_loads(self):
        """
        Compute spring load contribution and add to endpoint nodes. 
        """
        load = max(self.t_cortex, 0.0)  # tensile only

        self.node_1.add_tensile_load(load)
        self.node_2.add_tensile_load(load)

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
    # 4. Update Cortex Stiffness and Activation (after signalling)
    # ------------------------------------------------------------------
    def update_cortex_stiffness_and_activation(self, dt):
        """
        Updates cortex stiffness from local RhoA.
        """
        # Compute RhoA signal as of connecting nodes
        local_rhoa = 0.5 * (self.node_1.P_RhoA + self.node_2.P_RhoA)
        rhoa_signal = float(np.clip(local_rhoa, 0.0, 1.0))

        # Instant stiffness update (operates on seconds timescale)
        # Baseline 1.0, max 3.0 when RhoA is 1.0
        self.k_cortex = self.k_cortex_base + (rhoa_signal * self.k_cortex_range)

        # First-order relaxation activation update towards RhoA-dependent target
        # Baseline 0.95, drops toward 0.75 at max RhoA
        a_target = self.a_cortex_base - (rhoa_signal * self.a_cortex_range)
        rate = dt / self.mech.get('tau_remodel', 30)
        self.a_cortex += rate * (a_target - self.a_cortex)
        self.a_cortex = float(np.clip(self.a_cortex , 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            'id': self.id,
            'side': self.side,
            'extension': round(self.L_current - self.L_cortex, 4),
            'stiffness': round(self.k_cortex, 4),
            'tension': round(self.t_cortex, 4),
            'alignment': round(self.alignment, 3),
            'activation': round(self.a_cortex, 3)
        }

    def __repr__(self):
        return (
            f"Spring(id={self.id} | side={self.side} | "
            f"L={self.L_current:.3f} L0={self.L_cortex:.3f} | "
            f"k_cortex={self.k_cortex:.3f} | T={self.t_cortex:.4f})"
        )