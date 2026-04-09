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
from abm.mechanics import bilinear_tension, relax_toward
from src.utils import require

class CortexSpring:
    """
    Cortical junction between two adjacent membrane nodes.
    """
    def __init__(self, spring_id, node_1, node_2, rest_length, cfg):
        self.id = spring_id
        self.node_1 = node_1
        self.node_2 = node_2

        # --- Cortical properties ---
        mech = cfg['mechanics']

        # Activation
        self.a_base = require(mech, 'a_cortex_base') # Pretension
        self.a_range = require(mech, 'a_cortex_range') # Contraction recruitment capacity
        self.a = self.a_base # Iniate at base
        self.tau_remodel = require(mech, 'tau_remodel')

        # Stiffness
        self.k_base = require(mech, 'k_cortex_base') # Basal contractile stiffness
        self.k_range = require(mech, 'k_cortex_range') # Stiffness increase capacity
        self.k = self.k_base # Initiate at base
        self.kc_ratio = require(mech, 'kc_ratio') # Compressive stiffness 

        # Rest Length and Tension
        self.L0 = rest_length 
        self.T = 0.0 

        # --- Geometry ---
        self.L = rest_length # Current/Activated length
        self.unit_vec = np.zeros(2) # Unit vector difference between nodes

    # ------------------------------------------------------------------
    # 1. Update Geometry and Tension
    # ------------------------------------------------------------------
    def update_geometry_tension(self):
        """
        Recompute spring geometry and tensions from current node positions. 
        Uses k_active set by update_stiffness() at previous step. 
        """
        # --- Geometry ---
        diff = self.node_2.pos - self.node_1.pos
        length = np.linalg.norm(diff)
        if length < 1e-10:
            return
        
        self.L = length
        self.unit_vec = diff / length
        
        # --- Tension ---
        # Bilinear tension using current k_active
        self.T = bilinear_tension(
            l=self.L, l0=self.L0 * self.a, 
            k=self.k, kc_ratio=self.kc_ratio
        ) 

    # ------------------------------------------------------------------
    # 2. Load Accumulation
    # ------------------------------------------------------------------
    def accumulate_loads(self):
        """
        Compute spring load contribution and add to endpoint nodes. 
        """
        load = max(self.T, 0.0)  # tensile only

        self.node_1.add_tensile_load(load)
        self.node_2.add_tensile_load(load)

    # ------------------------------------------------------------------
    # 3. Force Application
    # ------------------------------------------------------------------
    def apply_forces(self):
        """
        Apply equal and opposite mechanical forces on connected nodes.
        Pulls nodes together in the tensile regime. 
        Pushes nodes apart in the compressive regime. 
        """
        force_vec = self.T * self.unit_vec

        self.node_1.apply_force(force_vec)
        self.node_2.apply_force(-force_vec)

    # ------------------------------------------------------------------
    # 4. Update Cortex Stiffness and Activation (after signalling)
    # ------------------------------------------------------------------
    def update_stiffness_and_activation(self, dt):
        """
        Updates cortex stiffness from local RhoA.
        """
        # Compute RhoA signal as of connecting nodes
        local_rhoa = 0.5 * (self.node_1.P_RhoA + self.node_2.P_RhoA)
        rhoa_signal = float(np.clip(local_rhoa, 0.0, 1.0))

        # Instant stiffness update (operates on seconds timescale)
        # Baseline 1.0, max 3.0 when RhoA is 1.0
        self.k = self.k_base + (rhoa_signal * self.k_range)

        # First-order relaxation activation update towards RhoA-dependent target
        # Baseline 0.95, drops toward 0.75 at max RhoA
        a_target = self.a_base - (rhoa_signal * self.a_range)

        self.a = relax_toward(
            current=self.a, target=a_target, 
            dt=dt, tau=self.tau_remodel
        )

        self.a = float(np.clip(self.a , 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            'id': self.id,
            'extension': round(self.L - self.L0, 4),
            'stiffness': round(self.k, 4),
            'tension': round(self.T, 4),
            'activation': round(self.a, 3), 
        }

    def __repr__(self):
        return (
            f"Spring(id={self.id} | "
            f"L={self.L:.3f} L0={self.L0:.3f} | "
            f"k={self.k:.3f} | a={self.a:.3f} | T={self.T:.4f})"
        )