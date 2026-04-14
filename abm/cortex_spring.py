# abm/spring.py
#
# Cortical spring between adjacent membrane nodes. 
# 
# Cortex springs form a ring that discretises the cell cortex;
# their combined tension resists deformation and their stiffness tracks
# local RhoA activation at each endpoint.

import numpy as np
from abm.helpers.mechanics import bilinear_tension, relax_toward
from src.utils import require

class CortexSpring:
    """
    Cortical junction between two adjacent membrane nodes.

    State:
      L0, L — rest length and current length
      k — current stiffness (RhoA-dependent)
      a - current activation (RhoA-dependent, relaxed toward target)
      T — current bilinear tension from (L, L0 × a, k)
      unit_vec — unit direction from node_1 to node_2
    """
    def __init__(self, id, node_1, node_2, rest_length, cfg):
        self.id = id
        self.node_1 = node_1
        self.node_2 = node_2

        # --- Config-derived parameters ---
        cortex_cfg = require(cfg, 'cortex')
        mech_cfg = require(cfg, 'mechanics')

        # Activation: baseline pretension + RhoA-driven contraction capacity.
        self.a_base = require(cortex_cfg, 'a_base') 
        self.a_drop = require(cortex_cfg, 'a_drop') 
        self.a = self.a_base # start at baseline       

        # Stiffness: baseline + RhoA-driven stiffening capacity.
        self.k_base = require(mech_cfg, 'k_base')
        self.k_gain = require(cortex_cfg, 'k_gain') 
        self.k = self.k_base # start at baseline

        # Shared mechanical paramters
        self.kc_ratio = require(cfg, 'mechanics', 'kc_ratio') # compression : tension stiffness
        self.tau_remodel = require(cfg, 'mechanics', 'tau_remodel')

        # --- Mechanical state ---
        self.L0 = rest_length 
        self.L = rest_length 
        self.T = 0.0 

        # --- Geometry ---
        self.unit_vec = np.zeros(2)

    # ------------------------------------------------------------------
    # 1. Geometry and tension (called each step before force application)
    # ------------------------------------------------------------------
    def update_geometry_tension(self):
        """Recompute L, unit_vec, and T from current node positions."""
        # --- Geometry ---
        diff = self.node_2.pos - self.node_1.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return
        
        self.L = length
        self.unit_vec = diff / length
        
        # --- Tension ---
        # Tension uses the bilinear law with effective rest length L0 × a.
        # Stiffness k set from previous step.
        self.T = bilinear_tension(
            l=self.L, l0=self.L0 * self.a, 
            k=self.k, kc_ratio=self.kc_ratio
        ) 

    # ------------------------------------------------------------------
    # 2. Load contribution (tensile-only stimulus to both endpoints)
    # ------------------------------------------------------------------
    def accumulate_loads(self):
        """Contribute tensile stimulus to endpoint nodes."""
        load = max(self.T, 0.0)  # no load in compression
        self.node_1.add_tensile_load(load)
        self.node_2.add_tensile_load(load)

    # ------------------------------------------------------------------
    # 3. Force application (equal and opposite along the spring axis)
    # ------------------------------------------------------------------
    def apply_forces(self):
        """
        Apply ±T × unit_vec to the endpoints.

        Positive T pulls the endpoints together (tensile regime);
        negative T pushes them apart (compressive regime).
        """
        force_vec = self.T * self.unit_vec
        self.node_1.apply_force(force_vec)
        self.node_2.apply_force(-force_vec)

    # ------------------------------------------------------------------
    # 4. Remodelling (called each step after signalling)
    # ------------------------------------------------------------------
    def update_stiffness_and_activation(self, dt):
        """
        Update stiffness (instantaneous) and activation (relaxed) from
        mean RhoA endpoint nodes.
     
        RhoA drives stiffness upward: k = k_base + rhoa × k_range (fast), 
            baseline 1.0, max 3.0 when RhoA is 1.0
        RhoA drives activation downward: a → a_base − rhoa × a_range (slow),
            baseline 0.95, drops toward 0.75 at max RhoA

        The fast/slow split reflects the physical separation between
        motor engagement (seconds) and cytoskeletal turnover (minutes).
        """
        # Mean endpoint RhoA.
        mean_rhoa = 0.5 * (self.node_1.rhoa + self.node_2.rhoa)

        # Instant stiffness update
        self.k = self.k_base + (mean_rhoa * self.k_gain)

        # First-order relaxation towards RhoA-dependent target activation
        a_target = self.a_base - (mean_rhoa * self.a_drop)
        self.a = relax_toward(
            current=self.a, target=a_target, 
            dt=dt, tau=self.tau_remodel
        )

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
            f"Spring(id={self.id} | L={self.L:.3f} L0={self.L0:.3f} | "
            f"k={self.k:.3f} | a={self.a:.3f} | T={self.T:.4f})"
        )