# abm_v2/spring.py

import numpy as np
from abm.mechanics import bilinear_tension


class Spring:
    """
    Cortical junction between two adjacent membrane nodes.

    Purely mechanical — no signalling.
    Stiffness derived from RhoA of endpoint nodes:
        delta_rhoA = mean(node_1.P_RhoA, node_2.P_RhoA) - rhoa_rest
        k_active   = k_cortex × (1 + rhoa_k_gain × delta_rhoA)

    Tension from bilinear law:
        stretched: T = k_active × (L - L0)
        compressed: T = k_active × kc_ratio × (L - L0)
    """

    def __init__(self, spring_id, node_1, node_2,
                 rest_length, k_cortex, lut, cfg):

        self.id = spring_id
        self.lut = lut
        self.cfg = cfg
        self.node_1 = node_1
        self.node_2 = node_2

        # Cortical properties
        self.L_cortex = rest_length
        self.k_cortex = k_cortex
        self.k_active = k_cortex      # updated each step by EndothelialCell
        self.t_cortex = 0.0

        # Geometry
        self.L_current = rest_length
        self.unit_vec = np.zeros(2)
        self.alignment = 0.0
        self._init_alignment = 0.0    # frozen at init by EndothelialCell

    # ------------------------------------------------------------------
    # 1. Geometry 
    # ------------------------------------------------------------------
    def update_geometry(self, flow_direction):
        """
        Recompute length and alignment from current node positions.
        """
        diff = self.node_2.pos - self.node_1.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        self.L_current = length
        self.unit_vec  = diff / length
        self.alignment = abs(np.dot(self.unit_vec, flow_direction))

    # ------------------------------------------------------------------
    # 2. Tension & Stiffness
    # ------------------------------------------------------------------
    def update_cortex(self):
        """
        Recompute stiffness and tension from RhoA activation at endpoints. 
        """
        mech = self.cfg['mechanics']

        # Get RhoA from mean of endpoints nodes
        mean_rhoa = 0.5 * (self.node_1.P_RhoA + self.node_2.P_RhoA)
        delta_rhoa = max(mean_rhoa - self.lut.rhoa_rest, 0.0) # get RhoA activity above rest
        rhoa_max = mech['delta_rhoa_max']

        # Recompute stiffness from updated RhoA activation
        self.k_active = self.k_cortex * (
            1.0 + mech['rhoa_k_gain'] * delta_rhoa
        )

        # Set updated cortical tension.
        self.t_cortex = bilinear_tension(
            l_current=self.L_current,
            l_rest=self.L_cortex,
            k_tensile=self.k_active,
            kc_ratio=mech['kc_ratio']
        )


    # ------------------------------------------------------------------
    # 3. Force application
    # ------------------------------------------------------------------

    def apply_forces(self):
        """
        Equal and opposite cortical forces on connected nodes.
        """
        if self.L_current < 1e-10:
            return
        force_vec = self.t_cortex * self.unit_vec
        self.node_1.apply_force(force_vec)
        self.node_2.apply_force(-force_vec)


    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        threshold  = self.cfg['cell_geometry'].get('polar_threshold', 0.85)
        population = 'lateral' if self._init_alignment > threshold else 'polar'

        return {
            'id':             self.id,
            'population':     population,
            'L_current':      round(self.L_current,      4),
            'L_cortex':       round(self.L_cortex,       4),
            'k_active':       round(self.k_active,       4),
            'alignment':      round(self.alignment,       3),
            'init_alignment': round(self._init_alignment, 3),
            'tension':        round(self.t_cortex,         4),
        }

    def __repr__(self):
        return (
            f"Spring(id={self.id} | "
            f"L={self.L_current:.3f} L0={self.L_cortex:.3f} | "
            f"k={self.k_active:.3f} | "
            f"T={self.t_cortex:.4f})"
        )