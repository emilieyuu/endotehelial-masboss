# abm/stress_fibre.py
#
# Stress fibre cable connecting upstream and downstream pole nodes. 
#
# Two mechanical effects:
#   1. Axial contraction — pulls poles nodes inward (distributed across pole nodes)
#   2. Lateral squeeze — pushes flank nodes inward (Poisson coupling)
#
# Tensions from Hooke's Law (tensile only, slack in compression). 
# Activation (a_sf) from mean RhoC via first order lag. 

import numpy as np
from abm.mechanics import hookes_law
from abm.geometry import project_all, max_abs_projection, perpendicular

class StressFibre:
    """
    Contractile stress fibre cable along flow axis, connects upstream and downstream poles. 
    """

    def __init__(self, node_up, node_down, rest_length, cfg):
        self.node_up = node_up
        self.node_down = node_down
        self.mech = cfg['mechanics']

        # Stress fibre properties
        self.L_sf = rest_length 
        self.a_sf = 0.0 
        self.t_sf = 0.0 # axial cable tension
        self.nu_sf = cfg['mechanics']['nu_sf']

        # Geometry 
        self.L_current = 0.0
        self.axis_unit = np.zeros(2)
        self.perp_unit = np.zeros(2)
        self.cable_mid = 0.0 

    # ------------------------------------------------------------------
    # 1. Update Geometry and Tension
    # ------------------------------------------------------------------
    def update_sf_geometry_tension(self):
        """
        Recompute SF geometry an tension. 
        Tension = k_sf × max(L - L0, 0) × a_sf.
        Zero when slack (L <= L0) or inactive (a_sf ≈ 0).
        """
        # --- Geometry ---
        diff = self.node_down.pos - self.node_up.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        self.L_current = length
        
        # --- Define axes ---
        self.axis_unit = diff / length # flow/SF axis
        self.perp_unit = perpendicular(self.axis_unit)
        self.cable_mid = 0.5 * (self.node_up.pos + self.node_down.pos)

        # --- Tension ---
        k_sf = self.mech.get('k_sf_fraction', 0.5) * self.mech.get('k_cortex', 1.0)
        self.t_sf = hookes_law(l=self.L_current, l0=self.L_sf, 
                               k=k_sf) * self.a_sf
        
    # ------------------------------------------------------------------
    # 3. Load Accumulation
    # ------------------------------------------------------------------
    def accumulate_sf_loads(self, positions, nodes, centroid):
        """
        Accumulate axial tension contributions to signalling. 
        """
        if self.t_sf < 1e-6: 
            return
        
        projections = project_all(positions, centroid, self.axis_unit)
        max_proj = max_abs_projection(projections)

        for node, dx_mag in zip(nodes, projections):
            weight_axial = (abs(dx_mag) / max_proj)**2
            # print(f"tension {node.id}, dx_mag: {dx_mag}")
            # print(f"tension {node.id}, weight_axial: {weight_axial}")

            node.tensile_load += self.t_sf * weight_axial

    # ------------------------------------------------------------------
    # 3. Force Application
    # ------------------------------------------------------------------

    def apply_sf_forces(self, nodes, positions):
        """Unified internal SF forces: Axial contraction and Gaussian squeeze."""
        if self.t_sf < 1e-6:
            return
        
        centroid = np.mean([n.pos for n in nodes], axis=0)
        x_coords = [n.pos[0] for n in nodes]
        max_dx = (np.max(x_coords) - np.min(x_coords)) / 2
        
        nu_sf = self.mech.get('nu_sf', 1.2)
        sigma = self.L_current * 0.20 # Tight squeeze for the waist

        for node in nodes:
            dx_vec = node.pos - centroid
            dx_mag = np.dot(dx_vec, self.axis_unit) # Projection on SF axis
            
            # 1. AXIAL INTERNAL PULL (Poles Only)
            if node.role in ('upstream', 'downstream'):
                weight_axial = (abs(dx_mag) / max_dx)**2
                # Pull INWARD toward centroid
                f_axial = -np.sign(dx_mag) * self.t_sf * weight_axial
                # node.apply_force(f_axial * self.axis_unit)
                node.tensile_load += self.t_sf * weight_axial

            # 2. LATERAL GAUSSIAN SQUEEZE (All nodes)
            weight_sq = np.exp(-(dx_mag**2) / (2 * sigma**2))
            side_sign = np.sign(np.dot(dx_vec, self.perp_unit))
            f_sq = -nu_sf * self.t_sf * weight_sq * side_sign * self.perp_unit
            node.apply_force(f_sq)

    def _apply_axial_forces(self, positions, nodes, centroid):
        """
        Apply axial contraction forces. 
        """
        projections = project_all(positions, centroid, self.axis_unit)
        max_proj = max_abs_projection(projections)

        for node, dx_mag in zip(nodes, projections):
            weight_axial = (abs(dx_mag) / max_proj)**2

            # Direction toward centroid
            f_axial = -np.sign(dx_mag) * self.t_sf * weight_axial

            node.apply_force(f_axial * self.axis_unit)

    # ------------------------------------------------------------------
    # 3. Update Activation
    # ------------------------------------------------------------------
    def update_sf_activation(self, mean_rhoc, dt):
        """
        SF activation from mean RhoC.
        a_sf = clip(rhoc_gain × mean_rhoc, 0, 1)
        First-order lag with tau_remodel models SF assembly timescale.
        """
        rhoc_gain = self.mech.get('rhoc_gain', 1.0)
        rhoc_signal = rhoc_gain * max(mean_rhoc, 0.0)
        a_target = float(np.clip(rhoc_signal, 0.0, 1.0))
        
        alpha = dt / self.mech.get('tau_remodel', 30)
        self.a_sf += alpha * (a_target - self.a_sf)
        self.a_sf = float(np.clip(self.a_sf, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            "length": float(self.L_current),
            "rest_length": float(self.L_sf),
            "tension": float(self.t_sf),
            "activation": float(self.a_sf),
        }

    def __repr__(self):
        return (
            f"StressFibre(L={self.L_current:.3f} L0={self.L_sf:.3f} | "
            f"a_sf={self.a_sf:.3f} | T={self.t_sf:.4f})"
        )