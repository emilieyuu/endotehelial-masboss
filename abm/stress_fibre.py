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
from abm.geometry import perpendicular, project_onto_axis, axial_projection

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
        self.nu_sf = self.mech.get('nu_sf', 0.3)
        self.contract_frac = self.mech.get('sf_contract_fraction', 0.1)
        self.k_sf = self.mech.get('k_sf_fraction', 0.7) * self.mech.get('k_cortex', 1.0)

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
        self.t_sf = hookes_law(l=self.L_current, l0=self.L_sf, 
                               k=self.k_sf) * self.a_sf
        
    # ------------------------------------------------------------------
    # 2. Load Accumulation
    # ------------------------------------------------------------------
    def accumulate_sf_loads(self, polar_nodes):
        """
        Accumulate SF axial tension for junction loading. 
        """
        radius = self.L_current / 2

        for node in polar_nodes: 
            projection = axial_projection(node.pos, self.cable_mid, self.axis_unit, radius)
            weight = projection**2
            load = max(weight * self.t_sf, 0.0)
            node.tensile_load += load

    # ------------------------------------------------------------------
    # 3. Force Application
    # ------------------------------------------------------------------

    def apply_sf_forces(self, nodes, positions):
        """Unified SF forces: Axial pull at poles, Parabolic squeeze at waist."""
        if self.t_sf < 1e-6:
            return
        
        radius = self.L_current / 2

        for node in nodes: 
            proj = axial_projection(node.pos, self.cable_mid, self.axis_unit, radius)

            # Axial Contraction
            f_axial_mag = (proj**2) * self.t_sf * self.contract_frac
            node.apply_force(-np.sign(proj) * f_axial_mag * self.axis_unit)

            # squeeze
            dx_vec = node.pos - self.cable_mid    
            side_sign = np.sign(np.dot(dx_vec, self.perp_unit))        
            f_sq_mag = (1-proj**2) * self.t_sf * self.nu_sf
            node.apply_force(-side_sign * f_sq_mag * self.perp_unit)
        
        # # Calculate half-length of the cell for normalization
        # x_coords = [n.pos[0] for n in nodes]
        # max_dx = (np.max(x_coords) - np.min(x_coords)) / 2
        
        # nu_sf = self.mech.get('nu_sf', 0.3) # Scaled squeeze ratio

        # for node in nodes:
        #     # Distance from center projected onto SF axis
        #     dx_vec = node.pos - self.cable_mid
        #     dx_mag = np.dot(dx_vec, self.axis_unit) 
            
        #     # Normalize distance (0 at center, 1 at poles)
        #     dist_norm = min(abs(dx_mag) / max_dx, 1.0)

        #     # AXIAL PULL
        #     # weight_axial = dist_norm**2 
        #     # f_axial = -np.sign(dx_mag) * self.t_sf * weight_axial * 0.1 # 0.1 contraction fraction
        #     # node.apply_force(f_axial * self.axis_unit)

        #     # LATERAL SQUEEZE
        #     # weight_sq = 1 - (dist_norm**2)
            
            # # Calculate which side of the SF the node is on
            # side_sign = np.sign(np.dot(dx_vec, self.perp_unit))
            
            # # Apply inward force
            # f_sq = -nu_sf * self.t_sf * side_sign * weight_sq * self.perp_unit
            # node.apply_force(f_sq)

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