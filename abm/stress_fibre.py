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
from abm.mechanics import bilinear_tension
from abm.geometry import perpendicular, project_onto_axis, axial_projection

class StressFibre:
    """
    Contractile stress fibre cable along flow axis, connects upstream and downstream poles. 
    """

    def __init__(self, node_up, node_down, rest_length, cfg):
        self.node_up = node_up
        self.node_down = node_down
        self.mech = cfg['mechanics']

        
        self.a_sf_range = self.mech.get('a_sf_range', 0.3)
        self.a_sf_base = self.mech.get('a_sf_base', 0.81)
        self.a_sf = self.a_sf_base

        self.k_sf = self.mech.get('k_sf_fraction', 0.7) * self.mech.get('k_cortex', 1.0)
        self.nu_sf = self.mech.get('nu_sf', 0.3)

        # Stress fibre properties
        self.L_sf = rest_length #* self.a_sf
        self.t_sf = 0.0 # axial cable tension

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
        # self.t_sf = hookes_law(l=self.L_current, l0=self.L_sf * self.a_sf, 
        #                        k=self.k_sf) 
        self.t_sf = bilinear_tension(l=self.L_current, 
                                     l0=self.L_sf * self.a_sf, 
                                    k=self.k_sf) 
        
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
        """Unified SF forces: axial pull at poles, distributed parabolic squeeze at waist."""
        # No squeeze if fibre is slack
        if self.t_sf < 1e-6:
            return
        
    
        radius = self.L_current / 2
        
        # Axial coordinate along fibre, already normalized to [-1, 1] and clipped.
        # p=0 at waist, |p|=1 at poles. Vectorizes naturally over (N, 2) input.
        p = axial_projection(positions, self.cable_mid, self.axis_unit, radius)

        # Lateral coordinate (perpendicular to fibre): signed distance from axis.
        # Sign encodes which side of the fibre each node sits on.
        lateral = (positions - self.cable_mid) @ self.perp_unit

        # --- Squeeze weights: axial × lateral profile ---
        # Axial profile: parabolic, peaks at waist (p=0), zero at poles (|p|=1).
        # Non-negative by construction since p ∈ [-1, 1].
        axial_profile = 1.0 - p * p

        # Lateral profile: prefer flank nodes far from the axis. On-axis nodes
        # (lateral≈0) correctly get zero weight — no meaningful direction to push.
        lateral_profile = np.abs(lateral)

        weights = axial_profile * lateral_profile

        W = weights.sum()
        if W < 1e-12:
            return
        weights /= W

        # --- Force assembly ---
        # Total squeeze force the fibre distributes across the waist (Poisson coupling).
        F_total = self.t_sf * self.nu_sf

        # Direction: inward along perp axis. sign(lateral) points outward from the
        # fibre, so negate to push toward the axis. np.sign(0)=0 handles on-axis
        # nodes safely (and those already have zero weight anyway).
        f_mags = weights * F_total
        directions = -np.sign(lateral)

        # Assemble (N, 2) force vectors: scalar magnitudes × perp unit vector.
        forces = (f_mags * directions)[:, None] * self.perp_unit

        # --- Apply to nodes ---
        # Python loop at the API boundary. If node.apply_force just accumulates into
        # a field, consider a batched variant to keep this fully vectorized.
        for node, f in zip(nodes, forces):
            node.apply_force(f)

   

    # ------------------------------------------------------------------
    # 3. Update Activation
    # ------------------------------------------------------------------
    def update_sf_activation(self, mean_rhoc, dt):
        """
        SF activation from mean RhoC.
        a_sf = clip(rhoc_gain × mean_rhoc, 0, 1)
        First-order lag with tau_remodel models SF assembly timescale.
        """
        rhoc_signal = float(np.clip(mean_rhoc, 0.0, 1.0))
        a_target = self.a_sf_base - rhoc_signal * self.a_sf_range

        rate = dt / self.mech.get('tau_remodel', 30)
        self.a_sf += rate * (a_target - self.a_sf)
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