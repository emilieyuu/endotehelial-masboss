# abm/sf_spring.py
#
# A single contractile stress fibre spanning the cell along the flow axis. 
# 
# Models the central actin bundle that forms during flow exposure: 
# it connects the upstream and downstream polar nodes,
# loads polar junctions axially, and squeezes the waist laterally.

import numpy as np
from src.abm.helpers.mechanics import bilinear_tension, relax_toward
from src.abm.helpers.geometry import axial_coord, lateral_coord, perpendicular
from src.utils.config_utils import require

class StressFibreCable:
    """
    Contractile stress fibre cable along flow axis, 
    connects upstream and downstream poles. 
    """

    def __init__(self, node_up, node_down, rest_length, cfg):
        self.node_up = node_up
        self.node_down = node_down

        # --- Config-derived parameters ---
        sf_cfg = require(cfg, 'stress_fibre')
        mech_cfg = require(cfg, 'mechanics')

        # Activation: quiescent baseline + RhoC-driven contraction capacity
        self.a_base = require(sf_cfg, 'a_base')
        self.a_drop = require(sf_cfg, 'a_drop')
        self.a = self.a_base # start at baselien

        # Stiffness: fixed at init, scaled down from cortex
        self.k = require(mech_cfg, 'k_base') * require(sf_cfg, 'k_fraction')  

        # Poisson coupling coefficient: fraction of axial tension
        self.nu = require(sf_cfg, 'nu')

        # Shared mechanical paramters
        self.kc_ratio = require(mech_cfg, 'kc_ratio') # compression : tension stiffness
        self.tau_remodel = require(mech_cfg, 'tau_remodel')

         # --- Mechanical state ---
        self.L0 = rest_length
        self.L  = rest_length
        self.T  = 0.0

        # --- Geometric frame ---
        self.axis_unit = np.zeros(2)
        self.cable_mid = np.zeros(2)

    # ------------------------------------------------------------------
    # 1. Geometry and tension (called each step before force application)
    # ------------------------------------------------------------------
    def update_geometry_tension(self):
        """Recompute axis_unit, cable_mid, L, and T from current pole positions."""
        # --- Geometry ---
        diff = self.node_down.pos - self.node_up.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        self.L = length
        self.axis_unit = diff / length 
        self.cable_mid = 0.5 * (self.node_up.pos + self.node_down.pos)

        # --- Tension ---
        self.T = bilinear_tension(
            l=self.L, l0=self.L0 * self.a, 
            k=self.k, kc_ratio=self.kc_ratio
        ) 
        
    # ------------------------------------------------------------------
    # 2. Polar load contribution (axial-peak parabolic)
    # ------------------------------------------------------------------
    def accumulate_loads(self, polar_nodes):
        """
        Contribute axial SF tension as tensile stimulus at polar nodes.

        Junctional tension felt at cell-cell contacts despite the
        SF exerting no net axial force

        Weight peaks at the poles (|p|=1) and is zero at the waist (p=0)
        """
        half_L = self.L / 2
        if half_L < 1e-10:
            return

        for node in polar_nodes: 
            # Normalised axial coordinate: 0 at waist, ±1 at poles.
            p_axial = axial_coord(node.pos, self.cable_mid, self.axis_unit) / half_L 
            weight = p_axial * p_axial # parabolic, peaks at poles

            load = max(weight * self.T, 0.0)
            node.add_tensile_load(load)

    # ------------------------------------------------------------------
    # 3. Waist squeeze force (inverse parabolic, Poisson-coupled)
    # ------------------------------------------------------------------
    def apply_forces(self, nodes):
        """
        Apply inward lateral squeeze at the cell waist.

        Poisson coupling: converts fraction of axial tension into inward force. 
        Distribution: parabolic profile peaking at waist (p=0), zero at poles (|p|=1). 
        Normalisation: total magnitude is T × nu_sf, independent of node count.
        Direction: along ±perp_unit (perpendicular to the fibre axis, toward the axis)
        """
        if self.T < 1e-6:
            return
        
        positions = np.array([n.pos for n in nodes])
        
        # --- Node coordinates in local frame ---
        # Axial: normalised to [-1, 1]. Lateral: raw signed distance from axis.
        half_L = self.L / 2
        if half_L < 1e-10:
            return
        
        p_axial = axial_coord(positions, self.cable_mid, self.axis_unit) / half_L
        lateral = lateral_coord(positions, self.cable_mid, self.axis_unit)

        # --- Weight profile: parabolic axial, peaks at waist ---
        axial_profile = np.maximum(1.0 - p_axial * p_axial, 0.0)

        total = axial_profile.sum()
        if total < 1e-12:
            return
        weights = axial_profile / total

        # --- Total squeeze force from Poisson coupling ---
        F_total = self.T * self.nu

        # --- Per-node force assembly ---
        perp_unit = perpendicular(self.axis_unit)

        for node, w, lat in zip(nodes, weights, lateral):
            if abs(lat) < 1e-10:
                continue   #
            direction = -np.sign(lat) * perp_unit
            node.apply_force(w * F_total * direction)

            perp_unit = perpendicular(self.axis_unit)
   
    # ------------------------------------------------------------------
    # 4. Remodelling (called each step after signalling)
    # ------------------------------------------------------------------
    def update_activation(self, mean_rhoc, dt):
        """
        Update activation from mean cell-wide RhoC. 
        
        RhoC drives activation downward (a_base → a_base − a_range), 
            baseline 1.0, drops toward 0.70 at max RhoC
        """
        # First-order relaxation activation towards RhoC-dependent target
        a_target = self.a_base - (mean_rhoc * self.a_drop)
        self.a = relax_toward(
            current=self.a, target=a_target, 
            dt=dt, tau=self.tau_remodel
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            'extension': round(self.L - self.L0, 4),
            'stiffness': round(self.k, 4),
            'tension': round(self.T, 4),
            'activation': round(self.a, 3), 
            'axis': self.axis_unit.round(2)
        }

    def __repr__(self):
        return (
            f"StressFibre(L0={self.L0:.3f} L={self.L:.3f} | "
            f"k={self.k:.3f} a={self.a:.3f} T={self.T:.4f})"
        )