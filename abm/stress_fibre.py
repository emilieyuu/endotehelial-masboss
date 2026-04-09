# abm/stress_fibre.py
#
# Stress fibre cable connecting upstream and downstream pole nodes. 

import numpy as np
from abm.mechanics import bilinear_tension, relax_toward
from src.utils import require
from abm.geometry import axial_coord, lateral_coord, perpendicular

class StressFibre:
    """
    Contractile stress fibre cable along flow axis, connects upstream and downstream poles. 
    """

    def __init__(self, node_up, node_down, rest_length, cfg):
        self.node_up = node_up
        self.node_down = node_down

        # --- SF Porperties --- 
        mech = cfg['mechanics']
        self.tau_remodel = require(mech, 'tau_remodel')

        # Activation
        self.a_base = require(mech, 'a_sf_base')# No pretension, relaxed at initiation
        self.a_range = require(mech, 'a_sf_range') # Contraction recruitment capacity 
        self.a = self.a_base # Initiate at base

        # Stiffness
        self.k = require(mech, 'k_sf_fraction') * require(mech, 'k_cortex_base') # Fraction of cortex stiffness
        self.kc_ratio = require(mech, 'kc_ratio')
        self.nu_sf = require(mech, 'nu_sf') # Poisson-coupling coefficient

        # Rest length and tension
        self.L0 = rest_length 
        self.T = 0.0 # axial cable tension

        # --- Geometry ---
        self.L = rest_length
        self.axis_unit = np.zeros(2)
        self.cable_mid = np.zeros(2)

    # ------------------------------------------------------------------
    # 1. Update Geometry and Tension
    # ------------------------------------------------------------------
    def update_geometry_tension(self):
        """
        Recompute SF geometry an tension. 
        """
        # --- Geometry ---
        diff = self.node_down.pos - self.node_up.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        self.L = length
        
        # --- Define axes ---
        self.axis_unit = diff / length # flow/SF axis
        self.cable_mid = 0.5 * (self.node_up.pos + self.node_down.pos)

        # --- Tension ---
        self.T = bilinear_tension(
            l=self.L, l0=self.L0 * self.a, 
            k=self.k, kc_ratio=self.kc_ratio
        ) 
        
    # ------------------------------------------------------------------
    # 2. Load Accumulation
    # ------------------------------------------------------------------
    def accumulate_loads(self, polar_nodes):
        """
        Accumulate SF axial tension for junction loading. 

        Axial SF tension used purely for loading, modelling tension
        felt by cells at junctions despite net-zero mechanical force. 

        Unlike apply_forces (which squeezes the waist), loading peaks at
        the poles — this reflects where the cable physically attaches and
        where junctional tension is felt by the cell. The p² weight and
        the (1 - p²) weight in apply_forces are deliberately inverse.
        """
        # Half-length of fibre, used to normlaise axial position to [-1, 1
        half_L = self.L / 2

        for node in polar_nodes: 
            # Raw axial distance from the fibre midpoint, along the fibre axis.
            axial = axial_coord(node.pos, self.cable_mid, self.axis_unit)

            # Normalise to [-1, 1]. 
            p = axial / half_L # |p| = 1 at the poles and p = 0 at the waist.
            weight = p * p # Parabolic weight: peaks at the poles.

            load = max(weight * self.T, 0.0)
            node.add_tensile_load(load)

    # ------------------------------------------------------------------
    # 3. Force Application
    # ------------------------------------------------------------------
    def apply_forces(self, nodes, positions):
        """
        Apply SF lateral squeeze – inward at waist, zero at poles. 

        Lateral SF contraction applied net-zero force due to 
        "neighbouring cell forces".

        Poisson coupling converts a fraction of axial tension into 
        inwards lateral squeeze. 

        Distribution is parabolic in axial coordinate. Each node pushed 
        toward SF axis. 
        """
        if self.T < 1e-6:
            return
        
        # Normalised axial coord, parabolic profile peaks at waist
        half_L = self.L / 2
        p = axial_coord(positions, self.cable_mid, self.axis_unit) / half_L
        axial_profile = np.maximum(1.0 - p * p, 0.0)

        # Normalise weights so total squeeze = T × nu_sf
        total = axial_profile.sum()
        if total < 1e-12:
            return
        weights = axial_profile / total

        F_total = self.T * self.nu_sf

        # Direction: inward along the vector from each node to the fibre midpoint.
        # Lateral coordinated projected onto perpendicular axis for sign.
        lateral = lateral_coord(positions, self.cable_mid, self.axis_unit)
        perp_unit = perpendicular(self.axis_unit)

        # Each node's force points along -sign(lateral) * perp_unit,
        for node, w, lat in zip(nodes, weights, lateral):
            if abs(lat) < 1e-10:
                continue   # on-axis node, no direction to push
            direction = -np.sign(lat) * perp_unit
            node.apply_force(w * F_total * direction)

   
    # ------------------------------------------------------------------
    # 3. Update Activation
    # ------------------------------------------------------------------
    def update_activation(self, mean_rhoc, dt):
        """
        SF activation from mean cell RhoC.
        """
        # Compute RhoC signal from cell-wide mean RhoC
        rhoc_signal = float(np.clip(mean_rhoc, 0.0, 1.0))

        # First-order relaxation activation update towards RhoC-dependent target
        # Baseline 1.0, drops toward 0.70 at max RhoC
        a_target = self.a_base - (rhoc_signal * self.a_range)

        self.a = relax_toward(
            current=self.a, target=a_target, 
            dt=dt, tau=self.tau_remodel
        )

        self.a = float(np.clip(self.a, 0.0, 1.0))

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
            f"StressFibre(L={self.L:.3f} L0={self.L0:.3f} | "
            f"a_sf={self.a:.3f} | T={self.T:.4f})"
        )