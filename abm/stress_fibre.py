# abm/stress_fibre.py
#
# Internal stress fibre cables connecting FA node pairs across the cell.
# Active contractile elements driven by global sf_contract from EndothelialCell.
#
# Two mechanical effects:
#   1. Axial cable tension  — pulls FA nodes toward each other
#   2. Lateral squeeze — inward force on boundary nodes near each cable
#
# Biology:
#   Actomyosin bundle spanning the cell along the flow axis.
#   Generates active contractile tension when RhoC is active.
#   No rest length — tension proportional to contractility × length.
#   Longer cell → more myosin motors engaged → more tension.
#
import numpy as np
from abm.mechanics import activated_bilinear

class StressFibre:
    """
    An interal Stress Fibre cable connecting the most polar node.s. 
    """

    def __init__(self, node_up, node_down, rest_length, cfg):
        self.node_up = node_up
        self.node_down = node_down
        self.cfg = cfg
        self.mech = cfg['mechanics']

        # Stress fibre properties
        self.L_sf = rest_length
        self.nu_sf = self.mech.get('nu_sf', 0.3) # Poisson ratio

        self.a_sf = 0.0  
        self.t_sf = 0.0 # axial cable tension

        # Geometry 
        self.L_current = 0.0
        self.unit_vec  = np.zeros(2)
        self.cable_mid = 0.0 # Mid-ponint (x, y) of cable


    # ------------------------------------------------------------------
    # 1. Update Geometry and Tension
    # ------------------------------------------------------------------
    def update_geometry_and_tension(self):
        """
        Recompute cable length and tension at the beginning of each timestep. 

        t_sf = k_sf × a_sf × L_current
            Active pretension — no rest length.
            Proportional to activation (a_sf) and length (L_current).
            Longer cell → more fibre engaged → more tension.
        """
        # --- Geometry ---
        diff   = self.node_down.pos - self.node_up.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        self.L_current = length
        self.unit_vec  = diff / length
        self.cable_mid = 0.5 * (self.node_up.pos + self.node_down.pos)

        # --- Tension ---
        k_sf = self.mech['k_sf_fraction'] * self.mech['k_cortex']
        kc_ratio = self.mech['kc_ratio'] #* self.mech['k_sf_fraction'] 

        self.t_sf = activated_bilinear(
            l_current=self.L_current,
            l_rest=self.L_sf,
            k=k_sf,
            kc_ratio=kc_ratio, 
            a=self.a_sf
        ) 

    # ------------------------------------------------------------------
    # 3. Squeeze force — called by EndothelialCell per boundary node
    # ------------------------------------------------------------------

    def apply_forces(self, nodes, positions):
        if self.t_sf < 1e-10:
            return

        self._apply_axial_forces(nodes)
        self._apply_lateral_squeeze(nodes, positions)
    
    def _apply_axial_forces(self, nodes):
        force = self.t_sf * self.unit_vec

        upstream_nodes = [n for n in nodes if n.role == "upstream"]
        downstream_nodes = [n for n in nodes if n.role == "downstream"]

        n = len(upstream_nodes)

        if n == 0:
            return
        
        for node in upstream_nodes:
            node.apply_force(force/n)

        for node in downstream_nodes:
            node.apply_force(-force/n)

    def _apply_lateral_squeeze(self, nodes, positions):
        if self.a_sf < 1e-6:
            return
        
        ux, uy = self.unit_vec
        perp_unit = np.array([-uy, ux])

        # Vectorised signed perpendicular distances
        rel = positions - self.cable_mid
        distances = rel @ perp_unit

        max_dist = np.max(np.abs(distances))
        if max_dist < 1e-10:
            return
        
        weights = distances / max_dist
        magnitudes = -self.nu_sf * self.t_sf * weights

        # Apply forces
        for node, mag in zip(nodes, magnitudes):
            if abs(mag) > 1e-10:
                node.apply_force(mag * perp_unit)

    # ------------------------------------------------------------------
    # Update SF Activation
    # ------------------------------------------------------------------
    def update_a_sf(self, global_rhoc, rhoc_baseline, dt):
        if global_rhoc <= rhoc_baseline:
            target = 0.0
        else:
            target = (global_rhoc - rhoc_baseline) / (1.0 - rhoc_baseline)

        alpha = dt / self.mech.get('tau_remodel', 30)
        self.a_sf += alpha * (target - self.a_sf)
        self.a_sf = float(np.clip(self.a_sf, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self):
        return {
            "length": float(self.L_current),
            "unit_vec": self.unit_vec.copy(),   # direction of pull
            "tension": float(self.t_sf),

            "force_vector": (self.t_sf * self.unit_vec).copy(),

            "cable_mid": float(self.cable_mid),
            "contractility": float(self.a_sf),
        }

    def __repr__(self):
        return (
            f"StressFibre("
            f"L={self.L_current:.3f} | "
            f"contractility={self.a_sf:.3f} | "
            f"T_sf={self.t_sf:.4f} | "
            f"cable_mid={self.cable_mid:.3f})"
        )