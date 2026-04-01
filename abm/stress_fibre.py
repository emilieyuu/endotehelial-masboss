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
        self.a_sf = 0.0  
        self.t_sf = 0.0 # axial cable tension

        # Geometry 
        self.L_current = 0.0
        self.unit_vec = np.zeros(2)
        self.perp_unit = np.zeros(2)
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
        self.perp_unit = np.array([-self.unit_vec[1], self.unit_vec[0]])
        self.cable_mid = 0.5 * (self.node_up.pos + self.node_down.pos)

        # --- Tension ---
        k_cortex = self.mech.get('k_cortex', 1.0)
        k_sf = self.mech.get('k_sf_fraction', 0.5) * k_cortex
        kc_ratio = self.mech.get('kc_ratio', 0.1) 

        self.t_sf = activated_bilinear(
            l_current=self.L_current, l_rest=self.L_sf,
            k=k_sf, kc_ratio=kc_ratio, a=self.a_sf
        ) 

    # ------------------------------------------------------------------
    # 2. Apply Forces: Axial Contracion + Lateral Squeeze
    # ------------------------------------------------------------------
    def apply_forces(self, nodes, positions):
        if self.t_sf < 1e-10:
            return

        self._apply_axial_forces(nodes)
        self._apply_lateral_squeeze(nodes, positions)
    
    def _apply_axial_forces(self, nodes):
        """
        Contractile pull force along stress fibre axis. 

        Force is distributed evenly between all "polar" nodes to prevent deformation.
        """
        # Force as tension (magnitude) * direction
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
        """
        Lateral squeeze force on lateral nodes. 
        Weighed by perpendicular distance to SF cable. 
        """
        if self.a_sf < 1e-6:
            return

        # Vectorised signed perpendicular distances
        rel = positions - self.cable_mid
        distances = rel @ self.perp_unit
        
        max_dist = np.max(np.abs(distances))
        if max_dist < 1e-10:
            return
        
        weights = distances / max_dist
        nu_sf = self.mech.get('nu_sf', 0.6)
        forces = -nu_sf * self.t_sf * weights

        # Apply forces
        for node, force in zip(nodes, forces):
            if abs(force) > 1e-10:
                node.apply_force(force * self.perp_unit)

    # ------------------------------------------------------------------
    # 3. Update Activation
    # ------------------------------------------------------------------
    def update_activation(self, mean_rhoc, rhoc_rest, dt):
        """
        
        """
        if mean_rhoc <= rhoc_rest:
            target = 0.0
        else:
            target = (mean_rhoc - rhoc_rest) / (1.0 - mean_rhoc)

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