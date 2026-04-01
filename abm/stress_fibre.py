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
    Set of internal SF cables connecting upstream FA nodes to
    downstream FA nodes across the cell interior.

    Each cable:
        - Connects one upstream FA node to its paired downstream FA node
        - Generates axial tension proportional to k_sf x sigma x L_current
        - Generates lateral squeeze on boundary nodes within influence_radius

    FA positions are fixed at initialisation — substrate anchors.
    FA anchor force is computed separately in EndothelialCell.

    a_sf: global RhoC activation (contractility) (0-1), set each step by EndothelialCell
            0 = no fibres (TJP1-KO)
            1 = full contraction (DSP-KO)
    """

    def __init__(self, node_upstream, node_downstream, rest_length, cfg):
        self.node_upstream   = node_upstream
        self.node_downstream = node_downstream
        self.cfg             = cfg
        self.mech = cfg['mechanics']


        # Stress fibre properties
        mech = cfg['mechanics']
        self.L_sf = rest_length
        self.k_sf = self.mech.get('k_sf', 0.5) # cable stiffness, portion of cortex
        self.nu_sf = self.mech.get('nu_sf', 0.3) # Poisson ratio

        self.a_sf = 0.0  
        self.t_sf = 0.0 # axial cable tension

        # Geometry 
        self.L_current = 0.0
        self.unit_vec  = np.zeros(2)
        self.cable_mid = 0.0 # y midpoint of cable — squeeze reference

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    def get_sf_activation(self):
        return self.a_sf

    # ------------------------------------------------------------------
    # Update: Geometry and Tension
    # ------------------------------------------------------------------
    def update_geometry_and_tension(self):
        """
        Recompute cable length, unit vector, y midpoint, and tension.

        t_sf = k_sf × a_sf × L_current
            Active pretension — no rest length.
            Proportional to activation (a_sf) and length (L_current).
            Longer cell → more fibre engaged → more tension.
        """
        diff   = self.node_downstream.pos - self.node_upstream.pos
        length = np.linalg.norm(diff)

        if length < 1e-10:
            return

        self.L_current = length
        self.unit_vec  = diff / length
        self.cable_mid = 0.5 * (self.node_upstream.pos + self.node_downstream.pos)

        # Compute cable tension from Hooke's Law
        # self.t_sf = activated_hookes(
        #     l_current=self.L_current, 
        #     l_rest=self.L_sf, 
        #     k=self.k_sf,
        #     a=self.a_sf, 
        # )
        self.t_sf = activated_bilinear(
            l_current=self.L_current,
            l_rest=self.L_sf,
            k=self.k_sf,
            kc_ratio=self.cfg['mechanics']['kc_ratio']*self.k_sf, 
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

            "cable_y": float(self.cable_y),
            "contractility": float(self.a_sf),
        }

    def __repr__(self):
        return (
            f"StressFibre("
            f"L={self.L_current:.3f} | "
            f"contractility={self.a_sf:.3f} | "
            f"T_sf={self.t_sf:.4f} | "
            f"cable_y={self.cable_y:.3f})"
        )