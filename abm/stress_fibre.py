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
from abm.mechanics import activated_hookes

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

    def __init__(self, node_upstream, node_downstream, rest_length,  cfg):
        self.node_upstream   = node_upstream
        self.node_downstream = node_downstream
        self.cfg             = cfg

        # Stress fibre properties
        mech = cfg['mechanics']
        self.k_sf = mech.get('k_sf', 0.5) # cable stiffness, portion of cortex
        self.nu_sf = mech.get('nu_sf', 0.3) # Poisson ratio
        self.a_sf = 0.0  # set each step by EndothelialCell
        #elf.a_sf = mech.get('a_sf', 0.4)
        self.t_sf = 0.0 # axial cable tension
        self.L_sf = rest_length

        # Geometry 
        self.L_current = 0.0
        self.unit_vec  = np.zeros(2)
        self.cable_mid = 0.0 # y midpoint of cable — squeeze reference

    # ------------------------------------------------------------------
    # Update: Geometry and Tension
    # ------------------------------------------------------------------
    def update(self):
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
        self.t_sf = activated_hookes(
            l_current=self.L_current, 
            l_rest=self.L_sf, 
            k=self.k_sf,
            a=self.a_sf, 
        )

    # ------------------------------------------------------------------
    # 3. Squeeze force — called by EndothelialCell per boundary node
    # ------------------------------------------------------------------

    def get_squeeze_force(self, node, max_perp_distance):
        """
        Lateral squeeze force magnitude for a node at position node_y.

        Returns scalar force in y-direction (signed):
            positive → push node upward (node is below cable)
            negative → push node downward (node is above cable)

        Formula:
            d = node_y - cable_y (signed distance from cable)
            weight = |d| / max_y_distance (0 at poles, 1 at max flank)
            F = -nu_sf × a_sf × weight × sign(d)

        node_y: node.pos[1]
        max_y_distance:  max |node_y - cable_y| across all boundary nodes
                         computed by EndothelialCell before calling this
        """
        if self.a_sf < 1e-6 or max_perp_distance < 1e-10:
            return 0.0
        
        ux, uy = self.unit_vec
        perp = np.array([-uy, ux])

        perp_dist = np.dot(node.pos - self.cable_mid, perp)
        weight = abs(perp_dist) / max_perp_distance

        return -self.nu_sf * self.t_sf * weight * np.sign(perp_dist)
    

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