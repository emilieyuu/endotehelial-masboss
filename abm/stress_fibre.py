# abm/stress_fibre.py
#
# Internal stress fibre cables connecting FA node pairs across the cell.
# Active contractile elements driven by global sf_contract from EndothelialCell.
#
# Two mechanical effects:
#   1. Axial cable tension  — pulls FA nodes toward each other
#   2. Lateral squeeze — inward force on boundary nodes near each cable

# Biology:
#   Actomyosin bundle spanning the cell along the flow axis.
#   Generates active contractile tension when RhoC is active.
#   No rest length — tension proportional to contractility × length.
#   Longer cell → more myosin motors engaged → more tension.
#
import numpy as np

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

    def __init__(self, node_upstream, node_downstream, cfg):
        self.node_upstream   = node_upstream
        self.node_downstream = node_downstream
        self.cfg             = cfg

        # Stress fibre properties
        mech = cfg['mechanics']
        self.k_sf = mech.get('k_sf_fraction', 0.5) # cable stiffness, portion of cortex
        self.nu_sf = mech.get('nu_sf', 0.3) # Poisson ratio
        self.a_sf = 0.0  # set each step by EndothelialCell
        self.t_sf = 0.0 # axial cable tension

        # Geometry 
        self.L_current = 0.0
        self.unit_vec  = np.zeros(2)
        self.cable_y = 0.0 # y midpoint of cable — squeeze reference

    # ------------------------------------------------------------------
    # 1. Geometry and tension
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
        self.cable_y   = 0.5 * (self.node_upstream.pos[1] + self.node_downstream.pos[1])
        self.t_sf = self.k_sf * self.a_sf * self.L_current


    # ------------------------------------------------------------------
    # 2. Axial cable forces on FA nodes
    # ------------------------------------------------------------------

    def apply_forces(self):
        """
        Pull upstream and downstream FA nodes toward each other.

        Combined with FA anchoring (which resists inward movement),
        this creates a stable equilibrium at the elongated length:
            - Without SF: cortex recoil wins → cell rounds up
            - With SF: cable tension + FA anchoring balance cortex → stays elongated
        """
        if self.t_sf < 1e-10:
            return

        force = self.t_sf * self.unit_vec
        self.node_upstream.apply_force(force) # pulled in +x, towards downstream
        self.node_downstream.apply_force(-force) # pulled in -x, towards upstream

    # ------------------------------------------------------------------
    # 3. Squeeze force — called by EndothelialCell per boundary node
    # ------------------------------------------------------------------

    def get_squeeze_force(self, node_y, max_y_distance):
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
        if self.a_sf < 1e-6 or max_y_distance < 1e-10:
            return 0.0

        perb_dist = node_y - self.cable_y
        weight = abs(perb_dist) / max_y_distance
        return -self.nu_sf * self.a_sf * weight * np.sign(perb_dist)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_squeeze_profile(self, nodes):
        """
        Returns squeeze force applied to each node.
        Used for diganostics and debugging. 
        """
        if self.a_sf < 1e-6:
            return {}

        max_y = max(abs(n.pos[1] - self.cable_y) for n in nodes)
        if max_y < 1e-10:
            return {}

        profile = {}

        for n in nodes:
            f = self.get_squeeze_force(n.pos[1], max_y)
            profile[n.id] = round(f, 4)

        return profile

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