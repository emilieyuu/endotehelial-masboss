# abm/endothelial_cell.py
#
# 2D endothelial cell — closed ring of membrane nodes connected by cortical springs,
# with an internal stress fibre cable connecting the two pole nodes.
#
# Mechanical systems:
#   1. Cortical springs     — passive elastic resistance, RhoA-stiffened
#   2. Uniform shear        — tensile at poles, tangential at flanks
#   3. Stress fibre cable   — active contractile, RhoC-driven
#   4. Poisson SF squeeze   — lateral narrowing from SF contraction
#   5. FA anchoring         — shape stabilisation, RhoC-driven, tracks elongation
#   6. Soft pressure        — area conservation

import numpy as np
from abm.membrane_node import MembraneNode
from abm.spring import Spring
from abm.stress_fibre import StressFibre
from abm.analysis.cell_measurement import measure_forces, measure_shape
from src.utils import safe_mean
from abm.mechanics import weighted_poisson

class EndothelialCell:
    """
    2D endothelial cell — closed ring of membrane nodes connected by
    cortical springs, with internal SF cables between FA node pairs.

    Mechanical systems:
        1. Uniform shear (tensile component) — loads poles, drives signalling
        2. Cortical springs — passive elastic resistance, RhoA-stiffened
        3. SF cables — active contractile pretension, RhoC-driven
        4. SF lateral squeeze — Poisson narrowing, RhoC-driven
        5. FA anchoring — substrate anchors at pole nodes, passive
        6. Soft pressure — area conservation

    Signalling:
        Node-level: f_normal → DSP/TJP1/JCAD → LUT → P_RhoA, P_RhoC
        Spring-level: reads P_RhoA from endpoint nodes → k_active
        Cell-level: mean P_RhoC across nodes → a_sf → SF cables
    """
    def __init__(self, cell_id, centroid, lut, cfg,
                 n_nodes=16, radius=12.0, flow_direction=np.array([1.0, 0.0])):

        self.id = cell_id
        self.n_nodes = n_nodes
        self.cfg = cfg
        self.lut = lut

        # Normalised Flow Direction
        flow = np.asarray(flow_direction, dtype=float)
        self.flow_direction = flow / np.linalg.norm(flow)

        # Build node ring and classify
        self.nodes = self._init_node_ring(centroid, n_nodes, radius, lut, cfg)
        self._classify_nodes()
        self.springs = self._init_springs(lut, cfg)
        
        # Build FA nodes and SF cables
        self.fa_positions, self.stress_fibre = self._init_fa_and_sf()

        # Global SF activation — set each step from node P_RhoC
        self.a_sf = 0.0

        # Cell Area 
        self.target_area = self._compute_area() # fixed, acts as reference
        self.current_area = self.target_area # dynamic, remodelled to maintain "incompressible cytoplasm"
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_node_ring(self, centroid, n_nodes, radius, lut, cfg):
        """
        Place n_nodes evenly on a circle, offset by π/2 so node 0
        starts at the top. Pole nodes land on the flow axis.
        """
        init_ar = self.cfg['cell_geometry'].get('init_ar', 1.0)
        r_x     = radius * np.sqrt(init_ar)   # semi-axis along flow
        r_y     = radius / np.sqrt(init_ar)   # semi-axis perpendicular

        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) + np.pi/2
        nodes  = []
        for i, angle in enumerate(angles):
            pos = np.array([
                centroid[0] + r_x * np.cos(angle),
                centroid[1] + r_y * np.sin(angle),
            ])
            nodes.append(MembraneNode(i, pos, self.lut, self.cfg))

        return nodes

    def _classify_nodes(self):
        """
        Upstream/downstream: argmin/argmax projection onto flow axis.
        Lateral: all remaining nodes.
        """
        threshold = self.cfg['cell_geometry'].get('polar_threshold', 0.866)
        centroid = self.centroid

        for node in self.nodes:
            radial = node.pos - centroid
            radial_unit = radial / np.linalg.norm(radial)
            projection = np.dot(radial_unit, self.flow_direction)

            if projection > threshold:
                node.role = 'downstream'
            elif projection < -threshold:
                node.role = 'upstream'
            else:
                node.role = 'lateral'

    def _init_springs(self, lut, cfg):
        """
        Connect adjacent nodes in a ring.
        Store _init_alignment frozen at init.
        """
        springs = []
        for i in range(self.n_nodes):
            n1 = self.nodes[i]
            n2 = self.nodes[(i + 1) % self.n_nodes]

            # Classify polar and flank spring from their connecting nodes. 
            if n1.role in ('upstream', 'downstream') or n2.role in ('upstream', 'downstream'):
                side = 'polar'
            else: 
                side = 'flank'

            diff = n2.pos - n1.pos
            dist = np.linalg.norm(diff)

            s = Spring(
                spring_id=i, node_1=n1, node_2=n2,
                rest_length=dist, side=side, 
                lut=lut, cfg=cfg
            )

            springs.append(s)

        return springs
    
    def _init_fa_and_sf(self):
        """
        Use only the centre pole node (closest to flow axis) per side.
        One upstream FA, one downstream FA, one SF cable.

        Off-axis pole nodes are classified as upstream/downstream for
        signalling purposes but do NOT get FA anchoring — they are free
        to move laterally, allowing the cell to narrow correctly.
        """
        # Find the single node closest to flow axis on each side
        upstream_nodes = [n for n in self.nodes if n.role == 'upstream']
        downstream_nodes = [n for n in self.nodes if n.role == 'downstream']

        # Centre pole = node with smallest |y| on each side
        up_centre = min(upstream_nodes,   key=lambda n: abs(n.pos[1]))
        dn_centre = min(downstream_nodes, key=lambda n: abs(n.pos[1]))

        # FA positions fixed at init — only centre pole nodes
        fa_positions = {
            up_centre.id: up_centre.pos.copy(),
            dn_centre.id: dn_centre.pos.copy(),
        }

        # Calculate SF rest length
        sf_dist = np.linalg.norm(dn_centre.pos - up_centre.pos)

        # Single SF cable along flow axis
        stress_fibre = StressFibre(
            up_centre, dn_centre, 
            sf_dist, self.cfg)

        return fa_positions, stress_fibre
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self):
        return np.array([n.pos for n in self.nodes])

    @property
    def centroid(self):
        return self.positions.mean(axis=0)
    
    @property 
    def rhoc_mean(self):
        return float(np.mean([n.P_RhoC for n in self.nodes]))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _get_neighbours(self, node_idx):
        """
        Return (previous (idx), next (idx)) for a node in the ring.
        """
        return (node_idx - 1) % self.n_nodes, (node_idx + 1) % self.n_nodes

    def _compute_area(self):
        """Shoelace formula — polygon area from node positions."""
        pos  = self.positions
        x, y = pos[:, 0], pos[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

    def _compute_node_normal(self, node_idx):
        """
        Outward unit normal at node i.

        Computed from adjacent edge vectors:
            e1 = curr - prev  (incoming edge)
            e2 = next - curr  (outgoing edge)

        For CCW winding, outward normal of edge (a→b) is (-dy, dx).
        Average the normals of both adjacent edges for a smooth result.

        Used for:
            1. Computing f_normal at each node (tensile shear component)
            2. Applying pressure forces along outward normal
        """
        n  = self.n_nodes
        prev = self.nodes[(node_idx - 1) % n].pos
        curr = self.nodes[node_idx].pos
        nxt  = self.nodes[(node_idx + 1) % n].pos

        e1 = curr - prev
        e2 = nxt - curr

        # Outward normals — rotate each edge 90° CCW
        n1 = np.array([-e1[1],  e1[0]])
        n2 = np.array([-e2[1],  e2[0]])

        normal = n1 + n2
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-10 else np.zeros(2)
    
    # ------------------------------------------------------------------
    # Force methods
    # ------------------------------------------------------------------

    def _apply_shear(self, flow_field):
        """
        Apply shear force to all nodes and set f_normal for signalling.

        Two effects from the same uniform shear force:

        Mechanical effect:
            Full force vector applied to every node.
            Mean subtracted to remove rigid body translation —
            cell is substrate-anchored, net force absorbed by substrate.

        Signalling effect:
            f_normal = |F · n̂| at each node — tensile shear component.
            Highest at poles (normal ∥ flow → full projection).
            Zero at flanks (normal ⊥ flow → zero projection).
            Stored on node, read by update_signalling() this step.
        """
        f_magnitude   = flow_field.magnitude
        drag_fraction = self.cfg['flow'].get('drag_fraction', 0.1)
        drag = f_magnitude * drag_fraction

        for i, node in enumerate(self.nodes):
            normal = self._compute_node_normal(i)
            cos_theta = np.dot(self.flow_direction, normal)

            # Signalling
            node.f_normal = f_magnitude * abs(cos_theta)
            node.f_tangential = f_magnitude * np.sqrt(1 - cos_theta**2)

            # weight force by side -- realistically poles feel more than flank
            node.f_total = np.sqrt(node.f_normal**2 + 0.5 * node.f_tangential**2)

            # Mechanical drag — opposing forces at poles only
            if node.role == 'downstream':
                node.apply_force( self.flow_direction * drag)
            elif node.role == 'upstream':
                node.apply_force(-self.flow_direction * drag)

    def _apply_pressure(self):
        """
        Soft area conservation — only fires on area deficit.

        Pressure = k_area × (target_area - current_area)
        Applied outward along local normal at each node,
        weighted by arc length dL at that node.

        dL = 0.5 × (distance to prev + distance to next neighbour)
        This gives nodes covering more boundary more pressure force —
        biologically accurate: pressure acts on membrane area.

        Only fires when area_deficit > 0 — never inflates beyond target.
        This prevents pressure from amplifying spurious elongation.
        """
        area_deficit = self.target_area - self.current_area
        if area_deficit <= 0:
            return

        pressure = self.cfg['mechanics']['k_area'] * area_deficit
        n = self.n_nodes

        for i, node in enumerate(self.nodes):
            # Arc length assigned to this node
            prev = self.nodes[(i - 1) % n].pos
            nxt = self.nodes[(i + 1) % n].pos
            dL = 0.5 * (np.linalg.norm(node.pos - prev) + np.linalg.norm(nxt  - node.pos))
            normal = self._compute_node_normal(i)
            node.apply_force(pressure * dL * normal)

    # ------------------------------------------------------------------
    # Timestep
    # ------------------------------------------------------------------
    def step(self, flow_field, dt):
        """
        Advance one mechanical timestep.

        Order:
            Force accumulation (1-6) → integration (7) →
            signalling (8-9) → SF update (10) → area sync (11)

        Forces must all be accumulated before integration.
        Signalling reads geometry AFTER integration (current deformed state).
        SF cables updated last — a_sf drives next step's cable forces.
        """
        # -- Force accumulation --

        # 1. Uniform shear on all nodes
        self._apply_shear(flow_field)

        # 2. Cortical spring geometry and forces
        for spring in self.springs:
            spring.update_geometry_and_tension(self.flow_direction)

        for spring in self.springs:
            spring.apply_forces()

        # 3. SF geometry and and forces
        self.stress_fibre.update_geometry_and_tension()
        self.stress_fibre.apply_forces(self.nodes, self.positions)


        # 8. Soft pressure (area conservation)
        self.current_area = self._compute_area()
        self._apply_pressure()

        # -- Integration --
        # 9. Integrate node positions
        gamma_base = self.cfg['integration']['gamma']
        max_disp = self.cfg['integration']['max_displacement']
        k_fa = self.cfg['mechanics'].get('k_fa', 3.0)

        for node in self.nodes:
            if node.id in self.fa_positions:
                a = self.stress_fibre.a_sf
                gamma = gamma_base * (1.0 + k_fa * a)
            else:
                gamma = gamma_base

            node.integrate_step(dt, gamma, max_disp)
        
        # -- Signalling and remodelling --

        # 10. Node signalling — f_normal already set in step 1
        for node in self.nodes:
            node.update_signalling()

        for s in self.springs:
            s.update_a_cortex()

        # 12. Global a_sf from updated node P_RhoC
        self.stress_fibre.update_a_sf(
            global_rhoc=self.rhoc_mean, 
            rhoc_baseline=self.lut.rhoc_rest, 
            dt=dt
        )

        # 13. Sync area
        self.current_area = self._compute_area()
 
    # ------------------------------------------------------------------
    # Cell State
    # ------------------------------------------------------------------
    def get_state(self):
        shape = measure_shape(self)
        forces = measure_forces(self)

        return {
            'cell_id': self.id,
            # Shape
            'ar': shape['ar'],
            'orientation': shape['orientation'],
            'area_ratio': shape['area_ratio'],
            # Signalling
            'mean_rhoa': safe_mean([n.P_RhoA for n in self.nodes]),
            'mean_rhoc': safe_mean([n.P_RhoC for n in self.nodes]),
            'a_sf': round(self.stress_fibre.a_sf, 3),
            # Force distribution
            'sf_tension': forces['sf_tension'],
            'a_cortex_pole': forces['a_cortex_pole'],
        }
    
    def get_diagnostics(self):
        """Full force diagnostics — use for analysis, not sweeps."""
        return measure_forces(self)
    
    def __repr__(self):
        s = self.get_state()
        return (
            f"EndothelialCell(id={self.id} | "
            f"ar={s['ar']:.2f} | "
            f"area={s['area_ratio']:.3f} | "
            f"a_sf={s['a_sf']:.3f} | "
            f"RhoA={s['mean_rhoa']:.3f} RhoC={s['mean_rhoc']:.3f})"
        )