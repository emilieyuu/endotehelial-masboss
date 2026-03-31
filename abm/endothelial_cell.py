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

        self.id          = cell_id
        self.n_nodes     = n_nodes
        self.cfg         = cfg
        self.lut         = lut

        # Normalised Flow Direction
        flow = np.asarray(flow_direction, dtype=float)
        self.flow_direction = flow / np.linalg.norm(flow)

        # Cell Mechanics 
        mech_cfg = self.cfg['mechanics']
        self.k_cortex = mech_cfg['k_cortex']

        # Build node ring and classify
        self.nodes = self._init_node_ring(centroid, n_nodes, radius, lut, cfg)
        self._classify_nodes()

        self.springs = self._init_springs(lut, cfg)
        self._classify_springs()
        
        # Build FA nodes and SF cables
        # fa_nodes: dict {node_id: fa_position} — fixed substrate positions
        # stress_fibres: list of StressFibre, one per FA pair
        self.fa_nodes, self.fa_max_displacement, self.stress_fibres = self._init_fa_and_sf()

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
        centroid  = self.centroid

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

            diff = n2.pos - n1.pos
            dist = np.linalg.norm(diff)

            s = Spring(
                spring_id=i, node_1=n1, node_2=n2,
                rest_length=dist,
                k_cortex=self.k_cortex,
                lut=lut, cfg=cfg
            )

            springs.append(s)
        return springs
    
    def _classify_springs(self):
        """
        Polar springs: connected to a pole node.
        Lateral springs: both nodes are lateral.
        """
        for s in self.springs:
            if s.node_1.role in ('upstream', 'downstream') or s.node_2.role in ('upstream', 'downstream'):
                s.side = 'polar'
            else: 
                s.side = 'flank'
    
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
        fa_nodes = {
            up_centre.id: up_centre.pos.copy(),
            dn_centre.id: dn_centre.pos.copy(),
        }

        fa_max_displacement = {
            up_centre.id:  up_centre.pos.copy(),
            dn_centre.id:  dn_centre.pos.copy(),
        }

        # Calculate SF rest length
        sf_dist = np.linalg.norm(dn_centre.pos - up_centre.pos)

        # Single SF cable along flow axis
        stress_fibres = [StressFibre(up_centre, dn_centre, sf_dist, self.cfg)]

        return fa_nodes, fa_max_displacement, stress_fibres
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self):
        return np.array([n.pos for n in self.nodes])

    @property
    def centroid(self):
        return self.positions.mean(axis=0)

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

            # Signalling
            node.f_normal = f_magnitude * abs(
                np.dot(self.flow_direction, normal)
            )
            node.f_total = f_magnitude

            # Mechanical drag — opposing forces at poles only
            if node.role == 'downstream':
                node.apply_force( self.flow_direction * drag)
            elif node.role == 'upstream':
                node.apply_force(-self.flow_direction * drag)

    def _apply_fa_anchoring(self):
        """
        FA substrate anchor forces on pole nodes.

        Each FA is a fixed point on the substrate set at initialisation.
        As the cell deforms, pole nodes drift from their FA positions.
        The FA pulls the node back: F = -k_fa × (node.pos - fa.pos)

        When node sits exactly on FA: force = 0
        When node drifts away: force pulls it back proportionally

        This is purely passive — no RhoC scaling.
        The RhoC effect on shape comes from SF cables and squeeze,
        not from FA grip strength.

        Combined with SF cable tension (which pulls poles inward),
        the FA anchoring resists that inward pull — the balance
        determines the steady-state elongated position.
        """
        mech = self.cfg['mechanics']
        k_fa_base = mech.get('k_fa', 2.0)
        fa_sf_gain = mech.get('fa_sf_gain', 1.5)  # NEW: FA strengthens with SF
        k_fa = k_fa_base * (1.0 + fa_sf_gain * self.a_sf)

        for node in self.nodes:
            if node.id not in self.fa_nodes:
                continue

            # Current axial displacement from centroid
            axial_pos = np.dot(node.pos, self.flow_direction)
            fa_axial  = np.dot(
                self.fa_max_displacement[node.id], self.flow_direction
            )

            # Ratchet — only update FA position if pole moved further out
            if abs(axial_pos) > abs(fa_axial):
                self.fa_max_displacement[node.id] = node.pos.copy()

            # FA force — resist displacement from maximum reached position
            rest = self.fa_max_displacement[node.id]
            disp = node.pos - rest

            # Axial component only — FAs resist axial contraction
            axial_disp = np.dot(disp, self.flow_direction) * self.flow_direction
            node.apply_force(-k_fa * axial_disp)

    def apply_sf_axial_forces(self):
        """
        Apply SF axial tension, distributed across FA node + neighbours.

        Distribution:
            centre node: 50%
            neighbours:  25% each
        """
        for sf in self.stress_fibres:

            if sf.t_sf < 1e-10:
                continue

            # Base force vector
            force = sf.t_sf * sf.unit_vec

            # Upstream side
            up_idx = sf.node_upstream.id
            up_prev, up_next = self._get_neighbours(up_idx)

            self.nodes[up_idx].apply_force(0.5 * force)
            self.nodes[up_prev].apply_force(0.25 * force)
            self.nodes[up_next].apply_force(0.25 * force)

            # Downstream side
            dn_idx = sf.node_downstream.id
            dn_prev, dn_next = self._get_neighbours(dn_idx)

            self.nodes[dn_idx].apply_force(-0.5 * force)
            self.nodes[dn_prev].apply_force(-0.25 * force)
            self.nodes[dn_next].apply_force(-0.25 * force)

    def _apply_sf_squeeze(self):
        """
        Lateral squeeze from SF contraction — computed by Cell.

        For each SF cable, for each boundary node:
            d = node.pos[1] - cable.cable_y
            weight = |d| / max_y_distance  (0 at poles, 1 at max flank)
            F = sf.get_squeeze_force(node.pos[1], max_y_distance)

        max_y_distance computed per cable across all nodes —
        normalises the weight so the most distant node always gets
        weight=1, regardless of cell shape.

        Applied as y-only force — squeeze is purely lateral.
        """
        for sf in self.stress_fibres:
            if sf.a_sf < 1e-6:
                continue
            
            # Perpendicular direction to fibre
            ux, uy = sf.unit_vec
            perp = np.array([-uy, ux])

            # Max y-distance from this cable across all nodes
            max_d = max(
                abs(np.dot(n.pos - sf.cable_mid, perp)) for n in self.nodes
            )

            if max_d < 1e-10:
                continue

            for node in self.nodes:
                f_squeeze = sf.get_squeeze_force(node, max_d)

                if abs(f_squeeze) > 1e-10:
                    node.apply_force(np.array(f_squeeze * perp))

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
        n        = self.n_nodes

        for i, node in enumerate(self.nodes):
            # Arc length assigned to this node
            prev = self.nodes[(i - 1) % n].pos
            nxt  = self.nodes[(i + 1) % n].pos
            dL   = 0.5 * (
                np.linalg.norm(node.pos - prev) +
                np.linalg.norm(nxt  - node.pos)
            )

            normal = self._compute_node_normal(i)
            node.apply_force(pressure * dL * normal)

    # ------------------------------------------------------------------
    # Signalling and Remodelling
    # ------------------------------------------------------------------
    def _update_a_sf(self, dt):
        """
        Update global SF activation (a_sf) from mean node P_RhoC.

        Biology:
            RhoC is a global whole-cell signal — SF contractility is
            uniform across all cables, not spatially varying.
            Mean P_RhoC across ALL nodes (not just lateral) gives the
            whole-cell RhoC activity level.

        a_sf target:
            delta_rhoC = mean(P_RhoC) - rhoc_rest
            a_sf_target = delta_rhoC / rhoc_max  (normalised 0-1)

        First-order lag toward target with tau_remodel:
            Models the finite timescale of SF assembly/disassembly.
            Fast assembly when RhoC rises, slow disassembly when it falls.

        Sets a_sf on all SF cables.
        """
        mech = self.cfg['mechanics']

        mean_rhoc = float(np.mean([n.P_RhoC for n in self.nodes]))
        delta_rhoc = max(mean_rhoc - self.lut.rhoc_rest, 0.0)
        rhoc_max = mech['delta_rhoc_max']

        a_sf_target  = min(delta_rhoc / rhoc_max, 1.0)

        alpha = dt / mech['tau_remodel']
        self.a_sf += alpha * (a_sf_target - self.a_sf)
        self.a_sf = float(np.clip(self.a_sf, 0.0, 1.0))

        # Propagate to all SF cables
        for sf in self.stress_fibres:
            sf.a_sf = self.a_sf

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

        # 2. Cortical spring geometry 
        for spring in self.springs:
            spring.update(self.flow_direction)

        # 3. Cortical spring forces
        for spring in self.springs:
            spring.apply_forces()

        # 4. SF geometry and tension
        for sf in self.stress_fibres:
            sf.update()


        self.apply_sf_axial_forces()

        # 6. SF lateral squeeze
        self._apply_sf_squeeze()

        # 7. FA anchoring (passive substrate anchors)
        self._apply_fa_anchoring()

        # 8. Soft pressure (area conservation)
        self.current_area = self._compute_area()
        self._apply_pressure()

        # -- Integration --
        # 9. Integrate node positions
        gamma = self.cfg['integration']['gamma']
        max_disp = self.cfg['integration']['max_displacement']
        for node in self.nodes:
            node.integrate_step(dt, gamma, max_disp)
        
        # -- Signalling and remodelling --

        # 10. Node signalling — f_normal already set in step 1
        for node in self.nodes:
            node.update_signalling()


        # 12. Global a_sf from updated node P_RhoC
        self._update_a_sf(dt)

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
            'a_sf': forces['a_sf'],
            # Force distribution
            'sf_tension': forces['sf_tension'],
            'cortex_k_pole': forces['cortex_k_pole'],
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