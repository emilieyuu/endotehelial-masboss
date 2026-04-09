# abm/endothelial_cell.py
#
# 2D endothelial cell — closed ring of membrane nodes connected by 
# cortical springs with an internal stress fibre cable.
#
# Forces:
#   1. Shear drag       — extensional at poles, amplified by SF tension
#   2. Cortex springs   — bilinear, RhoA-stiffened
#   3. SF contraction   — inward pull at poles (distributed)
#   4. SF squeeze       — inward at flanks (Poisson coupling)
#   5. Area pressure    — outward when area < target

import numpy as np
from abm.membrane_node import MembraneNode
from abm.cortex_spring import CortexSpring
from abm.stress_fibre import StressFibre
from abm.analysis.cell_measurement import measure_forces, measure_shape
from abm.geometry import axial_coord, axial_projection
from src.utils import safe_mean

class EndothelialCell:
    def __init__(self, cell_id, centroid, lut, cfg,
                 n_nodes=16, radius=12.0, flow_direction=np.array([1.0, 0.0])):

        self.id = cell_id
        self.n_nodes = n_nodes
        self.cfg = cfg
        self.lut = lut

        flow_direction = np.asarray(flow_direction, dtype=float)
        self.flow_direction = flow_direction 
        self.flow_axis = flow_direction / np.linalg.norm(flow_direction)
        self.radius = radius

        # Build geometry
        self.nodes = self._init_node_ring(centroid, n_nodes, radius, lut, cfg)
        self._classify_nodes()
        self.springs = self._init_springs(n_nodes, cfg)
        self.sf = self._init_sf()
        
        # Area conservation
        self.target_area = self._compute_area() # fixed
        self.current_area = self.target_area # dynamic
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_node_ring(self, centroid, n_nodes, radius, lut, cfg):
        """
        Place n_nodes evenly on a circle, offset by π/2 so node 0
        starts at the top. 
        """
        init_ar = self.cfg['cell_geometry'].get('init_ar', 1.2)
        r_x = radius * np.sqrt(init_ar) # semi-axis along flow
        r_y = radius / np.sqrt(init_ar) # semi-axis perpendicular to flow

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
            projection = axial_projection(node.pos, centroid, self.flow_axis, self.radius)

            if projection > threshold:
                node.role = 'downstream'
            elif projection < -threshold:
                node.role = 'upstream'
            else:
                node.role = 'lateral'

    def _init_springs(self, n_nodes, cfg):
        """
        Connect adjacent nodes in a ring.
        """
        springs = []

        # Create a spring for any two adjacent nodes on the ring. 
        for i in range(n_nodes):
            # Get nodes and distance between them 
            n1 = self.nodes[i]
            n2 = self.nodes[(i + 1) % n_nodes]
            dist = np.linalg.norm(n2.pos - n1.pos)
            s = CortexSpring(spring_id=i, node_1=n1, node_2=n2, rest_length=dist, cfg=cfg)

            springs.append(s)

        return springs
    
    def _init_sf(self):
        """
        Initiate a single stress fibre cable along flow axis. 
        Connects most upstream and downstream nodes. 
        """
        # Axial coordinate of every node along the flow axis, measured from centroid. 
        projections = axial_coord(self.positions, self.centroid, self.flow_axis)

        # Most-negative = upstream pole, most-positive = downstream.
        up_id = int(np.argmin(projections))
        dn_id = int(np.argmax(projections))
        up_node, dn_node = self.nodes[up_id], self.nodes[dn_id]

        # Rest length of the fibre is the initial pole-to-pole distance.
        sf_dist = projections.max() - projections.min()

        return StressFibre(node_up=up_node, node_down=dn_node, 
                           rest_length=sf_dist, cfg=self.cfg)
    
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
    def axial_half_extent(self):
        """
        Current half-length of the cell along the flow axis.

        Used to normalise axial node positions into a dimensionless
        [-1, 1] range for soft polarity weighting. 
        Computed from the current (deformed) geometry.
        """
        axial = axial_coord(self.positions, self.centroid, self.flow_axis)
        return float(np.max(np.abs(axial)))
        
    @property
    def polar_nodes(self):
        return [n for n in self. nodes if n.role in ("upstream", "downstream")]

    @property 
    def rhoc_mean(self):
        return float(np.mean([n.P_RhoC for n in self.nodes]))
    
    @property 
    def rhoa_mean(self):
        return float(np.mean([n.P_RhoA for n in self.nodes]))

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def _get_node_springs(self, node_idx):
        s_prev = self.springs[(node_idx - 1) % self.n_nodes]
        s_next = self.springs[node_idx % self.n_nodes]
        return s_prev, s_next
    
    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _compute_area(self):
        """Shoelace formula — polygon area from node positions."""
        pos  = self.positions
        x, y = pos[:, 0], pos[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

    def _compute_node_normal(self, node_idx):
        """Outward unit normal at node, averaged from adjacent edges."""
        n  = self.n_nodes
        prev = self.nodes[(node_idx - 1) % n].pos
        curr = self.nodes[node_idx].pos
        nxt  = self.nodes[(node_idx + 1) % n].pos

        e1 = curr - prev
        e2 = nxt - curr

        # Outward normals — rotate each edge 90° CCW
        n1 = np.array([e1[1],  -e1[0]])
        n2 = np.array([e2[1],  -e2[0]])

        normal = n1 + n2
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-10 else np.zeros(2)
 
    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------
    def _apply_shear_drag(self, flow):
        """
        Apply shear drag as an extensional force concentrated at the poles.
        """
        # p_ref = self.axial_half_extent
        # if p_ref < 1e-10:
        #     return
        
        # Axial coordinate of every node, signed relative to the centroid.
        axial = axial_coord(self.positions, self.centroid, self.flow_axis)
        p_ref = float(np.max(np.abs(axial)))
        if p_ref < 1e-10:
            return
        p = axial / p_ref

        # Quadratic weight: sharp concentration at poles, zero at waist.
        weights = p * p

        # Apply drag per-node along the flow axis, with sign giving direction:
        # upstream nodes (axial < 0) get pushed further upstream,
        # downstream nodes (axial > 0) get pushed further downstream.
        for node, w, a in zip(self.nodes, weights, axial):
            force = flow.drag_on_node(weight=w, axial_sign=np.sign(a))
            #print(node.id, force)
            node.apply_force(force)

        # centroid = self.centroid
        # for node in self.polar_nodes: 
        #     drag_force = drag
        #     axial_sign = np.sign(np.dot(node.pos - centroid, self.flow_axis))
        #     node.apply_force(drag_force * axial_sign * self.flow_axis)
       
    def _update_signalling_load(self, flow):
        """
        Update tensile and total loads for protein recruitment
        """
        stimulus = flow.shear_stimulus()
        for node in self.nodes: 
            node.shear_total = stimulus
            node.tensile_load += stimulus

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
            
            # 6. Apply Force
            normal = self._compute_node_normal(i)
            # Note: In your code, you had -normal; ensure this points OUTWARD
            node.apply_force(pressure * dL  * normal)


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
        for n in self.nodes: 
            n.reset_loads()
            n.reset_force()

        # -- Force Accumulation --
        # 1. Uniform shear on all nodes
        self._apply_shear_drag(flow_field)
        self._update_signalling_load(flow_field)

        # 2. Update Geometry and Tension
        for s in self.springs:
            s.update_geometry_tension()

        self.sf.update_geometry_tension()

        # 3. Accumulate Tensile Loading
        for s in self.springs:
            s.accumulate_loads()

        self.sf.accumulate_loads(self.polar_nodes)

        # 4. Apply mechanical forces
        for s in self.springs:
            s.apply_forces()

        self.sf.apply_forces(self.nodes, self.positions)

        # 8. Soft pressure (area conservation)
        self.current_area = self._compute_area()
        self._apply_pressure()

        # -- Integration --
        # 9. Integrate node positions
        gamma = self.cfg['integration'].get('gamma', 3.0)
        max_disp = self.cfg['integration'].get('max_displacement', 0.5)
        for node in self.nodes:
            node.integrate_step(dt, gamma, max_disp)
        
        # -- Signalling and Remodelling --
        # 10. Node signalling — f_normal already set in step 1
        for node in self.nodes:
            node.update_signalling()

        # 11. Update Cortex and SF properties from rhoa/rhoc
        for s in self.springs:
            s.update_stiffness_and_activation(dt)

        self.sf.update_activation(
            mean_rhoc=self.rhoc_mean, dt=dt
        )

        # 13. Sync area
        self.current_area = self._compute_area()
 
    # ------------------------------------------------------------------
    # Cell State
    # ------------------------------------------------------------------
    def get_state(self):
        shape = measure_shape(self)
        #forces = measure_forces(self)
        polar = [s for s in self.springs if s.side == 'polar']
        flank = [s for s in self.springs if s.side == 'flank']
        return {
            'cell_id': self.id,
            'ar': shape['ar'],
            'orientation': shape['orientation'],
            'area_ratio': shape['area_ratio'],
            'mean_rhoa_pole': safe_mean([n.P_RhoA for n in self.nodes if n.role in ('upstream', 'downstream')]),
            'mean_rhoa_lat': safe_mean([n.P_RhoA for n in self.nodes if n.role == 'lateral']),
            'mean_rhoa': safe_mean([n.P_RhoA for n in self.nodes]),
            'mean_rhoc': safe_mean([n.P_RhoC for n in self.nodes]),
            'a_sf': round(self.sf.a, 3),
            'sf_tension': round(self.sf.T, 3),
            'k_pole': round(np.mean([s.k for s in polar]), 3) if polar else 0,
            'k_flank': round(np.mean([s.k for s in flank]), 3) if flank else 0,
            'tensile_pole': safe_mean([n.tensile_load for n in self.nodes if n.role in ('upstream', 'downstream')]),
            'f_total': safe_mean([n.shear_total for n in self.nodes]),
        }
    
    def get_diagnostics(self):
        """Full force diagnostics — use for analysis, not sweeps."""
        return measure_forces(self)
    
    def __repr__(self):
        s = self.get_state()
        return (
            f"EndothelialCell(id={self.id} | ar={s['ar']:.2f} | "
            f"a_sf={s['a_sf']:.3f} | "
            f"RhoA={s['mean_rhoa']:.3f} RhoC={s['mean_rhoc']:.3f})"
        )