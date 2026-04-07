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
from abm.spring import Spring
from abm.stress_fibre import StressFibre
from abm.analysis.cell_measurement import measure_forces, measure_shape
from abm.geometry import get_pole_indices, length_along_axis, axial_weight, axial_sign, axis_alignment
from src.utils import safe_mean

class EndothelialCell:
    def __init__(self, cell_id, centroid, lut, cfg,
                 n_nodes=16, radius=12.0, flow_direction=np.array([1.0, 0.0])):

        self.id = cell_id
        self.n_nodes = n_nodes
        self.cfg = cfg
        self.lut = lut

        self.flow_direction = flow_direction 

        # Build geometry
        self.nodes = self._init_node_ring(centroid, n_nodes, radius, lut, cfg)
        self._classify_nodes()
        self.springs = self._init_springs(n_nodes, cfg)
        self.stress_fibre = self._init_sf()

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
            radial = node.pos - centroid
            radial_unit = radial / np.linalg.norm(radial)
            projection = np.dot(radial_unit, self.flow_direction)

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

            # Get side from role of connecting nodes 
            if n1.role in ('upstream', 'downstream') or n2.role in ('upstream', 'downstream'):
                side = 'polar'
            else: 
                side = 'flank'

            s = Spring(spring_id=i, node_1=n1, node_2=n2, rest_length=dist, side=side, cfg=cfg)

            springs.append(s)

        return springs
    
    def _init_sf(self):
        """
        Initiate a single stress fibre cable along flow axis. 
        Connects most upstream and downstream nodes. 
        """
        # Get nodes connecting stress fibre, and compute rest length
        up_node, dn_node = self._get_pole_nodes()
        sf_dist = length_along_axis(self.positions, self.flow_direction)

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
    
    def _get_pole_nodes(self):
        """ Get indices of all polar nodes. """
        i_up, i_dn = get_pole_indices(self.positions, self.flow_direction)
        return self.nodes[i_up], self.nodes[i_dn]
    
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
    
    def _compute_node_projections(self, node_idx):
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
    def _apply_shear(self, flow_field):
        """
        Shear stress effects:
        
        Signalling: compute f_normal and f_total at each node for
                    junction protein recruitment.
        
        Mechanical: extensional drag at pole nodes.
                    Base drag from fluid shear + amplification from SF tension.
                    SF tension amplification represents neighbour cell pulling
                    through junctions — stronger SF → neighbours resist more
                    → more outward force at poles.
        """
        centroid = self.centroid

        x_coords = [n.pos[0] for n in self.nodes]
        max_dx = (np.max(x_coords) - np.min(x_coords)) / 2

        for i, node in enumerate(self.nodes):
            self._update_signalling_load(flow_field, node, i)

            if node.role in ('upstream', 'downstream'):
                dx = node.pos[0] - centroid[0]
                # Quadratic weight: 1.0 at absolute tips, dropping quickly toward waist
                weight_axial = (abs(dx) / max_dx)**2
                
                # Forces: Base fluid drag + Reciprocal neighbor pull (eta * T_sf)
                #sf_ext_pull = self.stress_fibre.t_sf 
                base_drag = flow_field.drag
                f_ext_total = base_drag * weight_axial
                node.tensile_load += f_ext_total
                
                # Pull outward from centroid along flow direction
                axial_sign = np.sign(np.dot(node.pos - centroid, self.flow_direction))
                node.apply_force(f_ext_total * axial_sign * self.flow_direction)
    
    def _update_signalling_load(self, flow_field, node, idx):
        """
        Update tensile and total loads for protein recruitment
        """
        # Get relative node norma
        normal = self._compute_node_normal(idx)

        # Get tensile shear component and total magnitude
        fnorm, fmag = flow_field.get_signalling_forces(normal)

        # Get cortex tension at a node based on tension from neighbouring springs
        s_prev, s_next = self._get_node_springs(idx)
        cortex_tension = s_prev.t_cortex + s_next.t_cortex

        # Compute sf tension felt at a node based on alignment
        alignment = axis_alignment(normal, self.flow_direction) # change to alignment to sf unit vec??
        sf_tension = self.stress_fibre.t_sf * 2 * alignment

        # Compute tensile and total load
        node.f_tensile_load = sf_tension + cortex_tension + max(fmag, 0.0) 
        node.shear_total = max(fmag, 0.0)

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
            # 2. Get the two springs connected to this node
            # (Assuming springs[i] connects node[i] and node[i+1])
            s_prev = self.springs[(i - 1) % n]
            s_next = self.springs[i % n]
            
            # 3. Calculate mean local stiffness
            avg_k = 0.5 * (s_prev.k_active + s_next.k_active)
            
            # 4. Compliance Weighting
            # We normalize by k_cortex (basal) so WT is roughly 1.0
            # Lower stiffness (lower RhoA) -> Higher weight -> More outward push
            compliance_weight = (self.cfg['mechanics']['k_cortex'] / avg_k) ** 2

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
        self._apply_shear(flow_field)

        # 2. Cortical spring geometry and forces
        for s in self.springs:
            s.update_cortex_geometry_tension(self.flow_direction)

        for s in self.springs:
            s.accumulate_cortex_loads()

        for s in self.springs:
            s.apply_cortex_forces()

        # 3. SF geometry and and forces
        self.stress_fibre.update_sf_geometry_tension()
        #self.stress_fibre.accumulate_sf_loads(self.positions, self.nodes, self.centroid)
        self.stress_fibre.apply_sf_forces(self.nodes, self.positions)

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

        for s in self.springs:
            s.update_cortex_stiffness()

        self.stress_fibre.update_sf_activation(
            mean_rhoc=self.rhoc_mean, dt=dt
        )

        # 13. Sync area
        self.current_area = self._compute_area()
 
    # ------------------------------------------------------------------
    # Cell State
    # ------------------------------------------------------------------
    def get_state(self):
        shape = measure_shape(self)
        forces = measure_forces(self)
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
            'a_sf': round(self.stress_fibre.a_sf, 3),
            'sf_tension': round(self.stress_fibre.t_sf, 3),
            'k_active_pole': round(np.mean([s.k_active for s in polar]), 3) if polar else 0,
            'k_active_flank': round(np.mean([s.k_active for s in flank]), 3) if flank else 0,
            'f_tensile_pole': safe_mean([n.f_tensile_load for n in self.nodes if n.role in ('upstream', 'downstream')]),
            'f_total': safe_mean([n.f_total_load for n in self.nodes]),
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