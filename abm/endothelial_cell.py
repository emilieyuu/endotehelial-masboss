# abm/endothelial_cell.py
# 
# A closed ring of membrane nodes representing the cell cortex in 2D.
# Orchestrates springs, nodes, flow, and pressure each timestep.

import numpy as np
from abm.membrane_node import MembraneNode
from abm.spring import Spring

class EndothelialCell: 
    """
    A 2D endothelial cell represented as a closed ring of n_nodes
    membrane nodes connected by springs.

    Mechanical inputs each timestep:
        - Flow field: opposing forces on upstream/downstream pole nodes
                      stretch lateral springs, driving the Rho signal chain
        - Spring forces: cortical tension resists deformation

    Emergent behaviour:
        - RhoA stiffens lateral cortex  → resists compression → failed elongation if dominant
        - RhoC shortens SF rest length  → narrows lateral width → elongation if dominant
        - Shape (AR, orientation) emerges from Rho balance, not imposed externally
    """
    def __init__(self, cell_id: int, centroid: np.ndarray, lut, cfg: dict, 
                 n_nodes: int = 12, radius: float = 10.0, flow_direction=None):
        
        # Cell General Properties
        self.id = cell_id
        self.n_nodes = n_nodes
        self.cfg = cfg
        self.lut = lut

        # Normalised Flow Direction
        flow = np.asarray(flow_direction, dtype=float)
        self.flow_direction = flow / np.linalg.norm(flow)

        # Cell Mechanics 
        mech_cfg = self.cfg['mechanics']
        self.k_cortex = mech_cfg['k_cortex']
        self.rest_length = 2 * radius * np.sin(np.pi / n_nodes) # distance between adjacent nodes on regular n-gon: 2R sin(π/n)

        # Geometry: Initialise Node Ring and Springs
        self.nodes = self._init_node_ring(centroid, n_nodes, radius)
        self._classify_nodes()
        self.springs = self._init_springs(lut)

        # Cell Area 
        self.target_area = self._compute_area() # fixed, acts as reference
        self.current_area = self.target_area # dynamic, remodelled to maintain "incompressible cytoplasm"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_node_ring(self, centroid, n_nodes, radius):
        """
        Place n_nodes evenly around a circle.
        Angles spaced at 2π/n intervals, offset by π/2 so node 0 starts at the top
        """
        # Compute angles between nodes. 
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) + np.pi/2 # offset
        nodes = []

        for i, angle in enumerate(angles):
            pos = np.array([
                centroid[0] + radius * np.cos(angle),
                centroid[1] + radius * np.sin(angle),
            ])
            nodes.append(MembraneNode(i, pos))

        return nodes

    
    def _classify_nodes(self):
        """
        Label each node upstream, downstream, or lateral based on its
        radial projection onto the flow axis.
        """
        threshold =  0.9 #float(self.cfg['mechanics'].get('polar_threshold', 0.85))
        centroid = self.centroid

        for node in self.nodes:
            radial = node.pos - centroid
            radial_unit = radial / np.linalg.norm(radial)
            projection = np.dot(radial_unit, self.flow_direction)
            #print(projection)

            if projection > threshold:
                node.role = 'downstream'
            elif projection < -threshold:
                node.role = 'upstream'
            else:
                node.role = 'lateral'
    
    def _init_springs(self, lut):
        """
        Connect adjacent nodes in a ring.
        Store _init_alignment on each spring so population splits
        remain valid at high AR when current alignment rotates.
        """
        springs = []

        for i in range(self.n_nodes):
            # Node i connects to node (i+1) mod n_nodes — last node wraps back to node 0.
            node1 = self.nodes[i]
            node2 = self.nodes[(i + 1) % self.n_nodes]

            s = Spring(
                spring_id=i, node_1=node1, node_2=node2, # id matches its lower-indexed node.
                rest_length=self.rest_length, k_cortex = self.k_cortex,
                lut=lut, cfg=self.cfg
            )

            # Compute and store initial alignment for population classification
            diff = node2.pos - node1.pos
            norm = np.linalg.norm(diff)
            s._init_alignment = (abs(np.dot(diff / norm, self.flow_direction)) 
                                if norm > 1e-10 else 0.0)

            springs.append(s)

        return springs
        
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        return np.array([n.pos for n in self.nodes])
    
    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)
    
    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _compute_area(self):
        """
        Shoelace formula for the signed area of a polygon.
        A = ½ |Σ (xᵢ yᵢ₊₁ − xᵢ₊₁ yᵢ)|

        Works for any simple polygon — no assumption of convexity.
        """
        pos = self.positions
        x, y = pos[:, 0], pos[:, 1]

        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
    
    def _apply_pressure(self):
        """
        Soft area conservation — prevents cell collapse under SF shortening.
        Does not drive elongation — that comes from pole forces and Rho remodelling.
        k_area should be small relative to k_cortex.
        """
        pressure = self.cfg['mechanics'].get('k_area', 0.1) * \
                (self.target_area - self.current_area)
        centroid = self.centroid
        for node in self.nodes:
            outward = node.pos - centroid
            norm    = np.linalg.norm(outward)
            if norm < 1e-10:
                continue
            node.apply_force((outward / norm) * pressure)


    # ------------------------------------------------------------------
    # Timestep
    # ------------------------------------------------------------------
    def step(self, flow_field, dt):
        """
        Advance the cell by one mechanical time step. 

        1. External forces applied to nodes (flow field)
        2. Spring geometry updated from current node positions
        3. Spring forces applied to nodes
        4. Node positions integrated
        5. Signalling updated (reads tension from step 2)
        6. Remodelling updated (reads Rho from step 5)
        """
        # Step 1: Flow-driven pole forces — stretch lateral springs
        for node in self.nodes:
            shear_force = flow_field.get_force_on_node(node)
            node.apply_force(shear_force)

        self._apply_pressure()

        # Step 2: Spring geometry — recompute length, alignment, tension
        for spring in self.springs:
            spring.update_geometry(flow_field.direction)
        
        # Step 3: Spring forces — transmit tension to nodes
        for spring in self.springs:
            spring.apply_forces()

        # Step 4: Integrate — overdamped dynamics dx = (F/gamma) dt
        gamma = self.cfg['sim']['gamma']
        max_disp = self.rest_length * 0.1
        for node in self.nodes:
            node.update(dt, gamma, max_disp)

        # Step 5: Signalling — tension → proteins → Rho
        for spring in self.springs:
            spring.update_signalling()

        # Step 6: Remodelling — Rho → k_active and L_sf
        for spring in self.springs:
            spring.update_remodelling(dt)

        # Sync Area
        self.current_area = self._compute_area()

    # ------------------------------------------------------------------
    # Diagnostics & Debuggings
    # ------------------------------------------------------------------

    def _spring_populations(self):
        """
        Split springs into lateral (parallel to flow) and polar
        (perpendicular to flow) using initial alignment.
        Uses same threshold as _classify_nodes for consistency.
        """
        threshold = float(self.cfg['mechanics'].get('polar_threshold', 0.9))
        lateral   = [s for s in self.springs if s._init_alignment >  threshold]
        polar     = [s for s in self.springs if s._init_alignment <= threshold]
        return lateral, polar
    
    def measure_shape(self) -> dict:
        """
        PCA-based shape descriptors.
        Returns aspect ratio, orientation, and circularity.
        """
        pos      = self.positions
        centered = pos - pos.mean(axis=0)

        eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
        eigvals  = np.maximum(eigvals, 0.0)

        major_vec   = eigvecs[:, 1]
        major       = 2.0 * np.sqrt(eigvals[1])
        minor       = 2.0 * np.sqrt(eigvals[0])
        ar          = major / (minor + 1e-10)
        orientation = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

        perim = np.sum(np.linalg.norm(
            np.diff(np.vstack([pos, pos[0]]), axis=0), axis=1
        ))

        return {
            'ar':          round(ar, 3),
            'orientation': round(orientation, 2),
            'circularity': round(4.0 * np.pi * self.current_area /
                                 (perim ** 2 + 1e-10), 3),
        }

    def get_state(self) -> dict:
        """
        State snapshot for monitoring and phenotype classification.

        metrics:     shape readouts (ar, orientation, area_err)
        signalling:  Rho balance and spatial distribution
        mechanics:   tension by subsystem
        remodelling: k_active and L_sf state
        """
        shape            = self.measure_shape()
        lateral, polar   = self._spring_populations()

        # Signalling — spatial Rho distribution
        rho_balance  = float(np.mean([s.P_RhoC - s.P_RhoA for s in self.springs]))
        lateral_rhoc = float(np.mean([s.P_RhoC for s in lateral])) if lateral else 0.0
        polar_rhoa   = float(np.mean([s.P_RhoA for s in polar]))   if polar   else 0.0

        # Mechanics — tension by subsystem
        t_cortex = float(np.mean([s.tension_cortex for s in self.springs]))
        t_sf     = float(np.mean([s.tension_sf     for s in self.springs]))

        # Remodelling — mean state across all springs
        mean_k_active  = float(np.mean([s.k_active for s in self.springs]))
        mean_lsf_ratio = (float(np.mean([s.L_sf / s.L_cortex for s in lateral]))
                          if lateral else 1.0)

        return {
            'cell_id': self.id,
            'metrics': {
                'ar':          shape['ar'],
                'orientation': shape['orientation'],
                'circularity': shape['circularity'],
                'area_err':    round(self.current_area / self.target_area, 4),
            },
            'signalling': {
                'rho_balance':  round(rho_balance,  3),
                'lateral_rhoc': round(lateral_rhoc, 3),
                'polar_rhoa':   round(polar_rhoa,   3),
            },
            'mechanics': {
                't_cortex': round(t_cortex, 4),
                't_sf':     round(t_sf,     4),
            },
            'remodelling': {
                'mean_k_active':  round(mean_k_active,  4),
                'mean_lsf_ratio': round(mean_lsf_ratio, 4),
            },
        }

    def __repr__(self):
        s = self.get_state()
        return (
            f"EndothelialCell(id={self.id} | "
            f"n={self.n_nodes} | "
            f"centroid={self.centroid.round(2)} | "
            f"ar={s['metrics']['ar']:.2f} | "
            f"area_err={s['metrics']['area_err']:.3f} | "
            f"rho_bal={s['signalling']['rho_balance']:+.3f} | "
            f"lsf={s['remodelling']['mean_lsf_ratio']:.3f} | "
            f"t_sf={s['mechanics']['t_sf']:.4f})")