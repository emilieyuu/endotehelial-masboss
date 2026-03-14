# abm/endothelial_cell.py
# 
# A closed ring of membrane nodes representing the cell cortex in 2D.
# The orchestrator connecting springs to nodes. 
#

import numpy as np
from abm.membrane_node import MembraneNode
from abm.spring import Spring

class EndothelialCell: 
    """
    Cells are represented as ring of membrane nodes connected by springs. 
    """
    def __init__(self, cell_id: int, centroid: np.ndarray, lut, cfg: dict, 
                 n_nodes: int = 12, radius: float = 10.0):
        
        # Cell General Properties
        self.id = cell_id
        self.n_nodes = n_nodes
        self.cfg = cfg

        # Get Cell meachnical parameters
        mech_cfg = self.cfg['mechanics']
        self.k_area = mech_cfg['k_area']
        self.k_spring = mech_cfg['k_spring']

        # Rest length between adjacent nodes on regular n-gon
        self.rest_length = 2 * radius * np.sin(np.pi / n_nodes) 

        # Initialise node ring and springs
        self.nodes = self._init_node_ring(centroid, n_nodes, radius)
        self.springs = self._init_springs(lut)

        # Measure target area
        self.target_area = self._compute_area()
        self.current_area = self.target_area

        print(f"DEBUG: Initialised Cell {self.id}: ")
        print(self.__repr__())
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_node_ring(self, centroid, n_nodes, radius):
        """
        Place n_nodes evenly around a circle.
        Angles spaced at 2π/n intervals, starting at angle 0 (rightmost point).
        """

        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) # Generate n_nodes evenly spaced numbers between 0 and 2pi radians (circle)   
        nodes = []

        for i, angle in enumerate(angles):
            pos = np.array([
                centroid[0] + radius * np.cos(angle),
                centroid[1] + radius * np.sin(angle),
            ])
            nodes.append(MembraneNode(i, pos))

        return nodes
    
    def _init_springs(self, lut):
        """
        Connect each node to the next (ring topology).
        Node i connects to node (i+1) mod n_nodes — the last node wraps back to node 0.
        Each spring gets a unique id matching its lower-indexed node.
        """
        springs = []

        for i in range(self.n_nodes):
            node1 = self.nodes[i]
            node2 = self.nodes[(i + 1) % self.n_nodes]

            springs.append(Spring(
                spring_id=i, node_1=node1, node_2=node2, 
                rest_length=self.rest_length, lut=lut, 
                cfg=self.cfg, k_base=self.k_spring
            ))

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
    # Internal Geometry
    # ------------------------------------------------------------------
    def _compute_area(self):
        """
        Shoelace formula for the signed area of a polygon.
        Works for any simple polygon — no assumption of convexity.

        Used both to initialise target_area and to measure current area
        each step for the pressure force.
        """

        pos = self.positions
        x, y = pos[:, 0], pos[:, 1]

        # A = ½ |Σ (xᵢ yᵢ₊₁ − xᵢ₊₁ yᵢ)|
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
 
    
    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------
    def apply_spring_forces(self):
        """
        Ask each spring to compute its force and push it onto both nodes.
        Spring.calculate_forces() handles the bilinear law internally.
        """
        for s in self.springs:
            s.calculate_forces()

    def apply_pressure_forces(self):
        """
        Applies area-conservative pressure force.
        Maintains incompressibility of the cytoplasm: 
            - When cell shrink below target, internal pressure pushed node outward. 
            - When cell expands, pressure pushed node inwards. 
        """
        # Calculate current area of cell
        current_area = self._compute_area()

        # Pressure magnitude based on area difference
        pressure = self.k_area * (self.target_area - current_area)

        # Apply force to each node along it outward normal
        for i, node in enumerate(self.nodes):
            prev_node = self.nodes[(i - 1) % self.n_nodes]
            next_node = self.nodes[(i + 1) % self.n_nodes]

            # Vector between neighbours gives the tangent
            tangent  = next_node.pos - prev_node.pos

            outward_normal  = np.array([tangent[1], -tangent[0]]) # outward normal = (dy, -dx)
            normal_unit = outward_normal / (np.linalg.norm(outward_normal) + 1e-10)
        
            node.apply_force(normal_unit * pressure)

    # ------------------------------------------------------------------
    # Timestep
    # ------------------------------------------------------------------
    def step(self, flow_field, dt):
        # Step 1: External Environment (Flow Drag)
        drag_force = flow_field.get_force_on_node()
        for node in self.nodes:
            node.apply_force(drag_force)

        # Step 2: Mechanical Sensing
        # Springs re-measure lengths and alignment to flow
        for spring in self.springs:
            spring.update_geometry(flow_field.direction)

        # Step 3: Signaling & Remodeling
        # Proteins are recruited, LUT is queried, and Kt/L0 adapt over time (tau)
        for spring in self.springs:
            spring.update_signalling()
            spring.update_stiffness(dt)

        # Step 4: Internal Forces
        self.apply_spring_forces()
            
        # Apply area conservation pressure
        self.apply_pressure_forces()

        # Step 5: Integration (Movement)
        # Move nodes based on net force / gamma
        gamma = self.cfg['sim']['gamma']
        for node in self.nodes:
            node.update(dt, gamma)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    def measure_shape(self) -> dict:
        """
        PCA on node positions gives the same shape descriptors as ImageJ.
 
        PCA finds the axes of maximum variance in the point cloud.
        The major axis is the direction nodes are most spread out —
        i.e., the long axis of the cell.
 
        Aspect ratio = major axis length / minor axis length
          = 1.0 for a perfect circle
          > 1.0 for an elongated cell
 
        Orientation = angle of major axis to the x-axis (flow direction)
          = 0° means elongated along flow ✓  (correct elongation response)
          = 90° means elongated perpendicular (incorrect)
 
        Circularity = 4π × area / perimeter²
          = 1.0 for a perfect circle
          → 0 for a very elongated shape
          Same formula as ImageJ's circularity metric.
 
        Area (shoelace) and perimeter (sum of edge lengths) are also returned
        for checking conservation and overall cell size changes.
        """
        pos      = self.positions
        centered = pos - pos.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
        eigvals  = np.maximum(eigvals, 0)
 
        major     = 2.0 * np.sqrt(eigvals[1])
        minor     = 2.0 * np.sqrt(eigvals[0])
        major_vec = eigvecs[:, 1]
 
        x, y  = pos[:, 0], pos[:, 1]
        area  = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perim = np.sum(np.linalg.norm(
            np.diff(np.vstack([pos, pos[0]]), axis=0), axis=1))
 
        return {
            'aspect_ratio': major / (minor + 1e-10),
            'orientation':  np.degrees(np.arctan2(major_vec[1], major_vec[0])),
            'area':         area,
            'perimeter':    perim,
            'circularity':  4.0 * np.pi * area / (perim ** 2 + 1e-10),
        }

    def __repr__(self):
        return (f"id={self.id} | "
                f"n_nodes={self.n_nodes} | "
                f"centroid={self.centroid.round(2)} \n "
                f"target_area={self.target_area:.3f}) | current_area={self.current_area:.3f}")

    