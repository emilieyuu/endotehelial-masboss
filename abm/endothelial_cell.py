# abm/endothelial_cell.py
# 
# A closed ring of membrane nodes representing the cell cortex in 2D
#

import numpy as np
from abm.membrane_node import MembraneNode

class EndothelialCell: 
    """
    Cells are represented as ring of membrane nodes connected by springs. 
    """
    def __init__(self, cell_id: int, centroid: np.ndarray, 
                 n_nodes: int = 12, radius: float = 10.0):
        
        self.id = cell_id
        self.n_nodes = n_nodes

        # Area constraint — incompressible cytoplasm
        self.target_area = np.pi * radius**2 # area constraint (incompressible cytoplasm)
         
        # Initial resting length of springs between nodes
        self.rest_length = 2 * radius * np.sin(np.pi / n_nodes)

        # Initae node ring
        self.nodes = self._init_node_ring(centroid, n_nodes, radius)
    

    def _init_node_ring(self, centroid, n_nodes, radius):
        # Global cell properties
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) # Generate n_nodes evenly spaced numbers between 0 and 2pi radians (circle)   
        nodes = []

        for i, angle in enumerate(angles):
            pos = np.array([
                centroid[0] + radius * np.cos(angle),
                centroid[1] + radius * np.sin(angle),
            ])
            nodes.append(MembraneNode(i, pos))

        return nodes
    
    # ----
    # Positions
    # ----
    @property
    def positions(self) -> np.ndarray:
        return np.array([n.pos for n in self.nodes])
    
    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)
    
    # ----
    # Topology
    # ----

    def get_pairs(self):
        """ Return adjacent node pairs. """
        return [(self.nodes[i], self.nodes[(i+1) % self.n_nodes]) for i in range(self.n_nodes)]
    
    # ----
    # Shape measurement
    # ----

    def measure_shape(self) -> dict:
        """
        PCA on node positions — same metrics as ImageJ/FIJI output.
        Aspect ratio, orientation, area, perimeter, circularity.
        """
        positions = self.positions
        centered  = positions - self.centroid
        cov       = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)

        major = 2 * np.sqrt(eigvals[1])
        minor = 2 * np.sqrt(eigvals[0])
        major_vec = eigvecs[:, 1]

        x, y = positions[:,0], positions[:,1]
        area = 0.5 * abs(
            np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))
        )
        perim = np.sum(np.linalg.norm(
            np.diff(np.vstack([positions, positions[0]]), axis=0),
            axis=1
        ))
        circularity = 4 * np.pi * area / (perim**2 + 1e-10)

        return {
            'aspect_ratio':   major / (minor + 1e-10),
            'orientation':    np.degrees(
                                np.arctan2(major_vec[1], major_vec[0])),
            'area':           area,
            'perimeter':      perim,
            'circularity':    circularity,
        }

    def __repr__(self):
        c = self.centroid.round(2)
        return (f"EndothelialCell(id={self.id} | "
                f"n_nodes={self.n_nodes} | "
                f"centroid={c} | "
                f"rest_length={self.rest_length:.3f})")

    def compute_spring_forces(self):
        """ 
        Compute spring forces on each node from its 2 neighbours, 
        modulated by local RhoA actvity
        """

    def compute_pressure_forces(self):
        pass

    def step_mechanics():
        pass

    def measure_shape(self):
        """
        Compute aspect ratio and orientation angle of the cell. 
        """