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
    def __init__(self, cell_id: int, centroid: np.ndarray, n_nodes: int = 12, radius = 10.0):
        
        self.id = cell_id
        self.n_nodes = n_nodes

        # Global cell properties
        self.centroid = centroid # [X, Y] coordinates represnting centre of cell
        self.target_area = np.pi * radius**2 # area constraint (incompressible cytoplasm)
         
        self.positions, self.nodes = self._init_node_ring(centroid, n_nodes, radius)

        # Initial resting length of springs between nodes
        self.rest_length = 2 * radius * np.sin(np.pi / n_nodes)
    

    def _init_node_ring(self, centroid, n_nodes, radius):
        # Global cell properties
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) # Generate n_nodes evenly spaced numbers between 0 and 2pi radians (circle)   

        # Create nodes evenly spaces around circle         
        positions = np.column_stack([
            centroid[0] + radius * np.cos(angles), # Converts angles to coordinates using parametric equation of circle
            centroid[1] + radius * np.sin(angles) # np.column_stack: joins x and y coordinates into pairs (cols of x, y coordinates)
        ])
        
        # Initiate n nodes using calculated positions
        nodes = [MembraneNode(i, pos) for i, pos in enumerate(positions)] 

        return positions, nodes

    def get_pairs(self):
        """ Return adjacent node pairs. """
        return [(self.nodes[i], self.nodes[(i+1) % self.n_nodes]) for i in range(self.n_nodes)]

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