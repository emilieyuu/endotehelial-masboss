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

        # Create nodes evenly spaces around circle
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            # Generate n_nodes evenly spaced numbers between 0 and 2pi radians (circle)
            
        self.positions = np.column_stack([
            centroid[0] + radius * np.cos(angles),
            centroid[1] + radius * np.sin(angles)
        ])
            # Converts angles to coordinates using parametric equation of circle
            # np.column_stack: joins x and y coordinates into pairs (cols of x, y coordinates)

        # Initial resting length of springs between nodes
        self.rest_length = 2 * radius * np.sin(np.pi / n_nodes)
    

    def get_pairs(self):
        """ Return adjacent node pairs. """
        pass

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