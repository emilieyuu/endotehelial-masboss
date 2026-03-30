# abm/flow_field.py
#
# Represents a flow field – the mechanical stimulus from fluid shear
# stress acting on the cell membrane. 
# Purely environmental.
#
# Responsibilities: 
#   - Store flow magnitude and direction. 
#   - Computes spatially-varying drag force on a single node. 

import numpy as np

class FlowField(): 
    """
    Uniform laminar shear flow field. 

    Shear stress is a surface force that varies with each node's exposure to flow. 
    Node feels flow differentially – this drives cell deformation. 
    """
    def __init__(self, magnitude=1.0, direction=None): 
        """
        Uniform laminar flow field.

        magnitude: magnitude of shear stress (normalised 0-1).
        direction: unit vector of flow direction, default +x axis. 
        """
        if direction is None: 
            direction = np.array([1.0, 0.0])

        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Flow direction vector cannot be zero.")
        
        self.magnitude = magnitude # Shear stress magnitude
        self.direction = direction / norm # unit vector direction representation

    # ------------------------------------------------------------------
    # Core force computation
    # ------------------------------------------------------------------

    def get_force_on_node(self, node):

        if node.role == 'upstream':
            return -self.direction * self.magnitude
        elif node.role == 'downstream':
            return self.direction * self.magnitude
        else:
            return np.zeros(2)
    


    # ------------------------------------------------------------------
    # Diagnostics & Debugging
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"direction={self.direction.round(3)})"
        )