# abm/flow_field.py
#
# Represents a flow field. 
# Purely environmental.
#

import numpy as np


class FlowField(): 

    def __init__(self, magnitude=1.0, direction=np.array([1.0, 0.0])): 
        """
        Uniform laminar flow field.

        magnitude: magnitude of shear (normalised 0-1 for now)
        direction: unit vector of flow direction
        """
        self.magnitude = magnitude # Shear stress magnitude
        self.direction = direction / np.linalg.norm(direction) # unit vector direction representation

    def get_force_on_node(self): 
        """
        Each node feels drag in flow direction. 
        """
        return self.direction * self.magnitude
