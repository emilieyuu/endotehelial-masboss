# abm/flow_field.py
#
# Environmental flow field acting as external stimuli for cellular responses. 

import numpy as np

class FlowField:
    """
    External flow acting on membrane nodes and junctions. 
    1. Drag
    2. Tensile loading
    """
    def __init__(self, magnitude, drag_frac, direction=np.array([1.0, 0.0])):
        """
        magnitude: magnitude of flow (non-dimensional)
        direction: unit-vector representationg of flow direction
        """
        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)

        if norm < 1e-10:
            raise ValueError("Flow direction cannot be zero.")

        self.magnitude = magnitude
        self.drag = magnitude * drag_frac
        self.direction = direction / norm
        print(f">>> INFO: Initiated flow field with magnitude {self.magnitude} and unit direction {self.direction}")

    
    def drag_on_node(self, weight, axial_sign):
        """
        Compute drag force on a single node given polarity weight and axial sign. 
        Flow provides magnitude and direction
        """
        return weight * self.drag * axial_sign * self.direction
    
    def shear_stimulus(self):
        """
        Scalar shear magnitude felt uniformly by any node under flow. 
        """
        return max(self.magnitude, 0.0)

    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"direction={self.direction.round(3)})"
        )