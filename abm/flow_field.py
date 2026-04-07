# abm/flow_field.py
#
# Environmental flow field acting as external stimuli for cellular responses. 

import numpy as np
from abm.geometry import axis_alignment

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

    def get_drag_force(self):
        return self.drag
    
    def get_signalling_forces(self, node_normal):
        """
        Calculates tensile and tangential and total components at a point of the membrane. 
        
        f_normal: tensile component
        f_tanegtial: sliding component
        f_total: weighted magnitude
            Junction proteins respon more strongly to tensile than tangential loading
        """
        alignment = axis_alignment(node_normal, self.direction)
        f_normal = self.magnitude * abs(alignment) # tensile magnitude
        f_mag = self.magnitude # uniform shear magnitude felt by all nodes
        
        return f_normal, f_mag

    def get_force_on_node(self, node):
        return self.direction * self.magnitude
    

    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"direction={self.direction.round(3)})"
        )