# abm_v2/flow_field.py

import numpy as np

class FlowField:

    def __init__(self, magnitude=1.0, direction=None):
        if direction is None:
            direction = np.array([1.0, 0.0])

        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise ValueError("Flow direction cannot be zero.")

        self.magnitude  = magnitude
        self.direction  = direction / norm

    def get_signalling_forces(self, node_normal):
        """
        Calculates tensile and tangential and total components at a point of the membrane. 
        
        f_normal: tensile component
        f_tanegtial: sliding component
        f_total: weighted magnitude
            Junction proteins respon more strongly to tensile than tangential loading
        """
        cos_theta = np.dot(self.direction, node_normal)
        f_n = self.magnitude * abs(cos_theta) # tensile magnitude
        f_t = self.magnitude * np.sqrt(1 - cos_theta**2) # tangential magnitude
        
        # Weighted total magnitude
        tangential_weight = 0.5
        f_total = np.sqrt(f_n**2 + tangential_weight * f_t**2)

        return f_n

    def get_force_on_node(self, node):
        return self.direction * self.magnitude
    

    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"direction={self.direction.round(3)})"
        )