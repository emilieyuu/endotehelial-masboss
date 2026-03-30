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

    def get_force_on_node(self, node):
        return self.direction * self.magnitude
    

    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"direction={self.direction.round(3)})"
        )