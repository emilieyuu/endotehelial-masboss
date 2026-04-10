# abm/flow_field.py
#
# Environmental flow field. 
# Provides shear magnitude and drag force parameters to the cell.
#
# Responsibilities:
#   1. Mechanical: extensional drag at polar nodes (drag_on_node)
#   2. Signalling: uniform shear stimulus at every node (magnitude)

import numpy as np
from src.utils import require

class FlowField:
    """External flow acting on membrane nodes."""
    def __init__(self, cfg):
        flow_cfg = cfg['flow']

        # Magnitude clamped to non-negative 
        self.magnitude = max(require(flow_cfg, 'magnitude'), 0.0)

        # Drag force magnitude
        drag_frac = require(flow_cfg, 'drag_fraction')
        self.drag = self.magnitude * drag_frac

        # Normalised flow direction
        direction = np.asarray(require(flow_cfg, 'direction'), dtype=float)
        norm = np.linalg.norm(direction)

        if norm < 1e-10:
            raise ValueError("Flow direction cannot be zero vector.")
        
        self.direction = direction / norm
    
    def drag_on_node(self, weight, axial_sign):
        """
        Drag force on a single node.

        weight: polarity weight in [0, 1] 
        axial_sign: ±1, tells us pole of node 

        Returns: 2D force vector along ±direction.
        """
        return weight * self.drag * axial_sign * self.direction

    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"drag={self.drag:.3f}, direction={self.direction.round(3)})"
        )