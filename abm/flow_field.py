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
    
    # def get_force_on_node(self, node_pos, cell_centroid):
    #     """
    #     Compute shear force on a sinfle membrane node. 

    #     The flow magnitude scales with how much node faces into flow: 
    #         weight = 0.5 + 0.5 * cos(θ)
    #     where θ is the angle between the radial vector and upstream direction. 

    #     node_pos: position of the node (2D array).
    #     cell_centroid  current centroid of the cell (2D array).
    #     """
        
        # # Compute radial vector (node - centroid)
        # radial = node_pos - cell_centroid
        # r = np.linalg.norm(radial)

        # if r < 1e-10:
        #     # Node at centroid — apply uniform half-weight
        #     return self.direction * self.magnitude * 0.5
        
        # radial_unit = radial / r

        # # Compute angle between radial vector and upstream direction (-flow)
        # # +1 = node faces directly into flow (upstream pole)
        # # -1 = node faces directly away (downstream pole)
        # upstream_angle = np.dot(radial_unit, -self.direction)

        # # Compute differential weighting of magnitude depending on node vector relative to upstream. 
        # # Map [-1, 1] → [0, 1]
        # weight = 0.5 + 0.5 * upstream_angle

        # return self.direction * self.magnitude * weight # force felt by node depends on orientaion

    # ------------------------------------------------------------------
    # Diagnostics & Debugging
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"FlowField(magnitude={self.magnitude:.3f}, "
            f"direction={self.direction.round(3)})"
        )