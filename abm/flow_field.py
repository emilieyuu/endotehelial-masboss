# abm/flow_field.py
#
# A shear stress field that assigns local mechanical input to each membrane node, 
# differentiated by the node's geometric relationship to flow. 
#

import numpy as np

def get_mechanical_input(shear: float, normal_force: float, protein: str) -> float:
        """
        Select the relevant force component for each junction protein

        DSP:  tangential shear    → lateral tensile loading
        TJP1: normal force        → upstream pressure loading  
        JCAD: combined magnitude  → overall junction load
        """
        if protein == "DSP":
            return abs(shear)
        elif protein == "TJP1":
            return abs(normal_force)
        elif protein == "JCAD":
            # Overall junction tension — magnitude of total force vector
            return np.sqrt(shear**2 + normal_force**2) / np.sqrt(2)
        else: 
             raise ValueError(f"Unknown protein: {protein}")

class FlowField(): 

    def __init__(self, shear_stress: float = 1.0, direction: np.ndarray = np.array([1.0, 0.0])): 
        """
        Uniform laminar flow field.

        shear_stress: magnitude of shear (normalised 0-1 for now)
        direction:    unit vector of flow direction
        """
        self.shear_stress = shear_stress # Shear stress magnitude
        self.direction = direction / np.linalg.norm(direction)

    def compute_normals(self, positions: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """
        Compute outward unit normal for each node positions. 

        positions: (N,2) array of node positions
        centre:    (2,)  cell centroid
        Returns:   (N,2) array of unit normals
        """

        relative = positions - centre # get vectors from centre to each node
        norms = np.linalg.norm(relative, axis=1, keepdims=True) # compute vector length
        return relative / (norms + 1e-10) # divide by norm to produce unit vectors 
    
    def classify_faces(self, outward_normals: np.ndarray) -> np.ndarray:
        """
        Classify each node as upstream/lateral/downstream.

        Upstream:   normal aligned WITH flow    dot > +0.5
        Downstream: normal aligned AGAINST flow dot < -0.5
        Lateral:    perpendicular               dot in [-0.5, +0.5]

        Returns: string array of face labels.
        """
        alignment = outward_normals @ self.direction # dot product

        return np.where(alignment > 0.5, 'upstream', 
                np.where(alignment < -0.5, 'downstream',
                                            'lateral'))
    def compute_node_forces(self, outward_normal: np.ndarray):
        """
        Compute shear, normal force, tangent, and flow force for one node. 
         outward_normal: (2,) unit vector pointing away from cell centre

        Returns:
            shear_magnitude: tangential component of flow force
            normal_force:    pressure component of flow force
            tangent:         (2,) tangent vector at this node
            flow_force:      (2,) total flow force vector on node
        """
        # Tangent: rotate normal 90 degrees
        tangent = np.array([-outward_normal[1], outward_normal[0]]) 

        # Shear: how much flow acts tangtially at this node                                                            
        shear_magnitude = abs(np.dot(self.direction, tangent)) * self.shear_stress

        # Normal force: how much flow pushes/pulls normally at this node. 
        # Positive = compression (upstream), Negative = suction (downstream)
        normal_force = np.dot(self.direction, outward_normal) * self.shear_stress

        # Combined flow force vector
        flow_force = (shear_magnitude * tangent) + (normal_force * outward_normal)

        return shear_magnitude, normal_force, tangent, flow_force
    
    
   