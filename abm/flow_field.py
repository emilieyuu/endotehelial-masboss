# abm/flow_field.py
#
# A shear stress field that assigns local mechanical input to each membrane node, 
# differentiated by the node's geometric relationship to flow. 
#

import numpy as np

class FlowField(): 

    def __init__(self, shear_rate=1.0, direction=np.array([1.0, 0.0])): 
        self.shear_rate = shear_rate # Shear stress magnitude
        self.direction = direction # Flow direction

    def classify_faces(self, positions):
        """
        Classify each node as upstream/lateral/downstream based on 
        normal alignment with flow. 
        """

        centre = positions.mean(axis=0)
        relative = positions - centre # get vectors from centre to each node

        # Normal vectors
        norms = np.linalg.norm(relative, axis=1, keepdims=True) # compute vector length
        outward_normals = relative / (norms + 1e-10) # divide by norm to produce unit vectors 
        
        # Alignment (-1 = Upstream, 1 = Downstream)
        alignment = outward_normals @ self.direction # dot product

        # Classify 
        face_types = np.where(alignment < -0.5, 'upstream', 
                     np.where(alignment > 0.5, 'downstream'), 
                     'lateral')
        
        return face_types, outward_normals
    
    def get_shear_at_node(self, outward_normal):
        # get shear force field instead?
        """ 
        Calculates shear magnitude
        Shear force tangentally along membrane surface. 
        """

        tangent = np.array([-outward_normal[1], outward_normal[0]]) # compute tangent direction
                                                                    # Rotates normal 90 degrees
        alignment = np.dot(self.direction, tangent) # measure alignment with flow 

        return self.shear_rate * abs(alignment) # shear magnitude