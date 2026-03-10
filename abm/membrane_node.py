# abm/junction_node.py
#
# A MembraneNode is the unit making up the cell membrane.
# Nodes are connected to each other by a Spring – this forms the internal actin cortex. 
# A MembraneNode 

import numpy as np

class MembraneNode():
    """
    A single point on the cell membrane. 
    """

    def __init__(self, node_id: int, position: np.ndarray):
        # ID, position
        self.id = node_id
        self.pos = np.array(position, dtype=float)

        # Geometry: Set by flow field
        self.outward_normal = np.zeros(2)
        self.tangent = np.zeros(2)
        self.face_type = 'lateral' # iniated to lateral, modified dependeing on flow

        # Mechanical inputs from flow
        self.shear = 0.0
        self.normal_force = 0.0
        self.flow_force = np.zeros(2) # combined flow vector for mechanics
        self.tension = 0.0 # spring tension, set by Cell


        # Junction protein states, 0 at init, not yet mechanically activated
        self.DSP, self.TJP1, self.JCAD = 0, 0, 0

        # Rho activity (derived from MaBoSS)
        self.P_RhoA, self.P_RhoC = 0.0, 0.0
     
        # Cytoskeletal state
        self.contractility = 0.0
        self.sf_alignment = 0.0

    def update_state_from_flow(self, shear: float, normal_force:float, outward_normal: np.ndarray, 
                         face_type: str, tangent: np.ndarray, flow_force: np.ndarray):
        """
        Write flow field output onto node. 
        """
        self.shear = shear
        self.normal_force = normal_force
        self.outward_normal = np.array(outward_normal)
        self.face_type = face_type
        self.tangent = np.array(tangent)
        self.flow_force = np.array(flow_force)

    def update_protein_states(self, force_threshold_tensile=0.3, force_threshold_compress=0.1):
        """
        Concert local mechanical state into boolean protein activation 
        """
        #Set parameters based on flow here

    def update_rho_activity(self): 
        # Run MaBoSS directly or call external function to run maboss and get result
        self.P_RhoA = 0.0 # Some updated value
        self.P_RhoC = 0.0 # Some updated value
        pass

    def get_spring_stiffmess(self): 
        """
        RhoA increases cortical tension -> stiffer springs
        """
        # *spring stiffness as a function of P(RhoA)

    def get_orientation(self):
        """
        Rho drives stress fibre alignement to flow
        return bias angle (0 = aligned with flow)
        """ 
        # high RhoC = strong pull towards flow
        # Low RhoC = orientation is free/random
        pass

    def __repr__(self):
        return (f"MembraneNode(id={self.id} | "
                f"pos={self.pos.round(2)} | "
                f"face={self.face_type} | "
                f"shear={self.shear:.3f} | "
                f"normal_force={self.normal_force:.3f} | "
                f"flow_force={self.flow_force.round(3)})")

