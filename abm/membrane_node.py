# abm/junction_node.py
#
# A MembraneNode is the unit making up the cell membrane.
# Nodes are connected to each other by a Spring – this forms the internal actin cortex. 
# A MembraneNode 

import numpy as np

class MembraneNode():
    """
    A point on the cell membrane. 

    Owns: 
        - position in 2D space
        - junction proteins states
        - RhoA/RhoC activity levels
        - mechanical state (local force)
    """

    def __init__(self, node_id: int, position: np.ndarray):
        # ID, position
        self.id = node_id
        self.pos = position # x and y coordinates of memagent

        # Junction protein states, initialised to WT Perb
        self.DSP = 1
        self.TJP1 = 1
        self.JCAD = 1

        # Rho activity (derived from MaBoSS)
        self.P_RhoA = 0.0
        self.P_RhoC = 0.0

        # LocalMechnical state
        self.local_force = np.ndarray(2) # F = k(L - L0), (cortical tension)
        self.tension = 0.0 # tension in adjacent springs

        # flow-related mechanics
        self.local_shear = 0.0 # local shear from flow
        self.face_type = 'lateral' # change to normal vector instead?

        # Cytoskeletal state
        # self.contractility
        # self.sf_alignment

    

    def update_protein_states(self):
        """
        Concert local mechanical state into boolean protein activation 
        (shear stress -> kinetic params). 
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