# abm/junction_node.py
#
# A MembraneNode is the unit making up the cell membrane.
# Adjacent nodes are connected to each other by a Spring.
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
        self.force = np.zeros(2)

    def apply_force(self, force):
        """
        All forces applied to node (flow + springs)
        """
        self.force += force

    def update(self, dt=0.1, gamma=1.0):
        """
        Overdampeddynamics update: dx = (F / gamma) * dt
        gamma: viscous drag coefficient (thickness of envrionment/damping)
        dt: mechanical timestep
        """

        displacement = (self.force / gamma) * dt

        self.pos += displacement
        self.force = np.zeros(2) # reset for next time step
                                 # forces in the model are "instantaneous"

        # # Geometry: Set by flow field
        # self.outward_normal = np.zeros(2)
        # self.tangent = np.zeros(2)
        # self.face_type = 'lateral' # iniated to lateral, modified dependeing on flow

        # # Mechanical inputs from flow
        # self.shear = 0.0
        # self.normal_force = 0.0
        # self.flow_force = np.zeros(2) # combined flow vector for mechanics
     
        # # Cytoskeletal state
        # self.contractility = 0.0
        # self.sf_alignment = 0.0

    # def update_state_from_flow(self, shear: float, normal_force:float, outward_normal: np.ndarray, 
    #                      face_type: str, tangent: np.ndarray, flow_force: np.ndarray):
    #     """
    #     Write flow field output onto node. 
    #     """
    #     self.shear = shear
    #     self.normal_force = normal_force
    #     self.outward_normal = np.array(outward_normal)
    #     self.face_type = face_type
    #     self.tangent = np.array(tangent)
    #     self.flow_force = np.array(flow_force)

    # def update_cytoskeleton(self, P_RhoA: float, P_RhoC: float,
    #                          dt: float,
    #                          tau_contractility: float = 5.0,
    #                          tau_sf: float = 20.0):
    #     """
    #     Cytoskeletal state relaxes toward Rho-determined targets.

    #     P_RhoA, P_RhoC : averaged from this node's two adjacent springs
    #     dt             : timestep size

    #     First-order kinetics:
    #         contractility: fast  (tau~5)  — ROCK/MLC phosphorylation
    #         sf_alignment:  slow  (tau~20) — mDia2/actin polymerisation

    #     Timescale difference produces the observed behaviour:
    #     cells first change tension (fast) then gradually align stress
    #     fibres (slow) — consistent with elongation over 24h (Noria 2004).
    #     """
    #     self.contractility += dt * (P_RhoA - self.contractility) \
    #                           / tau_contractility
    #     self.sf_alignment  += dt * (P_RhoC - self.sf_alignment) \
    #                           / tau_sf

    def __repr__(self):
        return (f"MembraneNode(id={self.id} | "
                f"pos={self.pos.round(2)} | "
                f"force={self.force}")

    # def __str__(self):
    #     return (f"MembraneNode(id={self.id} | "
    #             f"pos={self.pos.round(2)} | "
    #             f"normal={self.outward_normal.round(3)} | "
    #             f"face={self.face_type} | "
    #             f"flow_force={self.flow_force.round(3)} | "
    #             f"contractility={self.contractility:.3f} | "
    #             f"sf_alignment={self.sf_alignment:.3f})")
    
  


