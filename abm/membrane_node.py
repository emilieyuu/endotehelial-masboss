# abm/membrane_node.py
#
# A MembraneNode is the unit making up the cell membrane.
# Adjacent nodes are connected to each other by a Spring.

import numpy as np

class MembraneNode():
    """
    A single point on the cell membrane. 
    Dynamics are overdamped (no inertia): dx = (F / gamma) * dt. 
    """

    def __init__(self, node_id: int, position: np.ndarray):
        self.id = node_id
        self.pos = np.array(position, dtype=float)
        self.force = np.zeros(2)
        self.role = 'lateral'

    def apply_force(self, force):
        """
        Accumulate force (shear, springs or pressure) contribution to this node.  
        Forces are instantaneous, reset to zero after update. 
        """
        force = np.asarray(force)
        if not np.all(np.isfinite(force)):
            raise ValueError(f"Node {self.id} received non-finite force: {force}")
        
        self.force += force

    def update(self, dt: float, gamma: float, max_displacement: float = 1.0):
        """
        Integrate position forward one timestep. 

        dt: mechanical timestep. 
        gamma: viscous drag coefficient – higher = slower movement. 
        max_displacement: soft cap on movement per step. 
            Prevents nodes teleporting if forces spike during exploration. 
            Should be ≈ 10% of rest length, pass dynamically. 
        """
        displacement = (self.force / gamma) * dt
        #print(f">>> DEBUG: Node({self.id}): Calculated displacement: {displacement}")

        # Soft cap — preserves direction, limits magnitude
        d_norm = np.linalg.norm(displacement)
        if d_norm > max_displacement:
            displacement = displacement / d_norm * max_displacement

        self.pos += displacement # update node position
        self.force = np.zeros(2) # reste for next time step

    # ------------------------------------------------------------------
    # Diagnostics & Debugging
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"MembraneNode(id={self.id} | "
            f"pos={self.pos.round(2)} | "
            f"force={self.force.round(4)})"
        )


    
  


