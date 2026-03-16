# abm/endothelial_cell.py
# 
# A closed ring of membrane nodes representing the cell cortex in 2D.
# Orchestrates springs, nodes, flow, and pressure each timestep.
#
# Responsibitilies: 
#   - Initialise and own node ring and spring ring.
#   - Run the mechanixal timestep in correct order. 
#   - Apply pressure forces (area conservation).
#   - Expose shape and state measurements. 

import numpy as np
from abm.membrane_node import MembraneNode
from abm.spring import Spring
from abm.abm_helpers import measure_shape

class EndothelialCell: 
    """
    A 2D endothelial cell represented as a closed ring of n_nodes
    membrane nodes connected by springs.

    Mechanical systems:
        1. Shear drag       — from FlowField, differential across membrane
        2. Spring forces    — cortical tension + stress fibre tension
        3. Pressure forces  — area conservation (cytoplasmic incompressibility)
    """
    def __init__(self, cell_id: int, centroid: np.ndarray, lut, cfg: dict, 
                 n_nodes: int = 12, radius: float = 10.0):
        
        # Cell General Properties
        self.id = cell_id
        self.n_nodes = n_nodes
        self.cfg = cfg

        # Cell Meachnics and Geometry
        mech_cfg = self.cfg['mechanics']
        self.k_area = mech_cfg['k_area']
        self.k_cortex = mech_cfg['k_cortex']
        self.rest_length = 2 * radius * np.sin(np.pi / n_nodes) # distance between adjacent nodes on regular n-gon: 2R sin(π/n)

        # Initialise Node Ring and Springs
        self.nodes = self._init_node_ring(centroid, n_nodes, radius)
        self.springs = self._init_springs(lut)

        # Cell Area 
        self.target_area = self._compute_area() # fixed, acts as reference
        self.current_area = self.target_area # dynamic, remodelled to maintain "incompressible cytoplasm"

        print(f"DEBUG: Initialised Cell {self.id}: ")
        print(self.__repr__())
    
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_node_ring(self, centroid, n_nodes, radius):
        """
        Place n_nodes evenly around a circle.
        Angles spaced at 2π/n intervals, starting at angle 0 (rightmost point).
        """
        # Compute angles between nodes. 
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)  
        nodes = []

        for i, angle in enumerate(angles):
            pos = np.array([
                centroid[0] + radius * np.cos(angle),
                centroid[1] + radius * np.sin(angle),
            ])
            nodes.append(MembraneNode(i, pos))

        return nodes
    
    def _init_springs(self, lut):
        """
        Connect each node to the next (ring topology).

        Each spring is initialised at geometric rest length with k_cortex from config.
        """
        springs = []

        for i in range(self.n_nodes):
            # Node i connects to node (i+1) mod n_nodes — last node wraps back to node 0.
            node1 = self.nodes[i]
            node2 = self.nodes[(i + 1) % self.n_nodes]

            springs.append(Spring(
                spring_id=i, node_1=node1, node_2=node2, # id matches its lower-indexed node.
                rest_length=self.rest_length, k_cortex = self.k_cortex,
                lut=lut, cfg=self.cfg
            ))

        return springs
        
    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        return np.array([n.pos for n in self.nodes])
    
    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)
    
    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _compute_area(self):
        """
        Shoelace formula for the signed area of a polygon.
        A = ½ |Σ (xᵢ yᵢ₊₁ − xᵢ₊₁ yᵢ)|

        Works for any simple polygon — no assumption of convexity.
        """
        pos = self.positions
        x, y = pos[:, 0], pos[:, 1]

        return 0.5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
 
    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------
    def _apply_pressure_forces(self):
        """
        Area-conservation pressure force — models cytoplasmic 
        incompressibility in 2D.
        """
        current_area = self._compute_area()
        centroid = self.centroid

        # Internal pressure magnitude depends pressure constant and distance from target
        # positive pressure → area below target → outward force
        # negative pressure → area above target → inward force
        pressure = self.k_area * (self.target_area - current_area)

        # Apply force to each node along  outward normal from cemtre
        for node in self.nodes:
            outward = node.pos - centroid
            norm = np.linalg.norm(outward)
            if norm < 1e-10:
                continue
            norm_unit = outward / norm
            node.apply_force(norm_unit * pressure)

    # ------------------------------------------------------------------
    # Timestep
    # ------------------------------------------------------------------
    def step(self, flow_field, dt):
        """
        Advance the cell by one mechanical time step. 
        """
        # Step 1: Shear Drag Force Onto Nodes
        centroid = self.centroid # compute centroid once
        for node in self.nodes:
            shear_force = flow_field.get_force_on_node(node.pos, centroid)
            node.apply_force(shear_force)

        # Step 2: Spring Geometry and Tension
        for spring in self.springs:
            spring.update_geometry(flow_field.direction)
        
        # Step 3: Spring Forces Onto Nodes
        for spring in self.springs:
            spring.apply_forces()
            
        # Step 4: Pressure Force Onto Nodes
        self._apply_pressure_forces()

        # Step 5: Integrate Node Positions
        gamma = self.cfg['sim']['gamma']
        max_disp = self.rest_length * 0.1
        for node in self.nodes:
            node.update(dt, gamma, max_disp)

        # Step 6: Spring Signaling 
        for spring in self.springs:
            spring.update_signalling()

        # Step 7: Spring Remodellig (update parameters for next step)
        for spring in self.springs:
            spring.update_remodelling(dt)

        # Step 8: Sync area for measure_shape() — must come after node positions update
        self.current_area = self._compute_area()

    # ------------------------------------------------------------------
    # Diagnostics & Debuggings
    # ------------------------------------------------------------------

    def _spring_populations(self):
        """
        Split springs into lateral (alignment > 0.5) and 
        perpendicular  (alignment <= 0.5) populations.
        """
        lateral  = [s for s in self.springs if s.alignment > 0.5]
        end_face = [s for s in self.springs if s.alignment <= 0.5]
        return lateral, end_face
    
    def measure_shape(self) -> dict:
        """
        PCA-based shape descriptors matching ImageJ conventions.
        """
        pos = self.positions
        centered = pos - pos.mean(axis=0)

        eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
        eigvals = np.maximum(eigvals, 0.0)

        major_vec = eigvecs[:, 1]
        major = 2.0 * np.sqrt(eigvals[1])
        minor = 2.0 * np.sqrt(eigvals[0])

        perim = np.sum(np.linalg.norm(
            np.diff(np.vstack([pos, pos[0]]), axis=0), axis=1
        ))

        ar = major / (minor + 1e-10)
        orientation = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
        elong_idx   = (ar - 1.0) * abs(np.cos(np.radians(orientation)))

        return {
            'aspect_ratio':     round(ar, 3),
            'orientation':      round(orientation, 2),
            'circularity':      round(4.0 * np.pi * self.current_area /
                                    (perim ** 2 + 1e-10), 3),
            'elongation_index': round(elong_idx, 4),
            'perimeter':        round(perim, 4),
        }
    
    def get_state(self) -> dict:
        """
        Minimal state snapshot for monitoring during simulation.

        metrics:
            ar          — primary phenotype readout
            orientation — elongation direction relative to flow
            area_err    — area conservation health (target 0.9–1.1)

        signalling:
            rho_balance     — RhoC - RhoA (positive = elongation-promoting)
            lateral_rhoc    — RhoC on lateral springs (direct fibre driver)

        mechanics:
            t_sf        — mean stress fibre tension (rising = elongation underway)
            t_cortex    — mean cortical tension (rising = stiffening/failed elongation)

        remodelling:
            mean_lsf_ratio — mean L_sf/L_cortex on lateral springs
                            <1.0 confirms fibres have shortened
                            stays at 1.0 in failed elongation phenotype
        """
        shape = self.measure_shape()
        lateral, end_face = self._spring_populations()

        # Signalling
        rho_balance = float(np.mean([s.P_RhoC - s.P_RhoA for s in self.springs])) # RhoC - RhoA (positive = RhoC dominant = elongation)
        lateral_rhoc = float(np.mean([s.P_RhoC for s in lateral])) if lateral else 0.0 # mean rhoc on lateral springs
        endface_rhoa = float(np.mean([s.P_RhoA for s in end_face])) if end_face else 0.0 # mean rhoa on perpendicular springs
        
        # Tension: 
        t_cortex       = float(np.mean([s.tension_cortex for s in self.springs])) # mean cortical tension
        t_sf           = float(np.mean([s.tension_sf for s in self.springs])) # mean stress fibre tension
        lateral_t_sf   = float(np.mean([s.tension_sf for s in lateral])) if lateral else 0.0 # mean fibre tension on lateral springs only

        # Remodellign
        mean_k_active  = float(np.mean([s.k_active for s in self.springs])) # mean cortical stiffness
            # > 1.0 means RhoA-driven stiffening has occured
        mean_lsf_ratio = float(np.mean([s.L_sf / s.L_cortex for s in lateral])) if lateral else 1.0 # mean L_sf/L_cortex across lateral springs
            # <1.0 means fibres have shortened (elongation underway)
            # =1.0 means no fibre remodelling yet
        
        return {
            "cell_id": self.id,
            "metrics": {
                "ar":              round(shape['aspect_ratio'], 3), # major (horizontal) / minor (vertical) axes
                "orientation":     round(shape['orientation'], 2),
                "circularity":     round(shape['circularity'], 3),
                "elongation_index":round(shape['elongation_index'],3),
                "area_err":        round(self.current_area / self.target_area, 4),
            },
            "signalling": {
                "lateral_rhoc": round(lateral_rhoc, 3), 
                "endface_rhoa": round(endface_rhoa, 3),
                "rho_balance":  round(rho_balance, 3),
            },
            "mechanics": {
                "t_cortex":      round(t_cortex, 4),
                "t_sf":          round(t_sf, 4),
                "lateral_t_sf":  round(lateral_t_sf, 4),
            },
            "remodelling": {
                "mean_k_active":  round(mean_k_active,  4),
                "mean_lsf_ratio": round(mean_lsf_ratio, 4),
            },
        }


    def __repr__(self):
        s = self.get_state()
        return (
            f"EndothelialCell(id={self.id} | "
            f"n={self.n_nodes} | "
            f"centroid={self.centroid.round(2)} | "
            f"ar={s['metrics']['ar']:.2f} | "
            f"area_err={s['metrics']['area_err']:.3f} | "
            f"rho_bal={s['signalling']['rho_balance']:+.3f} | "
            f"lsf={s['remodelling']['mean_lsf_ratio']:.3f} | "
            f"t_sf={s['mechanics']['t_sf']:.4f})"
        )
    