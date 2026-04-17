# abm/endothelial_cell.py
#
# 2D endothelial cell — closed ring of membrane nodes connected by
# cortical springs, with a single internal stress fibre cable. 
# The cell is the top-level agent of the ABM: orchestrates timestep and 
# owns the node/spring/SF structures.
#
# Forces applied per timestep:
#   1. Shear drag — extensional, polar nodes only
#   2. Cortex springs — bilinear, RhoA-stiffened
#   3. SF waist squeeze — inward at the cell waist, Poisson coupling
#   4. Area pressure — outward when current area < target

import numpy as np

from abm.membrane_node import MembraneNode
from abm.cortex_spring import CortexSpring
from abm.sf_cable import StressFibreCable

from abm.helpers.geometry import (
    axial_coord, polar_mask, 
    polygon_area, polygon_outward_normals, polygon_arc_lengths
)

from src.utils import require

class Cell:
    """
    Closed-ring polygonal cell with cortex springs and one stress fibre.

    State:
      nodes — list of MembraneNode (positions, loads, signalling)
      springs — list of CortexSpring connecting adjacent nodes
      sf — StressFibre spanning the flow axis
      flow_axis — unit direction vector (structural, fixed at init)
      target_area, current_area — for area conservation pressure
    """

    def __init__(self, cell_id, flow_axis, lut, cfg):
        self.id = cell_id
        self.lut = lut

        # Normalise flow axis
        self.flow_axis = np.asarray(flow_axis, dtype=float)
        self.flow_axis = self.flow_axis / np.linalg.norm(self.flow_axis)

        # --- Config values ---
        cell_cfg = require(cfg, 'cell')
        self.n_nodes = require(cell_cfg, 'n_nodes')
        self._p_area = require(cell_cfg, 'p_area')
        self._polar_angle = require(cell_cfg, 'polar_angle')

        centroid = np.asarray(require(cell_cfg, 'centroid'), dtype=float)
        radius = require(cell_cfg, 'radius')

        # --- Build geometry ---
        self.nodes = self._init_node_ring(centroid, radius, cfg)
        self.springs = self._init_springs(cfg)
        self.sf = self._init_sf(cfg)
        
        # -- Area conservation state ---
        self.target_area = polygon_area(self.positions) 
        self.current_area = self.target_area 
    
    # ------------------------------------------------------------------
    # Initialisation Helpers
    # ------------------------------------------------------------------
    def _init_node_ring(self, centroid, radius, cfg):
        """
        Place n_nodes evenly on a circle of given radius around centroid.
        Node 0 sits at the top (+π/2 offset) so the indexing has a
        consistent geometric reference.
        """
        angles = np.linspace(0, 2*np.pi, self.n_nodes, endpoint=False) + np.pi/2
        nodes  = []
        for i, angle in enumerate(angles):
            pos = centroid + radius * np.array([np.cos(angle), np.sin(angle)])
            nodes.append(MembraneNode(i, pos, self.lut, cfg))
        return nodes

    def _init_springs(self, cfg):
        """Connect adjacent nodes with cortex springs in a closed ring."""
        springs = []
        for i in range(self.n_nodes):
            n1 = self.nodes[i]
            n2 = self.nodes[(i + 1) % self.n_nodes]
            L_rest = np.linalg.norm(n2.pos - n1.pos)
            springs.append(CortexSpring(
                id=i, node_1=n1, node_2=n2, 
                rest_length=L_rest, cfg=cfg
            ))
        return springs
    
    def _init_sf(self, cfg):
        """Create a single stress fibre spanning the two most-polar nodes."""
        projections = axial_coord(self.positions, self.centroid, self.flow_axis)
        up_id = int(np.argmin(projections))
        dn_id = int(np.argmax(projections))
        L_rest = projections.max() - projections.min()

        return StressFibreCable(
            node_up=self.nodes[up_id],
            node_down=self.nodes[dn_id],
            nodes=self.nodes,
            rest_length=L_rest,
            cfg=cfg,
        )
    
    # ------------------------------------------------------------------
    # Geometry Properties
    # ------------------------------------------------------------------
    @property
    def positions(self):
        """(N, 2) array of current node positions."""
        return np.array([n.pos for n in self.nodes])

    @property
    def centroid(self):
        """Geometric centroid. Recomputed each call (cell may drift)."""
        return self.positions.mean(axis=0)
    
    # ------------------------------------------------------------------
    # Classification properties 
    # ------------------------------------------------------------------
    @property
    def polar_nodes(self):
        """Nodes currently within the polar cone"""
        mask = polar_mask(self.positions, self.centroid, 
                          self.flow_axis, self._polar_angle)
        return [n for n, m in zip(self.nodes, mask) if m]
    
    @property
    def lateral_nodes(self):
        """Nodes outside the polar cone. Complement of polar_nodes."""
        mask = polar_mask(self.positions, self.centroid, 
                          self.flow_axis, self._polar_angle)
        return [n for n, m in zip(self.nodes, mask) if not m]
    
    @property
    def polar_springs(self):
        """Springs with at least one polar endpoint."""
        polar_set = set(id(n) for n in self.polar_nodes)
        return [s for s in self.springs
                if id(s.node_1) in polar_set and id(s.node_2) in polar_set]

    @property
    def lateral_springs(self):
        """Springs with both endpoints lateral."""
        polar_set = set(id(n) for n in self.polar_nodes)
        return [s for s in self.springs
                if id(s.node_1) not in polar_set and id(s.node_2) not in polar_set]

    # ------------------------------------------------------------------
    # Signalling aggregates (cell-wide Rho means)
    # ------------------------------------------------------------------
    @property 
    def rhoc_mean(self):
        return float(np.mean([n.rhoc for n in self.nodes]))
    
    @property 
    def rhoa_mean(self):
        return float(np.mean([n.rhoa for n in self.nodes]))

    # ------------------------------------------------------------------
    # Force application (private)
    # ------------------------------------------------------------------
    def _apply_shear_drag(self, flow):
        """Apply extensional shear drag at polar nodes only."""

         # Signed axial coordinates
        # polar_positions = np.array([n.pos for n in polar])
        axial = axial_coord(self.positions, self.centroid, self.flow_axis)
        p_ref = np.max(np.abs(axial))

        if p_ref < 1e-10:
            return   # degenerate, nothing to do
        p = axial / p_ref

        raw_weights = p * p
        total = raw_weights.sum()
        if total < 1e-10:
            return
        weights = raw_weights * (self.n_nodes / total)

        # Apply per-node, signed by axial direction
        for node, w, a in zip(self.nodes, weights, axial):
            force = flow.drag_on_node(weight=w, axial_sign=np.sign(a))
            node.apply_force(force)
            

    def _apply_pressure(self):
        """
        Soft area conservation — outward pressure proportional to area deficit.

        Pressure magnitude = p_area × (target_area − current_area)
        Applied at each node along its outward normal, weighted by its local arc length

        Only fires on area_deficit to prevents elongation runaway.
        """
        area_deficit = self.target_area - self.current_area
        if area_deficit <= 0:
            return

        pressure = self._p_area * area_deficit
        positions = self.positions
        normals = polygon_outward_normals(positions)
        arc_lengths = polygon_arc_lengths(positions)

        for node, normal, dL in zip(self.nodes, normals, arc_lengths):
            node.apply_force(pressure * dL * normal)

    # ------------------------------------------------------------------
    # Timestep
    # ------------------------------------------------------------------
    def step(self, flow_field, dt):
        """
        Advance one timestep.

        Phase order:
        1. Reset accumulators
        2. External forces (flow drag, signalling stimulus)
        3. Update mechanical geometry and tensions (cortex, SF)
        4. Accumulate tensile stimuli on nodes (cortex, SF)
        5. Apply mechanical forces (cortex, SF)
        6. Area pressure
        7. Integration (node positions advance)
        8. Signalling (Hill → LUT → Rho)
        9. Remodelling (cortex stiffness/activation, SF activation)
        10. Sync current_area for next step's pressure

        Forces accumulate before integration. Signalling reads
        post-integration geometry. Remodelling sets state for next step.
        """
        # Reset Tensile Loads
        for n in self.nodes:
            n.reset_tensile_load()

        # 1. External stimuli from flow
        for node in self.nodes:
            node.shear_load = flow_field.magnitude
            #node.tensile_load += flow_field.magnitude

        # 2. Geometry + tension 
        for s in self.springs:
            s.update_geometry_tension()
        self.sf.update_geometry_tension()

        # 3. Tensile stimuli to load channels
        for s in self.springs:
            s.accumulate_loads()
        self.sf.accumulate_loads(self.polar_nodes)

        # 4. Mechanical forces applied to nodes
        for s in self.springs:
            s.apply_forces()
        self.sf.apply_forces()

        # 5. Area pressure
        self.current_area = polygon_area(self.positions)
        self._apply_pressure()

         # 6. Integration — each node advances by its net force
        for node in self.nodes:
            node.step(dt)

        # 7. Remodelling — cortex reads local RhoA, SF reads cell-wide RhoC mean
        for s in self.springs:
            s.update_stiffness_and_activation(dt)
        self.sf.update_activation(mean_rhoc=self.rhoc_mean, dt=dt)

        # 8. Sync area for next step's pressure calculation
        self.current_area = polygon_area(self.positions)
 
    # ------------------------------------------------------------------
    # Diganostics
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"EndothelialCell(id={self.id} | n_nodes={self.n_nodes} | "
            f"rhoa={self.rhoa_mean:.3f} rhoc={self.rhoc_mean:.3f} | "
        )