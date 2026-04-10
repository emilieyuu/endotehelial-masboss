# abm/membrane_node.py
#
# A discrete point on the cell membrane carrying mechanical and signalling state. 
# Nodes are the fundamental agents of the ABM.
#
# Responsibilities:
#   1. Mechanics — accumulate forces, integrate position
#   2. Load channels — accumulate mechanical stimuli for signalling
#   3. Signalling — map accumulated stimuli → junction proteins → Rho

import numpy as np
from abm.helpers.signalling import get_protein_recruitment
from abm.helpers.mechanics import overdamped_step
from src.utils import require

class MembraneNode:
    """
    A single membrane node: position, accumulated force, signalling state.
    """
    def __init__(self, node_id, position, lut, cfg):
        self.id = node_id
        self.pos = np.array(position, dtype=float)

        # --- Mechanics ---
        sim = require(cfg, 'simulation')
        self.force = np.zeros(2) # Force accumulator
        self.visc = require(sim, 'viscosity')
        self.max_disp = require(sim, 'max_displacement')

        # --- Load channels (signalling stimuli) --
        self.tensile_load = 0.0 # tensile stimulus (cortex + SF + flow baseline)
        self.shear_total = 0.0 # shear stimulus (flow magnitude)

        # --- Signalling ---
        self.lut = lut
        self.cfg = cfg

        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
        self.rhoa, self.rhoc = 0.0,  0.0
        
    # ------------------------------------------------------------------
    # Reset – called at the start of each timestep
    # ------------------------------------------------------------------
    def reset_loads(self):
        """Zero load channels before re-accumulation this step."""
        self.tensile_load = 0.0
        self.shear_total = 0.0

    def reset_force(self):
        """Zero force accumulator before re-accumulation this step."""
        self.force[:] = 0.0

    # ------------------------------------------------------------------
    # Accumulators — called by springs, stress fibre, flow
    # ------------------------------------------------------------------
    def add_tensile_load(self, load):
        """Add a tensile stimulus contribution from a mechanical agent."""
        self.tensile_load += load

    def apply_force(self, force):
        """Add a force vector contribution from a mechanical agent."""
        force = np.asarray(force)
        self.force += force

    # ------------------------------------------------------------------
    # Integration — called once per step after all forces are accumulated
    # ------------------------------------------------------------------
    def integrate_step(self, dt):
        """Converts net force to displacement."""
        self.pos += overdamped_step(self.force, self.visc, dt, self.max_disp)


    # ------------------------------------------------------------------
    # Signalling — called once per step after integration
    # ------------------------------------------------------------------
    def update_signalling(self):
        """
        Update protein recruitment and Rho activation from current loads.
        Pipeline: mechanical loads → Hill → junction proteins → LUT → Rho activation

        DSP: tensile loading 
        TJP1: total shear magnitude 
        JCAD: total shear magnitude 
        """
        # Clamp mechanical loads to non-negatice
        S_dsp  = max(self.tensile_load, 0.0)
        S_tjp1 = max(self.shear_total, 0.0) 
        S_jcad = max(self.shear_total, 0.0) 

        # Hill → junction protein recruitment
        self.DSP  = get_protein_recruitment(self.cfg, S_dsp, 'DSP')
        self.TJP1 = get_protein_recruitment(self.cfg, S_tjp1, 'TJP1')
        self.JCAD = get_protein_recruitment(self.cfg, S_jcad, 'JCAD')

        # LUT → Rho activation
        self.rhoa, self.rhoc = self.lut.query(self.DSP, self.TJP1, self.JCAD)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------  
    def get_state(self):
        return {
            'id': self.id, 
            'position': (self.pos[0].round(2), self.pos[1].round(2)), 
            'tensile_load': self.tensile_load, 
            'shear_total': self.shear_total, 
            'DSP': self.DSP, 
            'TJP1': self.TJP1, 
            'JCAD': self.JCAD, 
            'rhoa': self.rhoa, 
            'rhoc': self.rhoc
        }    
    
    def __repr__(self):
        return (
            f"MembraneNode(id={self.id} | pos={self.pos.round(2)} | "
            f"DSP={self.DSP:.3f} | TJP1={self.TJP1:.3f} | JCAD={self.JCAD:.3f} | "
            f"rhoa={self.rhoa:.3f} | rhoc={self.rhoc:.3f})"
        )