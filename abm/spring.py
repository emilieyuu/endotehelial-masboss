# abm/spring.py
#
# A Spring represents a junction between 2 adjacent membrane nodes. 
#
# Cortical Spring: Passive elastic resistance of actin cortex,
#                  actively stiffened by DSP -> RhoA
# Stress Fibre: Active contractile element driven by TJP1 -> RhoC, 
#               only develops along the flow axis

import numpy as np
from abm.abm_helpers import bilinear_tension, get_protein_recruitment


class Spring: 
    """
    Mechnical junction between 2 adjacent membrane nodes. 

    GEOMETRY (updated every step from node position): 
    - L_current: physical distance between node_1 and node_2
    - unit_vect: unit vector pointing from node_1 to node_2
    - alignment: angle between spring an flow

    CORTICAL (RhoA pathway): 
    - L_cortex: rest length of cortical spring, fixed
    - k_cortex: baseline cortical stiffness, fixed
    - k_active: current cortical stiffness, remodelled by RhoA

    STRESS FIBRE: 
    - L_sf: current rest length of stress fibre, shortened by RhoC 
            on lateral junctions
    - k_sf: stress fibre stiffness, fixed fraction of k_cortex
            (stiffer per unit length than cortex)

    TENSION (computer each step, sum of both subsystems)
    - tension_cortex = cortical component, k_active * (L_current - L_cortex)
    - tension_sf = sf component, k_sf * (L_current - L_sf) * alignment
    - tension_total = cortex + sf, scalar used to compute fprce vectprs
    """

    def __init__(self, spring_id, node_1, node_2, 
                 rest_length, k_cortex, lut, cfg):
        
        self.lut = lut
        self.cfg = cfg
        self.id = spring_id
        self.node_1 = node_1
        self.node_2 = node_2

        # Cortical Subsystem
        self.L_cortex = rest_length # fixed geometric reference
        self.k_cortex = k_cortex # fixed passive stiffness reference
        self.k_active = k_cortex # dynamic, remodelled towards target by RhoA

        # Stress Fibre Subsystem
        sf_fraction = cfg['mechanics'].get('k_sf_fraction', 0.4)
        self.k_sf = k_cortex * sf_fraction # fixed stiffness
        self.L_sf = rest_length # dynamic rest length, shortened by RhoA

        # Instantaneous Geometry – Updated at Each Step 
        self.L_current = rest_length
        self.unit_vec = np.zeros(2)
        self.alignment = 0.0
        self._init_alignment = 0.0

        # Tension Components
        self.tension_cortex, self.tension_sf, self.tension_total = 0.0, 0.0, 0.0

        # Signalling State
        self.DSP, self.TJP1, self.JCAD = 0.0, 0.0, 0.0
        self.P_RhoA, self.P_RhoC = 0.0, 0.0

    # ------------------------------------------------------------------
    # 1: Update Geometry and Tension
    # ------------------------------------------------------------------
    def update_geometry(self, flow_direction):
        """ 
        Recompute all geometry from current node positions, then 
        compute tension for both subsystems. 

        Called first in every timestep.
        """

        # Compute and update length (norm) of spring
        diff = self.node_2.pos - self.node_1.pos 
        length = np.linalg.norm(diff) # Euclidian norm of vector

        if length < 1e-10:
            return # If length has collapsed, skip division
        
        self.L_current = length

        # Compute Spring alignment to flow
        self.unit_vec = diff / length # unit vector pointing from node1 to node2 (spring orientation)
        self.alignment = abs(np.dot(self.unit_vec, flow_direction)) # |cos(a)| between spring and flow axis

        # Compute cortical tension
        # Rest length is L_cortex (fixed geometry), stiffness is k_active (RhoA-remodelled)
        self.tension_cortex = bilinear_tension(
            l_current=self.L_current, 
            l_rest=self.L_cortex, 
            k_tensile=self.k_active, 
            kc_ratio = self.cfg['mechanics']['kc_ratio']
        )

        # Stress Fibre Tension 
        # Rest length is L_sf (RhoC-remodelled), stiffness is k_sf (fixed)
        sf_extension = self.L_current - self.L_sf
        self.tension_sf = max(
            self.k_sf * sf_extension * self._init_alignment, # weighted by alignment (lateral highest)
            0.0 # fibres do not generate compressive force (only pull, no pushh)
        )

        self.tension_total = self.tension_cortex + self.tension_sf

    # ------------------------------------------------------------------
    # 2: Apply Spring Forces
    # ------------------------------------------------------------------
    def apply_forces(self):
        """
        Push the net spring tension onto both connected nodes as equal
        and opposite force vectors.

        When tension > 0 (spring stretched):
            node_1 gets +force_vec  (pulled toward node_2) 
            node_2 gets -force_vec  (pulled toward node_1)

        When tension < 0 (spring compressed):
            node_1 gets +force_vec  (pushed away from node_2) 
            node_2 gets -force_vec  (pushed away from node_1) 
        """
        if self.L_current < 1e-10:
            return

        force_vec = self.tension_total * self.unit_vec
        self.node_1.apply_force(force_vec)
        self.node_2.apply_force(-force_vec)

    # ------------------------------------------------------------------
    # 3: Update Signalling (Junction proteins + Rho)
    # ------------------------------------------------------------------
    def update_signalling(self):
        """
        Converts mechanical state -> junction protein recruitment -> Rho activity.
        
        tau: mechanical stimulus for each protein.
        tau_dsp: raw tension, no weighting.
        tau_tjp1: most strongly loaded a junctions perpendicular to flow.
        tau_jcad: loaded at lateral junction feeling flow.
        """

        # Compute Mechanical Input
        tensile = max(self.tension_total, 0.0) # no recruitment in compressed junctions
        tau_dsp  = tensile
        tau_tjp1 = tensile #abs(self.tension_total) * (1.0 - self.alignment) 
        tau_jcad = tensile * self.alignment

        # Get Junction Protein Recruitment
        self.DSP  = get_protein_recruitment(self.cfg, tau_dsp,  'DSP')
        self.TJP1 = get_protein_recruitment(self.cfg, tau_tjp1, 'TJP1')
        self.JCAD = get_protein_recruitment(self.cfg, tau_jcad, 'JCAD')

        # Get RhoA/RhoC activation
        self.P_RhoA, self.P_RhoC = self.lut.query(
            self.DSP, self.TJP1, self.JCAD
        )


    # ------------------------------------------------------------------
    # 4: Remodel Mechanical Parameters
    # ------------------------------------------------------------------
    def _remodel_cortex(self, dt):
        """
        RhoA pathway: raises cortical stiffness. 
        Only activity above baseline ($RhoA_basal) drives remodelling.

        param dt: Timestep
        """
        mech = self.cfg['mechanics']

        # Compute RhoA Activity Above Baseline
        delta_rhoa = max(self.P_RhoA - self.lut.rhoa_rest, 0.0) 

        # Compute Target Stiffness
        k_target = self.k_cortex * (1.0 + mech['rhoa_k_gain'] * delta_rhoa)

        # First Order Lag Remodelling
        alpha = dt / mech['tau_remodel'] # how much 
        self.k_active += alpha * (k_target - self.k_active)

    def _remodel_sf(self, dt):
        """
        RhoC pathway: shortens stress fibre rest length on flow-aligned junctions

        param dt: Timestep
        """
        mech = self.cfg['mechanics']

        # Compute RhoC Activity Above Baseline
        delta_rhoc = max(self.P_RhoC - self.lut.rhoc_rest, 0.0) 

        # # Only shorten L_sf is spring is currently under tension. 
        if self.L_current > self.L_cortex:
            # Stress fibres assemble at stretched junctions 
            # Compute How Much Fibre Should Shrink By
            shrink = mech['rhoc_l_shrink'] * delta_rhoc * self._init_alignment
            shrink = min(shrink, mech.get('rhoc_max_shrink', 0.25))

            # Compute Target Shrinkage (Capped at 40% of inital length)
            L_sf_target = self.L_cortex * (1.0 - shrink)
            L_sf_target = max(L_sf_target, self.L_cortex * 0.4)
        else: 
            # Spring is slack or compressed — fibres relax back toward cortex length
            # This also handles the case where flow is removed: fibres dissolve
            L_sf_target = self.L_cortex

        # First Order Lag Remodelling
        alpha = dt / mech['tau_remodel']
        self.L_sf += alpha * (L_sf_target - self.L_sf)

    def update_remodelling(self, dt):
        """
        Public Remodelling Method: Remodel cortex followed by stress fibres
        Uses tension from this step, updating parameters for the next one. 
        Sees mechanical state after nodes move.
        """
        self._remodel_cortex(dt)
        self._remodel_sf(dt)

    # ------------------------------------------------------------------
    # Diagnostics & Debugging
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Flat state snapshot for logging."""
        return {
            'id':             self.id,
            'L_current':      round(self.L_current,      4),
            'L_cortex':       round(self.L_cortex,       4),
            'L_sf':           round(self.L_sf,           4),
            'k_active':       round(self.k_active,       4),
            'alignment':      round(self.alignment,      3),
            'init_alignment': round(self._init_alignment,3),
            'tension_cortex': round(self.tension_cortex, 4),
            'tension_sf':     round(self.tension_sf,     4),
            'tension_total':  round(self.tension_total,  4),
            'DSP':            round(self.DSP,            3),
            'TJP1':           round(self.TJP1,           3),
            'JCAD':           round(self.JCAD,           3),
            'P_RhoA':         round(self.P_RhoA,         3),
            'P_RhoC':         round(self.P_RhoC,         3),
        }

    def __repr__(self):
        return (
            f"Spring(id={self.id} | "
            f"L={self.L_current:.3f} L0={self.L_cortex:.3f} Lsf={self.L_sf:.3f} | "
            f"k={self.k_active:.3f} | "
            f"align={self.alignment:.2f} init={self._init_alignment:.2f} | "
            f"T={self.tension_total:.4f} "
            f"[cortex={self.tension_cortex:.4f} sf={self.tension_sf:.4f}] | "
            f"RhoA={self.P_RhoA:.3f} RhoC={self.P_RhoC:.3f})")