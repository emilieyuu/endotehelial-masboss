# abm/analysis/cell_measurement.py
#
# External measurement functions for EndothelialCell.
# All return flat dicts suitable for DataFrame rows.

import numpy as np
from src.utils import safe_mean

# ------------------------------------------------------------------
# Shape
# ------------------------------------------------------------------
def measure_shape(cell):
    """PCA-based shape descriptors."""
    pos      = cell.positions
    centered = pos - pos.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    eigvals   = np.maximum(eigvals, 0.0)
    major_vec = eigvecs[:, 1]
    major     = 2.0 * np.sqrt(eigvals[1])
    minor     = 2.0 * np.sqrt(eigvals[0])
    ar        = major / (minor + 1e-10)
    orientation = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

    return {
        'ar':          round(ar, 3),
        'orientation': round(orientation, 2),
        'area_ratio':  round(cell.current_area / cell.target_area, 4),
    }


# ------------------------------------------------------------------
# Forces — focused diagnostics
# ------------------------------------------------------------------
def measure_forces(cell):
    """
    Key force diagnostics for understanding elongation mechanics.

    Returns flat dict. Measures chosen for phenotype interpretation:
        - Shear: pole vs lateral f_normal gradient (drives signalling)
        - Cortex: tension and stiffness by domain (resists shape change)
        - SF: cable tension, peak nodal force, squeeze (drives elongation)
        - FA: anchoring force (stabilises elongated shape)
        - Area: pressure magnitude (converts squeeze to elongation)
    """
    pole_nodes   = [n for n in cell.nodes if n.role in ('upstream', 'downstream')]
    lat_nodes    = [n for n in cell.nodes if n.role == 'lateral']
    pole_springs = [s for s in cell.springs if s.side == 'polar']
    lat_springs  = [s for s in cell.springs if s.side == 'flank']

    # --- Shear ---
    # pole_fn = safe_mean([n.f_normal for n in pole_nodes])
    # lat_fn  = safe_mean([n.f_normal for n in lat_nodes])

    # --- Cortical tension ---
    pole_tension   = safe_mean([s.t_cortex for s in pole_springs])
    lat_tension    = safe_mean([s.t_cortex for s in lat_springs])
    k_active_pole = safe_mean([s.k_cortex for s in pole_springs])
    k_active_lat  = safe_mean([s.k_cortex for s in lat_springs])

    # --- Stress fibre ---
    sf_tension = safe_mean(cell.stress_fibre.t_sf)

    # Peak nodal force from distributed application (50% to centre)
    sf_node_max = sf_tension * 0.5


    # --- Area conservation ---
    area_deficit  = cell.target_area - cell.current_area
    area_pressure = (cell.cfg['mechanics']['k_area'] * area_deficit
                     if area_deficit > 0 else 0.0)

    return {
        # Shear gradient
        # 'shear_fn_pole':     round(pole_fn, 3),
        # 'shear_fn_lat':      round(lat_fn, 3),
        # 'shear_fn_diff':     round(pole_fn - lat_fn, 3),

        # Cortex resistance
        'cortex_T_pole':     round(pole_tension, 4),
        'cortex_T_lat':      round(lat_tension, 4),
        'k_active_pole':     round(k_active_pole, 4),
        'k_active_lat':      round(k_active_lat, 4),

        # SF elongation drive
        'sf_tension':        round(sf_tension, 4),
        'sf_node_max':       round(sf_node_max, 4),
       # 'sf_squeeze_max':    round(max_squeeze, 4),

        # FA stabilisation
     #   'fa_force':          safe_mean(fa_forces),

        # Area pressure
        'area_pressure':     round(area_pressure, 4),
    }

