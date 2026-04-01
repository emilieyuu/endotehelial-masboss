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
    pole_fn = safe_mean([n.f_normal for n in pole_nodes])
    lat_fn  = safe_mean([n.f_normal for n in lat_nodes])

    total_ft = safe_mean([n.f_tangential for n in cell.nodes])

    # --- Cortical tension ---
    pole_tension   = safe_mean([s.t_cortex for s in pole_springs])
    lat_tension    = safe_mean([s.t_cortex for s in lat_springs])
    a_cortex_pole = safe_mean([s.a_cortex for s in pole_springs])
    a_cortex_lat  = safe_mean([s.a_cortex for s in lat_springs])

    # --- Stress fibre ---
    sf_tension = safe_mean(cell.stress_fibre.t_sf)

    # Peak nodal force from distributed application (50% to centre)
    sf_node_max = sf_tension * 0.5

    # Max squeeze magnitude across all lateral nodes
    # max_squeeze = 0.0
    # sf = cell.stress_fibre
    # if sf.t_sf < 1e-6:
    #     continue
    # ux, uy = sf.unit_vec
    # perp = np.array([-uy, ux])
    # max_d = max(abs(np.dot(n.pos - sf.cable_mid, perp)) for n in cell.nodes)
    # if max_d < 1e-10:
    #     continue

    # for n in lat_nodes:
    #     f = abs(sf.get_squeeze_force(n, max_d))
    #     if f > max_squeeze:
    #         max_squeeze = f

    # --- FA anchoring ---
    # k_fa_base = cell.cfg['mechanics'].get('k_fa', 2.0)
    # k_fa = k_fa_base * (1.0 + cell.a_sf)

    # fa_forces = []
    # for node_id, fa_pos in cell.fa_positions.items():
    #     node = cell.nodes[node_id]
    #     disp = node.pos - fa_pos
    #     axial_disp = np.dot(disp, cell.flow_direction) * cell.flow_direction
    #     fa_forces.append(abs(k_fa * axial_disp))

    # --- Area conservation ---
    area_deficit  = cell.target_area - cell.current_area
    area_pressure = (cell.cfg['mechanics']['k_area'] * area_deficit
                     if area_deficit > 0 else 0.0)

    return {
        # Shear gradient
        'shear_fn_pole':     round(pole_fn, 3),
        'shear_fn_lat':      round(lat_fn, 3),
        'shear_fn_diff':     round(pole_fn - lat_fn, 3),
        'shear_tangential':  total_ft,

        # Cortex resistance
        'cortex_T_pole':     round(pole_tension, 4),
        'cortex_T_lat':      round(lat_tension, 4),
        'a_cortex_pole':     round(a_cortex_pole, 4),
        'a_cortex_lat':      round(a_cortex_lat, 4),

        # SF elongation drive
        'sf_tension':        round(sf_tension, 4),
        'sf_node_max':       round(sf_node_max, 4),
       # 'sf_squeeze_max':    round(max_squeeze, 4),
        'a_sf':              round(cell.a_sf, 4),

        # FA stabilisation
     #   'fa_force':          safe_mean(fa_forces),

        # Area pressure
        'area_pressure':     round(area_pressure, 4),
    }


# ------------------------------------------------------------------
# Cell info — structural summary for testing and reporting
# ------------------------------------------------------------------
def cell_info(cell):
    """
    Structural summary of cell topology and classification.

    Returns dict with:
        - Node counts by role
        - Spring counts by side
        - FA node IDs and positions
        - SF cable endpoints and rest length
        - Initial geometry (target area, rest length)
    """
    # Node classification
    roles = {}
    for n in cell.nodes:
        roles.setdefault(n.role, []).append(n.id)

    # Spring classification
    sides = {}
    for s in cell.springs:
        sides.setdefault(s.side, []).append(s.id)

    # FA info
    fa_info = {}
    for nid, pos in cell.fa_nodes.items():
        fa_info[nid] = {
            'initial_pos': pos.copy(),
            'current_pos': cell.nodes[nid].pos.copy(),
            'ratchet_pos': cell.fa_max_displacement[nid].copy(),
            'role':        cell.nodes[nid].role,
        }

    # SF info
    sf_info = []
    for sf in cell.stress_fibres:
        sf_info.append({
            'upstream_node':   sf.node_upstream.id,
            'downstream_node': sf.node_downstream.id,
            'L_rest':          round(sf.L_rest, 3),
            'L_current':       round(sf.L_current, 3),
        })

    return {
        'n_nodes':        cell.n_nodes,
        'nodes_by_role': {role: {'count': len(ids), 'ids': ids}
                          for role, ids in roles.items()},
        'n_springs':      len(cell.springs),
        'springs_by_side': {side: {'count': len(ids), 'ids': ids}
                           for side, ids in sides.items()},
        'fa_nodes':       fa_info,
        'stress_fibres':  sf_info,
        'target_area':    round(cell.target_area, 2),
        'rest_length':    round(cell.rest_length, 4),
    }