# abm/analysis/cell_measurement.py
#
# External measurement functions for the CellAgent.

import numpy as np
from src.abm.helpers.geometry import polar_mask, axial_coord, lateral_coord

def safe_mean(lst):
    mean = float(np.mean(lst)) if lst else 0.0
    return round(mean, 3)

def log_ratio(val1, val2, eps=1e-10):
    log_r = np.log2((val1 + eps) / (val2 + eps))
    return round(log_r, 3)

# ------------------------------------------------------------------
# Shape
# ------------------------------------------------------------------
def measure_shape(cell):
    """ 
    Shape descriptors in the flow-aligned frame.

    Major and minor axes: The axial and lateral extents of the
    node positions, measured directly from the flow axis. 
    AR: Major/Minor axis ratio.
    Perimeter: Yhe sum of edge lengths around the polygon ring.
    """
    pos = cell.positions

    # Axial and lateral extents in the flow-aligned frame.
    axial = axial_coord(pos, cell.centroid, cell.flow_axis)
    lateral = lateral_coord(pos, cell.centroid, cell.flow_axis)

    major = float(axial.max()   - axial.min())
    minor = float(lateral.max() - lateral.min())
    ar = major / (minor + 1e-10)

    # Polygon perimeter via consecutive edge lengths (closes the ring).
    edges = np.roll(pos, -1, axis=0) - pos
    perimeter = float(np.linalg.norm(edges, axis=1).sum())

    return {
        'ar':          round(ar, 3),
        'major':       round(major, 3),
        'minor':       round(minor, 3),
        'perimeter':   round(perimeter, 3),
        'area_ratio':  round(float(cell.current_area / cell.target_area), 4),
    }

# ------------------------------------------------------------------
# Cell-Level Snapshot
# ------------------------------------------------------------------
def measure_cell(cell):
    """
    Rich per-step snapshot of cell state.

    Returns a flat dict suitable for one row of a results DataFrame.

    Columns are grouped:
      - identifiers
      - shape (AR + perimeter + area ratio)
      - signalling: Rho activation (means + spatial split + balance)
      - signalling: junction protein recruitment (means + DSP spatial split)
      - cortex mechanics (T, k, a — means + spatial splits)
      - stress fibre mechanics (T, k, a, squeeze)
    """
    # ---- partition ----
    # Node partition
    mask = polar_mask(cell.positions, cell.centroid, cell.flow_axis, cell._polar_angle)
    polar_n = [n for n, m in zip(cell.nodes, mask) if m]
    lateral_n = [n for n, m in zip(cell.nodes, mask) if not m]

    # Spring partition
    polar_set = set(id(n) for n in polar_n)
    polar_s = [s for s in cell.springs
               if id(s.node_1) in polar_set and id(s.node_2) in polar_set]
    lateral_s = [s for s in cell.springs
                 if id(s.node_1) not in polar_set and id(s.node_2) not in polar_set]

    # ---- shape ----
    shape = measure_shape(cell)

    # ---- signalling: Rho ----
    rhoa_mean = safe_mean([n.rhoa for n in cell.nodes])
    rhoc_mean = safe_mean([n.rhoc for n in cell.nodes])

    # ---- signalling: Junction proteins ----
    dsp_mean = safe_mean([n.DSP  for n in cell.nodes])
    tjp1_mean = safe_mean([n.TJP1 for n in cell.nodes])

    dsp_polar = safe_mean([n.DSP  for n in polar_n])
    dsp_lateral = safe_mean([n.DSP  for n in lateral_n])
    tjp1_polar = safe_mean([n.TJP1  for n in polar_n])
    tjp1_lateral = safe_mean([n.TJP1  for n in lateral_n])

    # --- mechanics ---
    t_polar = safe_mean([s.T for s in polar_s])
    t_lateral = safe_mean([s.T for s in lateral_s])

    cortex_a_mean = safe_mean([s.a for s in cell.springs])
    cortex_T_mean = safe_mean([s.T for s in cell.springs])

    sf_a = round(cell.sf.a, 3)
    sf_T = round(cell.sf.T, 3)

    return {
        # --- identifiers ---
        'cell_id':           cell.id,

        # --- shape ---
        'ar':                shape['ar'],
        'area_ratio':        shape['area_ratio'],
#        'perimeter':         shape['perimeter'],
        'major':             shape['major'],
        'minor':             shape['minor'],

        # --- signalling: Loading & Rho ---
        'rhoa_mean':         rhoa_mean,
        'rhoc_mean':         rhoc_mean,
        'rho_balance':       round(rhoa_mean - rhoc_mean, 3),

        # --- signalling: recruitment ---
        'dsp_mean':          dsp_mean,
        'tjp1_mean':         tjp1_mean,
        'jcad_mean':         safe_mean([n.JCAD for n in cell.nodes]),

        'tjp1_dsp_balance':  round(dsp_mean - tjp1_mean, 3),
        'dsp_spread':        log_ratio(dsp_polar, dsp_lateral),
        'tjp1_spread':       log_ratio(tjp1_polar, tjp1_lateral),

        # --- loading ---
        't_load_polar':      safe_mean([n.tensile_load for n in polar_n]),
        't_load_lat':        safe_mean([n.tensile_load for n in lateral_n]),

        # --- cortex mechanics ---
        'cortex_T_polar':        t_polar,
        'cortex_T_lateral':      t_lateral,
        'cortex_T_mean':         cortex_T_mean,
        'cortex_a_mean':         cortex_a_mean,
        'cortex_force_spread':   log_ratio(t_polar, t_lateral),

        # --- stress fibre mechanics ---
        'sf_T':              sf_T,
        'sf_a':              sf_a,
        'sf_squeeze':        round(sf_T * cell.sf.nu, 3),

        # --- mechanics balance ---
        'activation_balance': cortex_a_mean - sf_a,
        'tension_balance':    log_ratio(cortex_T_mean, sf_T),
    }

# ------------------------------------------------------------------
# Spring Snapshot
# ------------------------------------------------------------------
def measure_springs(cell):
    """
    Per-spring state for one timestep.
    Returns a list of dicts (one per spring).
    """
    polar_set = set(id(n) for n in cell.polar_nodes)
    rows = []
    for s in cell.springs:
        is_polar = (id(s.node_1) in polar_set and
                    id(s.node_2) in polar_set)
        row = s.get_state()
        row['classification'] = 'polar' if is_polar else 'lateral'
        rows.append(row)
    return rows

# ------------------------------------------------------------------
# Spring Snapshot
# ------------------------------------------------------------------
def measure_nodes(cell):
    """
    Per-node state for one timestep.
    Returns a list of dicts (one per node).
    """
    polar_set = set(id(n) for n in cell.polar_nodes)
    rows = []
    for n in cell.nodes:
        row = n.get_state()
        row['classification'] = 'polar' if id(n) in polar_set else 'lateral'
        rows.append(row)
    return rows