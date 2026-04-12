# abm/analysis/cell_measurement.py
#
# External measurement functions for the CellAgent.

import numpy as np
from abm.helpers.geometry import polar_mask, axial_coord, lateral_coord
from src.utils import safe_mean

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
      - shape (PCA + perimeter + area ratio)
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
    rhoa_mean = float(np.mean([n.rhoa for n in cell.nodes]))
    rhoc_mean = float(np.mean([n.rhoc for n in cell.nodes]))

    return {
        # --- identifiers ---
        'cell_id':           cell.id,

        # --- shape ---
        'ar':                shape['ar'],
        'area_ratio':        shape['area_ratio'],
        'perimeter':         shape['perimeter'],
        'major':             shape['major'],
        'minor':             shape['minor'],

        # --- loading ---
        't_load_polar':      safe_mean([n.tensile_load for n in polar_n]),
        't_load_lat':        safe_mean([n.tensile_load for n in lateral_n]),

        # --- signalling: Loading & Rho ---
        'rhoa_mean':         round(rhoa_mean, 3),
        'rhoc_mean':         round(rhoc_mean, 3),
        'rho_balance':       round(rhoc_mean - rhoa_mean, 3),
        'rhoa_polar':        safe_mean([n.rhoa for n in polar_n]),
        'rhoa_lateral':      safe_mean([n.rhoa for n in lateral_n]),

        # --- signalling: recruitment ---
        'dsp_mean':          safe_mean([n.DSP  for n in cell.nodes]),
        'tjp1_mean':         safe_mean([n.TJP1 for n in cell.nodes]),
        'jcad_mean':         safe_mean([n.JCAD for n in cell.nodes]),
        'dsp_polar':         safe_mean([n.DSP  for n in polar_n]),
        'dsp_lateral':       safe_mean([n.DSP  for n in lateral_n]),

        # --- cortex mechanics ---
        'cortex_T_polar':    safe_mean([s.T for s in polar_s]),
        'cortex_T_lateral':  safe_mean([s.T for s in lateral_s]),
        'cortex_k_polar':    safe_mean([s.k for s in polar_s]),
        'cortex_k_lateral':  safe_mean([s.k for s in lateral_s]),
        'cortex_a_mean':     safe_mean([s.a for s in cell.springs]),

        # --- stress fibre mechanics ---
        'sf_T':              round(cell.sf.T, 4),
        'sf_a':              round(cell.sf.a, 3),
        'sf_k':              round(cell.sf.k, 4),
        'sf_squeeze_total':  round(cell.sf.T * cell.sf.nu, 4),
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