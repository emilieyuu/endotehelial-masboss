# abm/helpers/geometry.py
#
# Pure geometric helpers. 
# Function take raw arrays and return raw arrays. 
# 
# Convention: an "axis" is a 2D direction vector, functions normalise it. 
import numpy as np

def axial_coord(points, origin, axis):
    """
    Signed axial coordinate of point(s) relative to origin, along axis. 
    Returns raw distance in length units (not normalised)

    points: (2,) or (N, 2) – point(s) to project
    origin: (2,) – reference point (projecton relative to this)
    axis: (2,) – direction vector, normalised internally
    Returns : float if a single point was passed, else (N,) array
    """
    # Normalis axis
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    
    # Compute projection of point onto axis with dot product
    points = np.atleast_2d(np.asarray(points, dtype=float))
    coords = (points - origin) @ axis 

    return float(coords[0]) if coords.size == 1 else coords

def lateral_coord(points, origin, axis):
    """
    Signed lateral coordinate of point(s)— perpendicular to axis.

    Sign convention: positive on the left side of `axis` (90° CCW rotation),
    negative on the right. 

    Same shape semantics as axial_coord().
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    perp = perpendicular(axis)

    points = np.atleast_2d(np.asarray(points, dtype=float))
    coords = (points - origin) @ perp

    return float(coords[0]) if coords.size == 1 else coords

def perpendicular(vec):
    """
    2D perpendicular vector – rotates input counter-clockwise
    """
    return np.array([-vec[1], vec[0]])

def polar_mask(points, origin, axis, angle_deg):
    """
    Boolean mask marking point withing angle_deg from axis, measured from origin.

    points: (N, 2) — points to classify
    origin: (2,) — reference point 
    axis: (2,) — direction vector, normalised internally
    angle_deg : float — half-cone angle in degrees. Points within this
                        angle of ±axis are marked polar.
    Returns: (N,) boolean array — True for polar, False for lateral.
    """
    # Normalise axis.
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # Offset vectors from origin to each point.
    offsets = points - origin
    norms = np.linalg.norm(offsets, axis=1)

    # |cos(angle)| between each offset and the flow axis.
    cos_angles = np.abs(offsets @ axis) / (norms + 1e-10)
    
    # A point is polar iff its angle to the axis is ≤ angle_deg,
    cos_threshold = np.cos(np.deg2rad(angle_deg))
    return cos_angles >= cos_threshold 

def polygon_area(points):
    """
    Signed polygon area via the shoelace formula, returned as absolute value.

    points: (N, 2) — ordered vertices of a closed polygon
    Returns: float — polygon area
    """
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )

def polygon_outward_normals(points):
    """
    Outward unit normals at each vertex of a closed polygon.

    Each vertex normal is the normalised sum of the two adjacent edge
    normals (each edge rotated 90° clockwise to point outward, assuming
    counter-clockwise vertex ordering).

    points: (N, 2) — ordered vertices
    Returns: (N, 2) — unit outward normals, one per vertex. 
    """
    prev_pts = np.roll(points, +1, axis=0)
    next_pts = np.roll(points, -1, axis=0)

    # Edge vectors incoming and outgoing at each vertex.
    e_in  = points - prev_pts
    e_out = next_pts - points

    # Rotate each edge 90° clockwise: (dx, dy) → (dy, -dx). 
    # Gives outward normals under counter-clockwise vertex ordering.
    n_in  = np.column_stack([e_in[:, 1],  -e_in[:, 0]])
    n_out = np.column_stack([e_out[:, 1], -e_out[:, 0]])

    normals = n_in + n_out

    # Normalise each row; leave degenerate rows as zero.
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    normals = normals / safe_norms
    normals[norms.flatten() <= 1e-10] = 0.0

    return normals


def polygon_arc_lengths(points):
    """
    Arc length assigned to each vertex of a closed polygon.

    A vertex arc length is half the sum of its two adjacent edge lengths. 

    points: (N, 2) — ordered vertices
    Returns: (N,) — arc length per vertex
    """
    prev_pts = np.roll(points, +1, axis=0)
    next_pts = np.roll(points, -1, axis=0)

    len_in  = np.linalg.norm(points - prev_pts, axis=1)
    len_out = np.linalg.norm(next_pts - points, axis=1)

    return 0.5 * (len_in + len_out)




