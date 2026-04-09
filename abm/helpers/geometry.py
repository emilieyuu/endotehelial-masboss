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




