# abm/geometry.py
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
    axis: (2,) – firection vector, normalised internally

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
    2D perpendicular (90° CCW rotation) of a vector.
    Used to build the lateral axis from the flow axis.
    """
    return np.array([-vec[1], vec[0]])
####

def axial_projection(pos, origin, axis, radius):
    vec = pos - origin
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    dist = np.dot(vec, axis)
    return np.clip(dist/(radius + 1e-8), -1.0, 1.0)



