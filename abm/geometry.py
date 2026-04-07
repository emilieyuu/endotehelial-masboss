# abm/geometry.py

import numpy as np

def project_onto_axis(vecs, axis):
    """
    Project vector(s) onto a given axis.

    Args:
        vecs : np.ndarray of shape (2,) or (n,2) — point(s) as vector(s)
        axis : np.ndarray of shape (2,) — axis to project onto

    Returns:
        float or np.ndarray: projection(s) along the axis
    """
    vecs = np.atleast_2d(vecs).astype(float)
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)  # unit vector along axis

    projections = vecs @ axis

    if projections.size == 1:
        return float(projections[0])
    
    return projections

def perpendicular(vec):
    """
    2D perpendicular vector to given vector. 
    """
    return np.array([-vec[1], vec[0]])

def axis_alignment(normal, direction):
    """ Return the angle (cos theta) between a normal vector and direction vector"""
    direction = direction / np.linalg.norm(direction)
    return abs(np.dot(normal, direction))

