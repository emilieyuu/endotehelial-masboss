# abm/geometry.py

import numpy as np

def project_points(pos, centre, axis):
    """
    Signed scalar projection of (point-origin) onto axis.

    Returns distance along axis (dx_mag). 
        dx_mag > 0: downstream
        dx_mag < 0: upstream
        dx_mag = 0: lateral
    """
    return float(np.dot(pos - centre, axis))

def project_all(positions, centre, axis): 
    """
    Project a list/array of points onto axis. 

    Return list of scalars (dx_mag for each point)
    """
    return [project_points(p, centre, axis) for p in positions]

def max_abs_projection(projections, eps=1e-8):
    """
    Maximum absolute projection. 
    eps: prevent division by zero during normalisation
    """
    return max(abs(p) for p in projections) + eps

def perpendicular(vec):
    """
    2D perpendicular vector to given vector. 
    """
    return np.array([-vec[1], vec[0]])

# ------------------------------------------------------------------
# Projection utilities
# ------------------------------------------------------------------
def project_positions(positions, direction):
    """ Return projection of positions onto the axis defined by direction. """
    direction = direction / np.linalg.norm(direction)
    return positions @ direction

def length_along_axis(positions, direction):
    """ Measure length of cell along an axis. """
    proj = project_positions(positions, direction)
    return proj.max() - proj.min()

def get_pole_indices(positions, direction):
    """ Get indices of most upstream and downstream nodes. """
    proj = project_positions(positions, direction)
    return int(np.argmin(proj)), int(np.argmax(proj))

def axis_alignment(normal, direction):
    """ Return the angle (cos theta) between a normal vector and direction vector"""
    direction = direction / np.linalg.norm(direction)
    return abs(np.dot(normal, direction))

# ------------------------------------------------------------------
# Node-wise helpers
# ------------------------------------------------------------------
def axial_position(pos, centroid, direction):
    """ Signed positions along flow axis relative to centroid. """
    direction = direction / np.linalg.norm(direction)
    return np.dot(pos - centroid, direction)

def axial_sign(pos, centroid, direction):
    """ Assigns +1 to downstream, and -1 to upstream. """
    return np.sign(axial_position(pos, centroid, direction))

def axial_weight(pos, centroid, direction, half_length):
    """ Quadratic weighing of position depending on distance fom centre. """
    dx = axial_position(pos, centroid, direction)
    return (abs(dx) / (half_length + 1e-8))**2

