"""Geometric validity only; neither material accuracy nor a corrective solver."""
import numpy as np


def signed_volumes(points, elements):
    t = np.asarray(points)[np.asarray(elements)]
    return np.einsum('ij,ij->i', np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0]), t[:,3]-t[:,0]) / 6


def assess_volume(points, rest_points, elements, minimum_ratio=.001, floor_tolerance=.02):
    points, rest_points, elements = np.asarray(points), np.asarray(rest_points), np.asarray(elements)
    if points.shape != rest_points.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Volume node shape mismatch')
    if not np.isfinite(points).all() or not np.isfinite(rest_points).all():
        raise ValueError('Nonfinite volume nodes')
    if elements.ndim != 2 or elements.shape[1] != 4 or not np.issubdtype(elements.dtype, np.integer):
        raise ValueError('Invalid tetrahedral indices')
    if not len(elements) or elements.min()<0 or elements.max()>=len(points):
        raise ValueError('Tetrahedral index outside nodes')
    rest = signed_volumes(rest_points, elements)
    if np.any(rest<=0):
        raise ValueError('Nonpositive reference tet volume')
    ratio = float(np.min(signed_volumes(points,elements)/rest))
    minimum_z = float(points[:,2].min())
    return dict(valid=ratio>minimum_ratio and minimum_z>=-floor_tolerance,
                minimum_signed_volume_ratio=ratio, minimum_z_m=minimum_z,
                minimum_ratio_gate=minimum_ratio, floor_tolerance_m=floor_tolerance)
