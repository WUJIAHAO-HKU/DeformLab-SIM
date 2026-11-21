"""Nonconflicting tetrahedron coloring for a unilateral volume constraint.

Experimental admissibility extension, not a new material calibration or an
energy-preserving integrator. No contact-plane clamping or global state reset.
"""
import numpy as np

def color_tetrahedra(elements,node_count):
    elements=np.asarray(elements,dtype=np.int32)
    if elements.ndim!=2 or elements.shape[1]!=4 or elements.min()<0 or elements.max()>=node_count:
        raise ValueError('Invalid tetrahedra')
    used=[set() for _ in range(node_count)];groups=[]
    for index,tet in enumerate(elements):
        unavailable=set().union(*(used[int(v)] for v in tet));color=0
        while color in unavailable:color+=1
        if color==len(groups):groups.append([])
        groups[color].append(index)
        for v in tet:used[int(v)].add(color)
    return [np.asarray(group,dtype=np.int32) for group in groups]

def volume_and_gradients(points):
    a,b,c=np.asarray(points,dtype=float)[1:]-points[0]
    g1=np.cross(b,c)/6;g2=np.cross(c,a)/6;g3=np.cross(a,b)/6
    return np.dot(a,np.cross(b,c))/6,np.array([-g1-g2-g3,g1,g2,g3])
