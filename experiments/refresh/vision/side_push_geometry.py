"""Independent fixed-material-front feature. Only oracle/scoring imports this."""
import hashlib
from collections import Counter
import numpy as np

SCAN_Z = np.arange(6, dtype=float)*.001+.1375

def freeze_front(rest, elements):
    rest=np.asarray(rest,dtype=np.float64);tets=np.asarray(elements,dtype=np.int64)
    if rest.ndim!=2 or rest.shape[1]!=3 or tets.ndim!=2 or tets.shape[1]!=4:
        raise ValueError('Expected native tetrahedral body')
    faces=Counter(tuple(sorted(t[[i,j,k]])) for t in tets for i,j,k in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)))
    tol=max(1e-9,1e-6*float(np.ptp(rest,axis=0).max()))
    front=[f for f,n in faces.items() if n==1 and np.all(abs(rest[list(f),1]-rest[:,1].min())<=tol)]
    if len(front)<2:raise ValueError('No frozen front-face triangles')
    return dict(triangles=[[int(i)for i in f]for f in sorted(front)],rest_sha256=hashlib.sha256(rest.tobytes()).hexdigest(),
        elements_sha256=hashlib.sha256(tets.tobytes()).hexdigest(),scan_z_m=SCAN_Z.tolist(),
        minimum_valid_lines=5,tolerance_m=1e-10,definition='median of rightmost x intersections of fixed rest-front material triangles with fixed world-z scanlines')

def leading_feature(nodes, feature, *, sample_time_s=0., sequence=0):
    p=np.asarray(nodes,dtype=np.float64);tri=p[np.asarray(feature['triangles'],int)]
    if not np.isfinite(p).all():raise ValueError('Nonfinite geometry')
    tol=feature['tolerance_m'];lines=[]
    for z in feature['scan_z_m']:
        hits=[]
        for t in tri:
            for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
                if abs(a[2]-z)<=tol:hits.append(a)
                if abs(b[2]-z)<=tol:hits.append(b)
                if (a[2]-z)*(b[2]-z)<0:
                    hits.append(a+(b-a)*((z-a[2])/(b[2]-a[2])))
        # Coincident edges may produce repeated hits; max is invariant to these.
        hit=max(hits,key=lambda v:v[0]) if hits else None
        lines.append(dict(z_m=z,valid=hit is not None,point_world=None if hit is None else hit.tolist()))
    values=[x['point_world'][0]for x in lines if x['valid']]
    valid=len(values)>=feature['minimum_valid_lines']
    return dict(observer='material_front_truth',valid=valid,position_m=float(np.median(values))if valid else None,
        reason='ok'if valid else 'insufficient_fixed_height_intersections',lines=lines,
        sample_time_s=float(sample_time_s),sequence=int(sequence),
        scope='truth-state diagnostic reference; visibility is independently audited, never used to segment images')

def support_margin(nodes):
    """Whole-body conservative footprint margin; negative is out of domain."""
    p=np.asarray(nodes,float)
    return float(np.min(.08-abs(p[:,:2])))

def point_box_clearance(nodes, center, half=(.04,.04,.01)):
    """Axis-aligned fixture only; caller must verify actual quaternion."""
    q=abs(np.asarray(nodes,float)-np.asarray(center,float))-np.asarray(half,float)
    return float(np.min(np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)))
