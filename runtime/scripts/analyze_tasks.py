"""Offline interaction checks; simulation success alone is not contact validation."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def valid_pair(cases,names):
    return all(cases[n]['completed_duration'] and all(cases[n]['physics_checks'].values()) for n in names)

def crossing_pairs(points,faces,return_pairs=False):
    """Nonadjacent triangle crossings (excludes coplanar/tangential contact).

    This detects penetration events, not a proof of global collision freedom.
    Broad phase uses bounding spheres then boxes; narrow phase segment/triangle.
    """
    tri=points[faces];centers=tri.mean(1);r=np.linalg.norm(tri-centers[:,None],axis=2).max(1)
    pairs=cKDTree(centers).query_pairs(float(2*r.max()+1e-9),output_type='ndarray')
    if not len(pairs):return np.empty((0,2),dtype=int) if return_pairs else 0
    i,j=pairs.T
    keep=~(faces[i,:,None]==faces[j,None,:]).any(axis=(1,2))
    keep&=(tri[i].max(1)>=tri[j].min(1)-1e-8).all(1)&(tri[j].max(1)>=tri[i].min(1)-1e-8).all(1)
    i,j=i[keep],j[keep]
    if not len(i):return np.empty((0,2),dtype=int) if return_pairs else 0
    def intersect(a,b,t):
        d=b-a;e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h)
        valid=np.abs(det)>1e-13;inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
        s=a-t[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1)
        v=inv*np.einsum('ij,ij->i',d,q);tt=inv*np.einsum('ij,ij->i',e2,q)
        return valid&(u>1e-6)&(v>1e-6)&(u+v<1-1e-6)&(tt>1e-6)&(tt<1-1e-6)
    hit=np.zeros(len(i),dtype=bool)
    for a,b in [(i,j),(j,i)]:
        for e in range(3):hit|=intersect(tri[a,e],tri[a,(e+1)%3],tri[b])
    return np.column_stack((i[hit],j[hit])) if return_pairs else int(hit.sum())

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
    result={'scope':'Geometric and causal task checks, not real-material validation','cases':{},'paired_controls':{}}
    for folder in sorted(root.iterdir()):
        if not (folder/'trajectory.npz').exists():continue
        m=json.loads((folder/'measurements.json').read_text());z=np.load(folder/'trajectory.npz');t=z['time_s'];x=z['nodes_0']
        info={'physics_checks':m['checks'],'completed_duration':m['completed_duration']}
        if m['task']=='cloth_grip':
            n=m['config']['geometry']['cloth_resolution'];top=x[:,-(n+1):,2].mean(1)
            hold=(t>1.8)&(t<2.5);release=t>3.5
            if hold.any() and release.any():
                held=float(np.median(top[hold]));released=float(np.median(top[release]))
                info.update(held_edge_z_m=held,released_edge_z_m=released,
                            supported_before_release=held>.43,release_drop_m=held-released,release_drop_passed=held-released>.05)
        elif m['task']=='cloth_fold':
            indices=np.unique(np.linspace(0,len(t)-1,min(len(t),12)).astype(int))
            counts=[crossing_pairs(x[k],z['faces_0']) for k in indices]
            info.update(sampled_triangle_crossings=counts,max_sampled_crossings=max(counts),sampled_self_intersection_free=max(counts)==0,
                        limitation='12 sampled frames; proper crossings only, coplanar overlap and between-frame tunnelling not certified')
        elif m['task']=='multi_soft':
            xb=z['nodes_1'];dz=xb[-10:,:,2].mean()-x[-10:,:,2].mean()
            info.update(final_centroid_separation_z_m=float(dz),stack_supported=bool(dz>m['config']['geometry']['size_m']*.65))
        elif m['task']=='soft_squeeze':
            height=np.ptp(x[:,:,2],axis=1);q=(t>2)&(t<2.6)
            if q.any():
                info.update(compressed_height_m=float(np.median(height[q])),measured_compression_fraction=float(1-np.median(height[q])/m['config']['geometry']['size_m']))
            info['noninverted_throughout']=m['minimum_volume_ratio']>0
        result['cases'][folder.name]=info
    cases=result['cases']
    if all(k in cases and 'held_edge_z_m' in cases[k] for k in ('cloth_grip','cloth_grip_frictionless')):
        difference=cases['cloth_grip']['held_edge_z_m']-cases['cloth_grip_frictionless']['held_edge_z_m']
        valid=valid_pair(cases,('cloth_grip','cloth_grip_frictionless'))
        result['paired_controls']['friction_support']={'height_difference_m':difference,'effect_observed':difference>.05,'valid_runs':valid,'passed':valid and difference>.05}
    if all(k in cases and 'final_centroid_separation_z_m' in cases[k] for k in ('multi_soft','multi_soft_no_contact')):
        difference=cases['multi_soft']['final_centroid_separation_z_m']-cases['multi_soft_no_contact']['final_centroid_separation_z_m']
        valid=valid_pair(cases,('multi_soft','multi_soft_no_contact'))
        result['paired_controls']['interbody_contact']={'separation_difference_m':difference,'effect_observed':difference>.03,'valid_runs':valid,'passed':valid and difference>.03}
    if all(k in cases for k in ('cloth_fold','cloth_fold_no_self_contact')):
        on=cases['cloth_fold']['max_sampled_crossings'];off=cases['cloth_fold_no_self_contact']['max_sampled_crossings']
        valid=valid_pair(cases,('cloth_fold','cloth_fold_no_self_contact'))
        result['paired_controls']['self_contact']={'on_crossings':on,'off_crossings':off,'valid_runs':valid,'passed':valid and on==0 and off>0,'note':'No difference or incomplete/invalid runs are inconclusive, not a pass.'}
    (root/'interaction-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
