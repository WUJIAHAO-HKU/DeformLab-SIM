"""Checks actual end-of-step state; never changes positions, velocities or policy."""
import numpy as np

class PublishedQuality:
    def __init__(self,rest,tets,mass,geometry,gauge_spec,hz):
        self.tet=np.asarray(tets,dtype=int);r=np.asarray(rest,dtype=float)
        d=r[self.tet[:,1:]]-r[self.tet[:,0,None]]
        self.rest_det=np.einsum('ti,ti->t',np.cross(d[:,0],d[:,1]),d[:,2])
        self.mass=np.asarray(mass,dtype=float);self.hz=hz;self.rows=[];self.previous=None
        g=geometry['oracle_geometry'];self.center=np.asarray(g['bench_position_m']);self.half=np.asarray(g['bench_half_extents_m'])
        self.spec=gauge_spec
        from gauge_geometry import bars
        self.bars=bars(gauge_spec)
    def record(self,step,x,v,gauge_pose):
        x=np.asarray(x,dtype=float);v=np.asarray(v,dtype=float);p=np.asarray(gauge_pose,dtype=float)
        finite=bool(np.isfinite(x).all()and np.isfinite(v).all()and np.isfinite(p).all())
        if not finite:raise RuntimeError('Nonfinite actual published material state')
        d=x[self.tet[:,1:]]-x[self.tet[:,0,None]]
        js=np.einsum('ti,ti->t',np.cross(d[:,0],d[:,1]),d[:,2])/self.rest_det
        inside=np.all(abs(x[:,:2]-self.center[:2])<self.half[:2],axis=1)
        support=float(np.where(inside,np.maximum(self.center[2]+self.half[2]-x[:,2],0),0).max())
        # Verify the constrained rotation, then use the measured quaternion.
        # Even its small residual is retained in micrometre contact checks.
        assert np.allclose(abs(p[3:]),[0,0,0,1],atol=2e-6,rtol=0)
        raw_q=np.asarray(gauge_pose)[3:];qx,qy,qz,qw=raw_q/np.linalg.norm(raw_q)
        rotation=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
        local=(x-p[:3])@rotation;penetration=0.
        for bar in self.bars:
            q=abs(local-bar['center'])-np.asarray(bar['size'])/2
            penetration=max(penetration,float(np.maximum(-np.max(q,axis=1),0).max()))
        c=np.asarray(self.spec['center_xy_m']);half=np.asarray(self.spec['opening_xy_m'])/2
        clearance=float(np.r_[x[:,:2].min(0)-(c-half),(c+half)-x[:,:2].max(0)].min())
        dv=np.zeros_like(v)if self.previous is None else v-(x-self.previous)*self.hz
        mismatch=float(np.linalg.norm(np.sum(self.mass[:,None]*dv,axis=0)))
        self.previous=x.copy()
        self.rows.append([step,step/self.hz,support,penetration,float(js.min()),clearance,mismatch])
    def save(self,path):
        a=np.asarray(self.rows,dtype=float)
        np.savez_compressed(path,values=a,columns=np.array(['step','time_s','support_penetration_m','gauge_penetration_m','minimum_J','aperture_clearance_m','velocity_reconstruction_momentum_kg_m_s']))
        assert len(a)>0
        physical=(a[:,2]<=5e-5)&(a[:,3]<=5e-5)&(a[:,4]>0)
        return dict(samples=len(a),end_of_native_step=True,state_intervention=False,controller_intervention=False,
            physical_invalid_sample_count=int(np.count_nonzero(~physical)),support_max_m=float(a[:,2].max()),
            gauge_max_m=float(a[:,3].max()),minimum_J=float(a[:,4].min()),
            final_minimum_clearance_m=float(a[a[:,1]>=4-1e-9,5].min())if np.any(a[:,1]>=4-1e-9)else None,
            velocity_reconstruction_momentum_max=float(a[1:,6].max())if len(a)>1 else 0.,
            velocity_metric_scope='Diagnostic of published velocity versus step displacement; not an energy or force certificate',
            quality_check_scope='Passive final-state checks; failure remains in the data. No clipping, rollback or extra solve.')
