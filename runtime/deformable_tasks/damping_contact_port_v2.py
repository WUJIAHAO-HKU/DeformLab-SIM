"""Read-only native terminal ports with separate joint, link and Jacobian velocity observations."""
import ctypes as C,json
import numpy as np

class DampingContactPort:
    def __init__(self,observer,runtime,destination):
        self.o=observer;self.runtime=runtime;self.destination=destination;self.records=[]
        observer.observer_damping_port_start.argtypes=[];observer.observer_damping_port_start.restype=C.c_int
        observer.observer_damping_port_read.argtypes=[C.c_ulonglong,C.POINTER(C.c_double),C.c_uint,C.POINTER(C.c_double)]
        observer.observer_damping_port_read.restype=C.c_int
        if observer.observer_damping_port_start():raise RuntimeError('Terminal contact port activation failed')
        self.path=destination/'damping-contact-port.jsonl'
        if self.path.exists():raise ValueError('Fresh port evidence required')

    def read(self,step):
        from isaacsim.core.experimental.utils.backend import use_backend
        a=np.full((1024,6),np.nan);meta=np.full(4,np.nan);ptr=C.POINTER(C.c_double)
        code=self.o.observer_damping_port_read(step,a.ctypes.data_as(ptr),1024,meta.ctypes.data_as(ptr))
        if code:raise RuntimeError('Terminal contact port read failed '+str(code))
        n=int(meta[0]);a=a[:n].copy()
        if not np.isfinite(a).all() or meta[3]!=step or not 0<=n<=1024:raise RuntimeError('Invalid terminal contact data')
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            v=self.runtime.bodies[0]['view'].get_nodal_velocities().numpy()[0]
            drive=self.runtime.drives[0];rigid=drive.get_dof_velocities().numpy()[0]
            tool=drive.link_names.index('Tool');link=drive._physics_articulation_view.get_link_velocities().numpy()[0,tool]
            jac=drive.get_jacobian_matrices().numpy()[0]
            if jac.shape[0] not in (len(drive.link_names),len(drive.link_names)-1):raise RuntimeError('Unexpected Jacobian topology')
            J=jac[tool if jac.shape[0]==len(drive.link_names) else tool-1].astype(float)
        if len(v)!=meta[2]:raise RuntimeError('Native node count changed')
        nodes=a[:,0].astype(int);top=a[:,1];sign=2*top-1
        error=float(np.abs(v[nodes,2]-a[:,4]).max(initial=0))
        rerror=float(np.abs(rigid[2]-a[:,5]).max(initial=0))
        if error!=0:raise RuntimeError('Native soft endpoint mismatch '+str(error))
        link_error=float(np.abs(link[2]-a[:,5]).max(initial=0))
        violation=None if not n else float(np.maximum(-(a[:,3]+np.maximum(a[:,2],0)/meta[1]),0).max())
        self.current=dict(step=step,inner_dt_s=float(meta[1]),contact_observation_available=bool(n),row_count=n,
            soft_velocity_readback_error_m_s=error,rigid_velocity_readback_error_m_s=rerror,
            public_link_velocity_m_s=link.astype(float).tolist(),tool_jacobian=J.tolist(),jacobian_joint_velocity_m_s=(J@rigid.astype(float)).tolist(),
            public_link_matches_native=bool(n and link_error<=1e-9),public_joint_matches_native=bool(n and rerror<=1e-9),
            native_to_public_link_error_m_s=link_error,readback_threshold_m_s=1e-9,
            before_max_velocity_violation_m_s=violation,rigid_velocity_m_s=rigid.astype(float).tolist(),rows=a.tolist())
        # Persist the observation even if a later consumer rejects the mapping.
        (self.destination/'damping-port-latest-native.json').write_text(json.dumps(self.current,allow_nan=False)+'\n')
        return a,float(meta[1]),rigid.astype(float)

    def after(self):
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            v=self.runtime.bodies[0]['view'].get_nodal_velocities().numpy()[0]
            drive=self.runtime.drives[0];rigid=drive.get_dof_velocities().numpy()[0]
            tool=drive.link_names.index('Tool');link=drive._physics_articulation_view.get_link_velocities().numpy()[0,tool]
        r=self.current;a=np.asarray(r['rows']).reshape(-1,6);nodes=a[:,0].astype(int);top=a[:,1]
        relative=(2*top-1)*(top*link[2]-v[nodes,2])
        r['after_max_velocity_violation_m_s']=None if not len(a) else float(np.maximum(-(relative+np.maximum(a[:,2],0)/r['inner_dt_s']),0).max())
        r['public_link_velocity_after_m_s']=link.astype(float).tolist()
        r['rigid_velocity_after_m_s']=rigid.astype(float).tolist()
        with self.path.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
        self.records.append(r)
