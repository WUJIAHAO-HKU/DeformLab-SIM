"""Opt-in read-only kernel observations for the pinned single-body polar path."""
import ctypes,json
import numpy as np

def validate_steps(spec,backend,nsteps):
    steps=spec.get('kernel_stage_steps',[])
    if not isinstance(steps,list) or len(steps)>8 or any(type(s) is not int or not 1<=s<=nsteps for s in steps) or len(set(steps))!=len(steps):
        raise ValueError('Kernel stage steps must be at most eight unique in-horizon integers')
    if steps and backend not in ('cp-polar','cp-split-damping','cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):raise ValueError('Kernel stage audit requires a supported polar path')
    if type(spec.get('contact_work_audit',False)) is not bool:raise ValueError('contact_work_audit must be boolean')
    if spec.get('contact_work_audit',False) and not steps:raise ValueError('Contact work audit requires selected kernel steps')
    if type(spec.get('articulation_work_audit',False)) is not bool:raise ValueError('articulation_work_audit must be boolean')
    if spec.get('articulation_work_audit',False) and (not steps or not spec.get('contact_work_audit',False)):raise ValueError('Articulation stage work requires contact audit and selected steps')
    if type(spec.get('drive_launch_capture',False)) is not bool:raise ValueError('drive_launch_capture must be boolean')
    if spec.get('drive_launch_capture',False) and (backend!='cp-contact-joint-terminal-rank-one' or spec.get('mode')!='coupon' or not spec.get('articulation_work_audit')):
        raise ValueError('Drive launch capture requires the fully observed rank-one coupon')
    if type(spec.get('drive_contact_coupling',False)) is not bool:raise ValueError('drive_contact_coupling must be boolean')
    if type(spec.get('root_position_residual',False)) is not bool:raise ValueError('root_position_residual must be boolean')
    if spec.get('root_position_residual',False) and not spec.get('drive_contact_coupling',False):
        raise ValueError('Root residual requires fully observed contact-drive candidate')
    if spec.get('drive_contact_coupling',False) and (backend!='cp-contact-joint-terminal-rank-one' or spec.get('mode')!='coupon' or not spec.get('articulation_work_audit') or spec.get('drive_launch_capture',False)):
        raise ValueError('Drive contact candidate requires independent correction observations and no launch-capture mode')
    policy=spec.get('contact_plane_policy')
    if policy is not None and (policy not in ('observe','geometry-control','complete','band-complete') or backend not in ('cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one') or spec.get('mode')!='coupon' or not steps or not spec.get('contact_work_audit',False)):
        raise ValueError('Plane coverage requires audited cp-contact-joint Cartesian coupon')
    if backend in ('cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one') and (spec.get('mode')!='coupon' or not steps or
        policy!='band-complete' or not spec.get('contact_work_audit') or not spec.get('articulation_work_audit')):
        raise ValueError('Terminal material experiment requires band-complete coupon with material/contact/articulation observations')
    return sorted(steps)

class KernelStageAudit:
    def __init__(self,runtime,adapter,spec,destination):
        self.runtime=runtime;self.dest=destination;self.observer=None;self.frames=[];self.contact=spec.get('contact_work_audit',False);self.artwork=spec.get('articulation_work_audit',False)
        self.steps=validate_steps(spec,adapter.backend,round(runtime.config['duration_s']*runtime.config['solver']['hz']))
        self.drive_contact=spec.get('drive_contact_coupling',False)
        if spec.get('root_position_residual',False)!=bool(adapter.evidence.get('root_residual',False)):
            raise RuntimeError('Root residual numerical plan and explicitly selected hook disagree')
        if self.drive_contact!=bool(adapter.evidence.get('drive_contact',False)):
            raise RuntimeError('Drive/contact numerical plan and explicitly selected hook disagree')
        self.drive_capture=spec.get('drive_launch_capture',False)
        if self.drive_capture!=bool(adapter.evidence.get('drive_launch_capture',False)):
            raise RuntimeError('Drive observation plan and explicitly selected hook disagree')
        self.plane_policy=spec.get('contact_plane_policy')
        if not self.steps:return
        o=adapter.observer
        if o is None:raise RuntimeError('Kernel audit needs active pinned hook')
        self.observer=o
        if self.plane_policy:
            rest=runtime.bodies[0]['rest'];z=rest[:,2]
            bottom=np.flatnonzero(abs(z-z.min())<1e-7).astype(np.uint32)
            top=np.flatnonzero(abs(z-z.max())<1e-7).astype(np.uint32)
            ptr=ctypes.POINTER(ctypes.c_uint)
            o.observer_plane_coverage_start.argtypes=[ctypes.c_char_p,ctypes.c_int,ptr,ctypes.c_uint,ptr,ctypes.c_uint]
            o.observer_plane_coverage_start.restype=ctypes.c_int
            o.observer_plane_coverage_stop.restype=ctypes.c_ulonglong
            o.observer_plane_coverage_errors.restype=ctypes.c_ulonglong
            o.observer_plane_coverage_frame.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double]
            o.observer_plane_coverage_frame.restype=ctypes.c_int
            policy={'observe':1,'geometry-control':2,'complete':3,'band-complete':4}[self.plane_policy]
            if self.plane_policy=='band-complete':
                drive=runtime.drives[0]
                if drive.dof_names!=['DriveX','DriveY','DriveZ']:raise RuntimeError('Guard requires pinned Cartesian DOF order')
                offsets=[]
                for path in ('/World/SoftA','/World/Bench','/World/Press/Tool/Shape'):
                    prim=runtime.stage.GetPrimAtPath(path)
                    offsets.append(float(prim.GetAttribute('physxCollision:contactOffset').Get()))
                    if float(prim.GetAttribute('physxCollision:restOffset').Get())!=0.:raise RuntimeError('Guard requires zero rest offsets')
                if not np.allclose(offsets,5e-5,atol=1e-11,rtol=0):raise RuntimeError('Guard collision offsets mismatch')
                self.plane_band=1e-4
                for name in ('observer_plane_guard_frame','observer_plane_guard_end'):
                    getattr(o,name).argtypes=[ctypes.c_double,ctypes.c_double];getattr(o,name).restype=ctypes.c_int
            if o.observer_plane_coverage_start(str(destination/'plane-coverage').encode(),policy,bottom.ctypes.data_as(ptr),len(bottom),top.ctypes.data_as(ptr),len(top)):
                raise RuntimeError('Plane coverage activation failed')
            (destination/'plane-coverage-fixture.json').write_text(json.dumps(dict(policy=self.plane_policy,bottom_nodes=bottom.tolist(),top_nodes=top.tolist(),
                contact_band_m=getattr(self,'plane_band',None),
                scope='Opt-in pinned Cartesian coupon. Native storage count retained; insufficient capacity or unsupported geometry rejected. Band policy guards omitted nodes of existing pairs at native kernel boundaries. No continuous collision or missing-pair guarantee. No position projection or force tuning.'),indent=2)+'\n')
        for name in ('observer_trace_start','observer_angular_start'):
            getattr(o,name).argtypes=[ctypes.c_char_p];getattr(o,name).restype=ctypes.c_int
        o.observer_trace_step.argtypes=[ctypes.c_int];o.observer_audit_step.argtypes=[ctypes.c_int]
        (destination/'material-kernels.tsv.inputs').mkdir()
        if o.observer_trace_start(str(destination/'material-kernels.tsv').encode()):raise RuntimeError('Trace activation failed')
        if o.observer_angular_start(str(destination/'material-root-states.f32').encode()):raise RuntimeError('Root capture activation failed')
        o.observer_audit_step(self.steps[0])
        if self.contact:
            o.observer_contact_coupon_start.argtypes=[ctypes.c_char_p];o.observer_contact_coupon_start.restype=ctypes.c_int
            o.observer_contact_step.argtypes=[ctypes.c_int];o.observer_contact_stop.restype=ctypes.c_ulonglong
            if o.observer_contact_coupon_start(str(destination/'contact-observation').encode()):raise RuntimeError('Coupon contact capture activation failed')
            if self.artwork:
                o.observer_articulation_work_start.argtypes=[ctypes.c_char_p];o.observer_articulation_work_start.restype=ctypes.c_int
                o.observer_articulation_work_stop.restype=ctypes.c_ulonglong
                if o.observer_articulation_work_start(str(destination/'articulation-observation').encode()):raise RuntimeError('Articulation stage capture activation failed')
                if self.drive_contact:
                    o.observer_drive_contact_audit_start.argtypes=[ctypes.c_char_p];o.observer_drive_contact_audit_start.restype=ctypes.c_int
                    o.observer_drive_contact_audit_stop.restype=ctypes.c_ulonglong
                    if o.observer_drive_contact_audit_start(str(destination/'drive-contact-observation').encode()):raise RuntimeError('Drive/contact correction capture activation failed')
                if self.drive_capture:
                    o.observer_drive_launch_start.argtypes=[ctypes.c_char_p];o.observer_drive_launch_start.restype=ctypes.c_int
                    o.observer_drive_launch_stop.restype=ctypes.c_ulonglong
                    if o.observer_drive_launch_start(str(destination/'drive-launch-observation').encode()):raise RuntimeError('Native drive launch capture activation failed')

    def capture(self,step,phase):
        from isaacsim.core.experimental.utils.backend import use_backend
        b=self.runtime.bodies[0]
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            p=b['view'].get_nodal_positions()[0].numpy()[0].copy()
            v=b['view'].get_nodal_velocities().numpy()[0].copy()
        self.frames.append((step,phase,p,v))

    def before(self,step):
        if self.observer and self.plane_policy:
            from isaacsim.core.experimental.utils.backend import use_backend
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                p,q=self.runtime.movers[0][0].get_world_poses();p=p.numpy()[0];q=q.numpy()[0]
            if not np.allclose(abs(q),[1,0,0,0],atol=1e-6,rtol=0):raise RuntimeError('Coverage fixture must remain world aligned')
            if self.observer.observer_plane_coverage_frame(step,float(self.runtime.table_z),float(p[2])-.01):raise RuntimeError('Coverage frame setup failed')
            if self.plane_policy=='band-complete':
                with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                    qz=float(self.runtime.drives[0].get_dof_positions().numpy()[0,2])
                if self.observer.observer_plane_guard_frame(qz,self.plane_band):raise RuntimeError('Coverage guard setup failed')
        if self.observer and step in self.steps:
            self.capture(step,'before');self.observer.observer_trace_step(step)
            if self.contact:self.observer.observer_contact_step(step)

    def after(self,step):
        if self.observer and self.plane_policy and self.observer.observer_plane_coverage_errors():
            raise RuntimeError(f'Native contact coverage rejected frame {step}; partial trajectory is invalid')
        if self.observer and self.plane_policy=='band-complete':
            from isaacsim.core.experimental.utils.backend import use_backend
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                upper=float(self.runtime.movers[0][0].get_world_poses()[0].numpy()[0,2])-.01
                qz=float(self.runtime.drives[0].get_dof_positions().numpy()[0,2])
            if self.observer.observer_plane_guard_end(upper,qz):raise RuntimeError(f'Native contact guard rejected frame {step}')
        if self.observer and step in self.steps:
            self.observer.observer_trace_step(-1);self.capture(step,'after')
            if self.contact:self.observer.observer_contact_step(-1)

    def finish(self):
        if self.observer:self.observer.observer_trace_step(-1)
        if not self.steps:return {}
        np.savez_compressed(self.dest/'material-stage-frames.npz',steps=[f[0] for f in self.frames],
            phases=[f[1] for f in self.frames],positions=[f[2] for f in self.frames],velocities=[f[3] for f in self.frames])
        result=dict(requested_steps=self.steps,recorded_boundaries=len(self.frames),
            scope='Intrusive read-only synchronization. Native physics step endpoints precede optional split damping. No performance claim.')
        if self.drive_contact:
            count=int(self.observer.observer_drive_contact_audit_stop())
            expected=2*len(self.steps)*(self.runtime.config['solver']['position_iterations']+1)
            result['drive_contact_coupling']=dict(records=count,expected_records=expected)
            if count!=expected:raise RuntimeError('Incomplete contact/drive correction observations')
        if self.drive_capture:
            count=int(self.observer.observer_drive_launch_stop())
            expected=2*len(self.steps)*(self.runtime.config['solver']['position_iterations']+1)
            result['drive_launch_capture']=dict(records=count,expected_records=expected)
            if count!=expected:raise RuntimeError('Incomplete native drive launch observations')
        if self.artwork:result['articulation_records']=self.observer.observer_articulation_work_stop()
        if self.contact:result['contact_records']=self.observer.observer_contact_stop()
        if self.plane_policy:result['plane_coverage']=dict(policy=self.plane_policy,prepares=int(self.observer.observer_plane_coverage_stop()))
        (self.dest/'material-stage-audit.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
