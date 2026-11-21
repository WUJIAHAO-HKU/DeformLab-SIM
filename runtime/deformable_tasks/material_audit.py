"""Opt-in, read-only response instrumentation and explicitly separate fixtures.

Numerical reaction calibration is not real-material or hardware validation.
Fixtures are authored before native initialization, never by overwriting a
deforming trajectory. Pressure is inferred from joint force and tool momentum.
"""
import json,sys
from pathlib import Path
import numpy as np

class MaterialAudit:
    MODES=('platform','coupon','calibration','empty_motion','weight')
    def __init__(self,runtime,spec,destination):
        self.runtime=runtime;self.spec=spec;self.destination=destination
        self.mode=spec['mode'];self.rows=[];self.field_rows=[];self.previous_velocity=None
        if self.mode not in self.MODES or runtime.config['task']!='soft_squeeze':raise ValueError('Invalid material fixture')
        self.gravity=-9.81 if self.mode=='platform' else (9.81 if self.mode=='weight' else 0.)
        self.external=0.;self.mass=None
        self.work_audit=None;self.previous_nodes=None;self.previous_node_time=0.
        if type(spec.get('full_nodal_motion',False)) is not bool:raise ValueError('full_nodal_motion must be boolean')
        self.nodal_motion=None

    def prepare(self):
        from pxr import UsdPhysics
        from isaacsim.core.simulation_manager import PhysxScene
        r=self.runtime;stage=r.stage
        if self.mode!='platform':
            PhysxScene('/World/PhysicsScene').set_gravity([0.,0.,self.gravity])
            # Common to all compared backends. Small collision skin permits
            # a low-strain frictionless coupon with measured, not commanded, strain.
            for path in ('/World/SoftA','/World/Bench','/World/Press/Tool/Shape'):
                prim=stage.GetPrimAtPath(path)
                if not prim.GetAttribute('physxCollision:contactOffset').Set(5e-5):raise RuntimeError('Missing collision offset')
                if not prim.GetAttribute('physxCollision:restOffset').Set(0.):raise RuntimeError('Missing rest offset')
        if self.mode in ('coupon','weight'):
            prim=stage.GetPrimAtPath('/World/SoftA')
            for name in ('settlingDamping','settlingThreshold','sleepThreshold'):
                if not prim.GetAttribute('physxDeformableBody:'+name).Set(0.):raise RuntimeError('Missing '+name)
        if self.mode in ('calibration','empty_motion'):
            pairs=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/World/SoftA'))
            pairs.CreateFilteredPairsRel().SetTargets(['/World/Bench','/World/Press/Tool','/World/Press/Tool/Shape'])
        if self.mode=='weight':
            pairs=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/World/SoftA'))
            targets=['/World/Bench']
            if self.spec.get('free_fall',False):targets+=['/World/Press/Tool','/World/Press/Tool/Shape']
            pairs.CreateFilteredPairsRel().SetTargets(targets)
            if r.config['material']['elasticity_damping']!=0 or r.config['solver']['linear_damping']!=0:raise ValueError('Weight momentum reference requires zero damping')
        if self.mode in ('calibration','weight'):r.target=lambda i,t:r.movers[i][1].copy()

    def bind(self,root):
        import warp as wp
        from omni.physx import get_physx_interface
        from isaacsim.core.experimental.utils.backend import use_backend
        r=self.runtime;self.drive=r.drives[0];self.tool=r.movers[0][0]
        if self.mode in ('coupon','weight'):
            native=json.loads((self.destination/'native-volumes.json').read_text())[0]['settings']
            if any(abs(native[name])>1e-10 for name in ('settling_damping','settling_threshold','sleep_threshold')):raise RuntimeError('Native settling/sleep did not match reference fixture')
        self.tool_index=self.drive.link_names.index('Tool')
        sys.path.insert(0,str(root/'work/native-readback'))
        import physx_deformable_readback as readback
        handle=readback.simulation_buffer(get_physx_interface(),r.bodies[0]['path'])
        state=wp.array(ptr=handle['device_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0').numpy()
        if not np.all(state[:,3]>0):raise RuntimeError('Invalid native mass')
        self.mass=1./state[:,3].astype(float)
        self.previous_nodes=r.bodies[0]['rest'].astype(float).copy()
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            self.tool_mass=float(self.tool.get_masses().numpy().ravel()[0])
            q=self.tool.get_world_poses()[1].numpy()[0]
            self.previous_velocity=self.tool.get_velocities()[0].numpy()[0].astype(float)
        if not np.allclose(np.abs(q),[1,0,0,0],atol=1e-6):raise RuntimeError('Fixture force frame must remain world-aligned')
        if abs(self.tool_mass-.2)>1e-6:raise RuntimeError('Unexpected tool mass')
        (self.destination/'audit-fixture.json').write_text(json.dumps(dict(mode=self.mode,spec=self.spec,gravity_z=self.gravity,
            tool_mass_kg=self.tool_mass,native_soft_mass_kg=float(self.mass.sum()),
            force_convention='F_contact_on_tool = m_tool * delta_v/dt - incoming_joint_force - m_tool*g - applied_external_force; joint child frame is world aligned.',
            scope='Numerical fixture; no experimental material or physical sensor data.'),indent=2)+'\n')
        self.previous_momentum=np.zeros(3)
        if self.spec.get('articulation_work_audit',False):
            from .articulation_work_audit import ArticulationWorkAudit
            self.work_audit=ArticulationWorkAudit(r,self.mode,self.destination)
        if self.spec.get('full_nodal_motion',False):
            from .nodal_motion_audit import NodalMotionRecorder
            self.nodal_motion=NodalMotionRecorder(self.destination,self.mass,r.config['solver']['hz'],
                round(r.config['duration_s']*r.config['solver']['hz']))
            self.capture_motion(0,'initial')

    def capture_motion(self,step,phase):
        if self.nodal_motion is None:return
        from isaacsim.core.experimental.utils.backend import use_backend
        b=self.runtime.bodies[0]
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            p=b['view'].get_nodal_positions()[0].numpy()[0]
            v=b['view'].get_nodal_velocities().numpy()[0]
        self.nodal_motion.capture(step,phase,p,v)

    def before_step(self):
        from isaacsim.core.experimental.utils.backend import use_backend
        t=(self.runtime.steps+.5)/self.runtime.config['solver']['hz']
        self.external=([0.,1.,-1.,5.,-5.,0.,0.,0.][min(int(t/.5),7)] if self.mode=='calibration' else 0.)
        if self.mode=='calibration':
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                self.tool.apply_forces_and_torques_at_pos(forces=[[0.,0.,self.external]],local_frame=False)

    def after_step(self,admissibility):
        from isaacsim.core.experimental.utils.backend import use_backend
        if self.work_audit:self.work_audit.after_step()
        r=self.runtime;hz=r.config['solver']['hz'];t=r.steps/hz
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            f,torque=self.drive.get_link_incoming_joint_force(link_indices=[self.tool_index])
            joint=f.numpy()[0,0].astype(float);v=self.tool.get_velocities()[0].numpy()[0].astype(float)
            projected=self.drive.get_dof_projected_joint_forces().numpy()[0]
        a=(v-self.previous_velocity)*hz;self.previous_velocity=v
        contact=self.tool_mass*a-joint-np.array([0.,0.,self.tool_mass*self.gravity+self.external])
        b=r.bodies[0];x=b['last']
        step_position_velocity=(x-self.previous_nodes)*hz
        self.previous_nodes=x.astype(float).copy()
        momentum=np.zeros(3);balance_reference=np.zeros(3)
        nv=None
        if self.mode=='weight':
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):nv=b['view'].get_nodal_velocities().numpy()[0]
            momentum=np.sum(self.mass[:,None]*nv,axis=0)
            balance_reference=np.array([0.,0.,self.mass.sum()*self.gravity])-(momentum-self.previous_momentum)*hz
            self.previous_momentum=momentum.copy()
        self.rows.append([t,*joint,*v,*a,*contact,self.external,*projected,*momentum,*balance_reference])
        if r.steps%max(1,hz//60)==0 or r.error:
            b=r.bodies[0];x=b['last'];rest=b['rest'];z0=rest[:,2]
            top=np.isclose(z0,z0.max(),atol=1e-7);bottom=np.isclose(z0,z0.min(),atol=1e-7)
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                if nv is None:nv=b['view'].get_nodal_velocities().numpy()[0]
                q=self.tool.get_world_poses()[1].numpy()[0]
            if not np.allclose(np.abs(q),[1,0,0,0],atol=1e-6):raise RuntimeError('Force frame rotated')
            J=r.geom.signed_tet_volumes(x,b['sim_elements'])/b['sim_restv']
            stats=admissibility.bodies[0]['stats'].numpy() if admissibility.enabled else np.zeros(8)
            self.field_rows.append([t,float(x[top,2].mean()),float(x[bottom,2].mean()),
                float(np.average(J,weights=b['sim_restv'])),float(J.min()),float(.5*np.sum(self.mass[:,None]*nv**2)),
                *np.sum(self.mass[:,None]*nv,axis=0),float(np.max(np.linalg.norm(nv,axis=1))),float(stats[0]),float(stats[3]),
                float(np.max(np.linalg.norm(step_position_velocity,axis=1))),float(np.sqrt(np.mean((step_position_velocity-nv)**2)))])

    def finish(self):
        if self.work_audit:self.work_audit.finish()
        columns=['time_s','joint_x_N','joint_y_N','joint_z_N','tool_vx','tool_vy','tool_vz','tool_ax','tool_ay','tool_az',
            'contact_x_N','contact_y_N','contact_z_N','external_z_N','dof_force_0','dof_force_1','dof_force_2',
            'weight_soft_px','weight_soft_py','weight_soft_pz','weight_reference_x_N','weight_reference_y_N','weight_reference_z_N']
        fields=['time_s','top_z_m','bottom_z_m','mean_J','min_J','kinetic_energy_J','momentum_x','momentum_y','momentum_z','max_speed_m_s','constraint_updates','projection_kinetic_work_J','step_position_speed_m_s','step_velocity_difference_rms_m_s']
        raw=np.asarray(self.rows);field=np.asarray(self.field_rows)
        if len(raw) and not np.isfinite(raw).all():raise RuntimeError('Nonfinite force trace')
        np.savez_compressed(self.destination/'material-audit.npz',samples=raw,columns=np.asarray(columns),fields=field,field_columns=np.asarray(fields),native_mass_kg=self.mass)
        result=dict(mode=self.mode,samples=len(raw),fields=len(field),force_reference='independent numerical validation pending',experimental_validation=False)
        if self.nodal_motion is not None:result['nodal_motion']=self.nodal_motion.finish()
        return result
