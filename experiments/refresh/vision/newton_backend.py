"""Pinned Newton VBD articulated fixture; no recorded-motion playback.

An independent solver implementation, not a reproduction of IsaacIPC/GMCP.
Joint force limits are unsupported by this pinned VBD version. The paired
native protocol therefore uses uncapped drives too; original72 is untouched.
"""
from pathlib import Path
import hashlib, inspect, importlib.metadata, json, time
import numpy as np
import warp as wp
import newton


class Backend:
    def __init__(self, fixture, settings, device='cuda:0'):
        f=fixture; self.settings=settings; self.dt=1./f['hz']; self.t=0.
        self.rest=np.asarray(f['rest'],np.float32);self.tets=np.asarray(f['elements'],np.int32)
        self.initial=np.asarray(f['tool_initial'],np.float32)
        self.gauge_spec=f['gauge'];self.hz=f['hz']
        m=f['material'];e=m['young_pa'];nu=m['poisson']
        b=newton.ModelBuilder(up_axis=newton.Axis.Z,gravity=-9.81)
        b.add_soft_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0,0,0),
            vertices=self.rest.tolist(),indices=self.tets.ravel().tolist(),density=m['density'],
            k_mu=e/(2*(1+nu)),k_lambda=e*nu/((1+nu)*(1-2*nu)),k_damp=0.,
            tri_ke=0.,tri_ka=0.,tri_kd=0.,edge_ke=0.,edge_kd=0.,particle_radius=0.,validate_mesh=True)
        cfg=newton.ModelBuilder.ShapeConfig(density=0.,mu=m['friction'],ke=settings['contact_ke'],kd=0.,margin=0.,gap=0.)
        g=f['geometry']
        b.add_shape_box(-1,xform=wp.transform(g['bench_position_m'],wp.quat_identity()),
            hx=g['bench_half_extents_m'][0],hy=g['bench_half_extents_m'][1],hz=g['bench_half_extents_m'][2],cfg=cfg,label='FiniteSupport')
        b.add_shape_box(-1,xform=wp.transform([0.,0.,-.08],wp.quat_identity()),hx=10.,hy=10.,hz=.05,cfg=cfg,label='Floor')
        inertia=wp.mat33(.001,0,0,0,.001,0,0,0,.001)
        # The native base is fixed. Omitting its inertial DOFs is exact for
        # the moving links; 1kg SlideX, 1kg SlideY, .2kg Tool are retained.
        parent=-1;joints=[];self.links=[]
        for i,(name,mass,axis) in enumerate([('SlideX',1.,(1.,0,0)),('SlideY',1.,(0,1.,0)),('Tool',.2,(0,0,1.))]):
            link=b.add_link(xform=wp.transform(self.initial,wp.quat_identity()),mass=mass,inertia=inertia,lock_inertia=True,label=name)
            j=b.add_joint_prismatic(parent,link,parent_xform=wp.transform(self.initial,wp.quat_identity()) if i==0 else wp.transform_identity(),
                axis=wp.vec3(*axis),target_ke=1e6,target_kd=4000.,target_pos=0.,target_vel=0.,limit_lower=-.5,limit_upper=.5,label='Drive'+name)
            self.links.append(link);joints.append(j);parent=link
        b.add_articulation(joints,label='Press');self.tool=parent
        half=g['tool_half_extents_m'];b.add_shape_box(self.tool,hx=half[0],hy=half[1],hz=half[2],cfg=cfg,label='ToolShape')
        self.gauge=b.add_link(xform=wp.transform([0.,0.,f['gauge']['initial_center_z_m']],wp.quat_identity()),mass=.2,inertia=inertia,lock_inertia=True,label='Gauge')
        j=b.add_joint_prismatic(-1,self.gauge,parent_xform=wp.transform([0.,0.,f['gauge']['initial_center_z_m']],wp.quat_identity()),
            axis=wp.vec3(0,0,1),target_ke=1e7,target_kd=4000.,target_pos=0.,target_vel=0.,limit_lower=-.1,limit_upper=.01,label='GaugeZ')
        b.add_articulation([j],label='GaugeDrive')
        from gauge_geometry import bars
        for bar in bars(f['gauge']):
            half=np.asarray(bar['size'])/2
            b.add_shape_box(self.gauge,xform=wp.transform(bar['center'],wp.quat_identity()),hx=half[0],hy=half[1],hz=half[2],cfg=cfg,label=bar['name'])
        b.color();self.model=b.finalize(device=device)
        self.model.soft_contact_ke=settings['contact_ke'];self.model.soft_contact_kd=0.;self.model.soft_contact_mu=m['friction']
        self.solver=newton.solvers.SolverVBD(self.model,iterations=settings['iterations'],friction_epsilon=settings['friction_epsilon'],
            integrate_with_external_rigid_solver=False,particle_enable_self_contact=False,
            rigid_joint_linear_ke=settings.get('joint_constraint_ke',1e9),rigid_joint_angular_ke=settings.get('joint_constraint_ke',1e9),
            rigid_body_particle_contact_buffer_size=1024)
        self.pipeline=newton.CollisionPipeline(self.model,broad_phase='nxn',soft_contact_margin=.0015,
            enable_rigid_soft_full_surface_contact=settings.get('full_surface',False),deterministic=True,verify_buffers=True)
        self.contacts=self.pipeline.contacts();self.state=self.model.state();self.next=self.model.state();self.control=self.model.control()
        newton.eval_fk(self.model,self.model.joint_q,self.model.joint_qd,self.state)
        wp.copy(self.next.body_q,self.state.body_q)
        self.mass=self.model.particle_mass.numpy()
        assert np.allclose(self.mass,f['mass'],rtol=5e-5,atol=1e-9)
        assert np.array_equal(self.model.tet_indices.numpy(),self.tets)
        assert np.allclose(self.state.particle_q.numpy(),self.rest,atol=1e-8,rtol=0)
        body=self.state.body_q.numpy();assert np.allclose(body[self.tool,:3],self.initial,atol=1e-7,rtol=0),body
        self.targets=np.zeros(4,np.float32);self.target_vel=np.zeros(4,np.float32)
        self.snapshots=[];self.compute_s=[];self.steps=0
        p=Path(inspect.getfile(newton)).parent
        self.receipt=dict(newton_version=importlib.metadata.version('newton'),warp_version=wp.__version__,settings=settings,
            particle_count=self.model.particle_count,tet_count=self.model.tet_count,body_mass=self.model.body_mass.numpy().tolist(),
            body_inertia=self.model.body_inertia.numpy().tolist(),shape_body=self.model.shape_body.numpy().tolist(),
            shape_transform=self.model.shape_transform.numpy().tolist(),shape_scale=self.model.shape_scale.numpy().tolist(),
            joint_ke=self.model.joint_target_ke.numpy().tolist(),joint_kd=self.model.joint_target_kd.numpy().tolist(),
            source_sha256={str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest()for x in p.rglob('*.py')},
            boundary='Dynamically solved three-axis press and single-axis gauge; uncapped force drives in both arms.',
            material='Stable Neo-Hookean; nominal E/nu/rho matched, finite-strain PhysX equivalence not assumed.',
            self_contact=False,geometric_feasibility_theorem=False)

    def advance(self, correction, offset, count):
        from side_push_config_v3 import target_pose
        from gauge_articulation import target
        samples=[]
        for _ in range(count):
            step=self.steps+1;t=step/self.hz
            command=target_pose(t,correction,offset);gz,gv=target(t,self.gauge_spec)
            self.targets[:3]=command-self.initial;self.targets[3]=gz;self.target_vel[3]=gv
            tick=time.monotonic()
            self.control.joint_target_q.assign(self.targets);self.control.joint_target_qd.assign(self.target_vel)
            self.state.clear_forces();self.pipeline.collide(self.state,self.contacts)
            self.solver.step(self.state,self.next,self.control,self.contacts,self.dt)
            self.state,self.next=self.next,self.state;wp.synchronize_device()
            self.compute_s.append(time.monotonic()-tick);self.steps=step
            samples.append(self.sample(command))
        return samples

    def sample(self,command=None):
        q=self.state.particle_q.numpy();v=self.state.particle_qd.numpy();b=self.state.body_q.numpy();bv=self.state.body_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(b).all():raise RuntimeError('Nonfinite Newton state')
        return dict(step=self.steps,positions=q,velocities=v,tool=b[self.tool],gauge=b[self.gauge],
            gauge_velocity=bv[self.gauge],command=self.initial.copy() if command is None else command)

