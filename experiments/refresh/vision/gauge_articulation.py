"""GPU-native single-axis gauge drive and render publication."""
import numpy as np
from gauge_geometry import bars

def author(stage,spec,iterations):
    from pxr import Gf,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
    from isaacsim.core.experimental.prims import Articulation,RigidPrim
    path='/World/PrecisionGauge';UsdGeom.Xform.Define(stage,path)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(path))
    aa=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(path));aa.CreateSolverPositionIterationCountAttr(iterations)
    for name,mass in [('Base',.01),('Tool',.2)]:
        x=UsdGeom.Xform.Define(stage,path+'/'+name);x.AddTranslateOp().Set(Gf.Vec3d(0,0,spec['initial_center_z_m']))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim());ma=UsdPhysics.MassAPI.Apply(x.GetPrim());ma.CreateMassAttr(mass);ma.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
    fixed=UsdPhysics.FixedJoint.Define(stage,path+'/WorldAnchor');fixed.CreateBody1Rel().SetTargets([path+'/Base']);fixed.CreateLocalPos0Attr(Gf.Vec3f(0,0,spec['initial_center_z_m']))
    j=UsdPhysics.PrismaticJoint.Define(stage,path+'/DriveZ');j.CreateAxisAttr('Z');j.CreateBody0Rel().SetTargets([path+'/Base']);j.CreateBody1Rel().SetTargets([path+'/Tool']);j.CreateLowerLimitAttr(-.1);j.CreateUpperLimitAttr(.01)
    drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(1e7);drive.CreateDampingAttr(4000.);drive.CreateMaxForceAttr(1e30);drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
    material=UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(path+'/ContactMaterial','Material'));material.CreateStaticFrictionAttr(.4);material.CreateDynamicFrictionAttr(.4);material.CreateRestitutionAttr(0.)
    display=UsdGeom.Xform.Define(stage,'/World/PrecisionGaugeDisplay');display.AddTranslateOp().Set(Gf.Vec3d(0,0,spec['initial_center_z_m']))
    for b in bars(spec):
        for parent,collision in [(path+'/Tool',True),('/World/PrecisionGaugeDisplay',False)]:
            cube=UsdGeom.Cube.Define(stage,parent+'/'+b['name']);cube.CreateSizeAttr(1.);cube.AddTranslateOp().Set(Gf.Vec3d(*b['center']));cube.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*b['size']));cube.CreateDisplayColorAttr([Gf.Vec3f(.34,.36,.39)])
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(spec.get('collisions_enabled',True))
                phys=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim());phys.CreateContactOffsetAttr(.0015);phys.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material(material.GetPrim()),materialPurpose='physics')
    UsdGeom.Imageable(stage.GetPrimAtPath(path+'/Tool')).MakeInvisible()
    return (Articulation(path,reset_xform_op_properties=False),RigidPrim(path+'/Tool',reset_xform_op_properties=False),display.GetPrim()),dict(root=path,method='GPU articulation position and velocity targets; force drive',stiffness=1e7,damping=4000.,max_force_n=1e30,position_iterations=iterations,specification=spec)

def target(t,spec):
    u=float(np.clip((t-3.2)/.8,0.,1.));w=u*u*u*(10+u*(-15+6*u));delta=spec['final_center_z_m']-spec['initial_center_z_m']
    return delta*w,delta/.8*30*u*u*(1-u)*(1-u)

class NativeGauge:
    def __init__(self,spec,hz,views):
        self.spec=spec;self.hz=hz;self.drive,self.body,self.display=views;self.times=[];self.commands=[];self.actual=[];self.velocities=[]
        assert list(self.drive.dof_names)==['DriveZ'];self.idx=[0]
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            self.drive.set_dof_positions([0.],dof_indices=self.idx);self.drive.set_dof_velocities([0.],dof_indices=self.idx)
            self.drive.set_dof_position_targets([0.],dof_indices=self.idx);self.drive.set_dof_velocity_targets([0.],dof_indices=self.idx)
    def before(self,t):
        from isaacsim.core.experimental.utils.backend import use_backend
        p,v=target(t,self.spec)
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            self.drive.set_dof_position_targets([p],dof_indices=self.idx);self.drive.set_dof_velocity_targets([v],dof_indices=self.idx)
    def capture(self,t):
        from isaacsim.core.experimental.utils.backend import use_backend
        from pxr import Gf
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            pos,q=self.body.get_world_poses();pos=pos.numpy()[0].copy();q=q.numpy()[0].copy()
            v=float(self.drive.get_dof_velocities(dof_indices=self.idx).numpy().reshape(-1)[0])
            joint=float(self.drive.get_dof_positions(dof_indices=self.idx).numpy().reshape(-1)[0])
        if not np.allclose(pos,[0,0,self.spec['initial_center_z_m']+joint],rtol=0,atol=2e-6):raise ValueError('GPU link and joint state disagree')
        if not np.allclose(abs(q),[1,0,0,0],rtol=0,atol=2e-6):raise ValueError('Gauge rotated')
        pose=np.r_[pos,q[1:],q[0]];p,_=target(t,self.spec);cmd=[0,0,self.spec['initial_center_z_m']+p,0,0,0,1]
        self.times.append(t);self.commands.append(cmd);self.actual.append(pose);self.velocities.append([0,0,v,0,0,0])
        # Display publication is shared at feedback boundaries in this paired cohort.
    def save(self,path):
        np.savez_compressed(path,time_s=self.times,commanded_pose_xyzw=self.commands,actual_pose_xyzw=self.actual,actual_velocity=self.velocities)
        return dict(samples=len(self.times),source='GPU articulation link pose and independent joint-coordinate consistency, every step',publication='GPU articulation position and velocity targets; no rollout pose setters',maximum_pose_error_m=float(np.max(abs(np.asarray(self.actual)[:,:3]-np.asarray(self.commands)[:,:3]))),reaction_force_observed=False)

def validate_geometry(raw,spec):
    selected=[]
    for b in bars(spec):
        matches=[s for s in raw['rigid_shapes'] if s['actor_name']=='/World/PrecisionGauge/Tool' and s['articulation_link'] and bool(s['simulation_shape'])==spec.get('collisions_enabled',True) and np.allclose(s.get('half_extents_m',[]),np.asarray(b['size'])/2,rtol=0,atol=2e-7) and np.allclose(s['shape_local_pose']['position'],b['center'],rtol=0,atol=2e-7)]
        if len(matches)!=1:raise RuntimeError('Gauge native local collider identity: '+b['name'])
        selected.append(dict(name=b['name'],shape=matches[0]))
    return dict(passed=True,colliders=selected,source='SDK immutable local collider geometry; dynamic world pose must come from GPU')
