"""Native kinematic target publication and actual rigid-state capture."""
import numpy as np
from gauge_geometry import bars

def target_pose(t,spec):
    if not 0<=t<=4.4+1e-8:raise ValueError('Outside frozen gauge horizon')
    u=float(np.clip((t-3.2)/.8,0.,1.));u=u*u*u*(10+u*(-15+6*u))
    return np.asarray([0.,0.,spec['initial_center_z_m']+(spec['final_center_z_m']-spec['initial_center_z_m'])*u,0.,0.,0.,1.],dtype=np.float32)

def node_box_penetration(points,pose,spec):
    """Sampled nodal box intersection; does not certify surface nonintersection."""
    x=np.asarray(points,dtype=float);pose=np.asarray(pose,dtype=float)
    if not np.allclose(pose[3:],[0,0,0,1],atol=2e-6,rtol=0):raise ValueError('Gauge orientation changed')
    values=[]
    for b in bars(spec):
        q=np.abs(x-(np.asarray(b['center'])+pose[:3]))-np.asarray(b['size'])/2
        sdf=np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.)
        values.append(np.maximum(-sdf,0.))
    return np.asarray(values)

class NativeGauge:
    def __init__(self,spec,hz,interface,soft_path):
        import sys
        sys.path.insert(0,".cache/deformlabsim-gauge-target-v4")
        import gauge_target_v3 as driver
        self.driver=driver;self.interface=interface;self.soft_path=soft_path
        self.sdk_initial=driver.read(interface,soft_path)
        self.spec=spec;self.hz=hz
        self.times=[];self.commands=[];self.actual=[];self.velocities=[]
        first=np.asarray(self.sdk_initial['pose_xyzw'])
        if not np.allclose(first,target_pose(0,spec),rtol=0,atol=2e-6):raise RuntimeError('Initial native gauge pose mismatch')
    def before(self,t):
        value=target_pose(t,self.spec)
        self.driver.set_target(self.interface,self.soft_path,float(value[2]))
    def capture(self,t):
        state=self.driver.read(self.interface,self.soft_path);pose=np.asarray(state['pose_xyzw']);vel=np.asarray(state['velocity'])
        if not np.isfinite(pose).all()or not np.isfinite(vel).all():raise RuntimeError('Invalid native gauge state')
        self.times.append(t);self.commands.append(target_pose(t,self.spec));self.actual.append(pose);self.velocities.append(vel)
    def save(self,path):
        np.savez_compressed(path,time_s=self.times,commanded_pose_xyzw=self.commands,actual_pose_xyzw=self.actual,actual_velocity=self.velocities)
        return dict(samples=len(self.times),source='Independent PhysX SDK kinematic actor; every physics step',simulation_view_binding='No additional GPU tensor view; exact SDK actor identity',
            publication='PxRigidDynamic::setKinematicTarget; independent SDK state readback',sdk_initial=self.sdk_initial,
            maximum_pose_error_m=float(np.max(np.abs(np.asarray(self.actual)[:,:3]-np.asarray(self.commands)[:,:3]))),
            reaction_force_observed=False,contact_law='Moving gauge contacts are handled by native PhysX; D augments the stationary support.')

def validate_native_boxes(geo,spec):
    """Independently bind all four native SDK collider dimensions and poses."""
    selected=[]
    for b in bars(spec):
        center=np.asarray(b['center'])+[0,0,spec['initial_center_z_m']]
        matches=[s for s in geo['rigid_shapes']if bool(s['simulation_shape'])==spec.get('collisions_enabled',True)and s['is_box']and not s['is_static']
            and np.allclose(s['shape_world_pose']['position'],center,rtol=0,atol=2e-6)
            and np.allclose(s['half_extents_m'],np.asarray(b['size'])/2,rtol=0,atol=2e-7)]
        if len(matches)!=1:raise RuntimeError('Native gauge collider identity: '+b['name'])
        selected.append(dict(name=b['name'],shape=matches[0]))
    return dict(passed=True,colliders=selected,source='Independent PhysX SDK shape readback')
