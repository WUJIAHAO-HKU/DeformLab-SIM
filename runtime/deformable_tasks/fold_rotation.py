"""Explicit rotating-edge fixture motion and measured-pose geometry.

All quaternions are wxyz; angular commands and measurements are radians.
Rigid clearance bounds concern saved poses, not continuous collision detection.
"""
import math
import numpy as np

def pose(time, motion):
    start=motion['start_s'];duration=motion['duration_s'];clearance=motion['terminal_edge_clearance_m']
    if not all(math.isfinite(x) for x in (time,start,duration,clearance)) or duration<=0 or not 0<clearance<.14:
        raise ValueError('Invalid rotating fold trajectory')
    u=min(1.,max(0.,(time-start)/duration));q=u*u*u*(10+u*(-15+6*u))
    angle=(math.pi-math.asin(clearance/.14))*q
    s=math.sin(angle);c=math.cos(angle)
    edge=np.array([0.,.14*c,.155+.14*s])
    center=edge-np.array([0.,-.0048*s,.0048*c])
    return dict(center=center,edge=edge,angle_rad=angle,orientation_wxyz=np.array([math.cos(angle/2),math.sin(angle/2),0.,0.]))

def angular_velocity(time,motion):
    # Analytic derivative of the same quintic position target, zero outside it.
    pose(time,motion)  # Keep identical finite-domain validation.
    u=min(1.,max(0.,(time-motion['start_s'])/motion['duration_s']))
    return (math.pi-math.asin(motion['terminal_edge_clearance_m']/.14))*30*u*u*(1-u)*(1-u)/motion['duration_s']

def rotation_matrix(quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-3:
        raise ValueError('Finite unit wxyz quaternion required')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def angular_error(actual, expected):
    a=rotation_matrix(actual);b=rotation_matrix(expected);r=a@b.T
    # atan2 remains accurate near zero and pi, and is invariant to q -> -q.
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return float(math.atan2(np.linalg.norm(v),np.clip((np.trace(r)-1)/2,-1,1)))

def attachment_targets(rest, initial_center, actual_center, orientation_wxyz):
    r=np.asarray(rest,float);initial=np.asarray(initial_center,float);center=np.asarray(actual_center,float)
    if r.ndim!=2 or r.shape[1]!=3 or initial.shape!=(3,) or center.shape!=(3,) or not all(np.isfinite(a).all() for a in (r,initial,center)):
        raise ValueError('Finite attachment coordinates required')
    return (r-initial)@rotation_matrix(orientation_wxyz).T+center

def rigid_clearance_bounds(center, orientation_wxyz, support=None):
    """Conservative separating-axis gap for the full 3D moving bar.

    Project the mid-section onto YZ and enlarge by its X extrusion projected
    into YZ. A positive gap proves separation; a nonpositive gap is inconclusive
    and must fail the clearance gate. Does not assume an exactly axial pose.
    """
    c=np.asarray(center,float);r=rotation_matrix(orientation_wxyz)
    if c.shape!=(3,) or not np.isfinite(c).all():raise ValueError('Finite center required')
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
    moving=c[1:]+(corners*.004)@r[1:,1:].T
    extrusion=.09*float(np.linalg.norm(r[1:,0]))
    obstacles={'fixed_bar':(np.array([-.14,.1502]),np.array([.004,.004]))}
    if support is not None:
        lo=np.asarray(support['lower_m'],float);hi=np.asarray(support['upper_m'],float)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(hi<=lo):raise ValueError('Invalid solid support')
        obstacles['support']=((lo[1:]+hi[1:])/2,(hi[1:]-lo[1:])/2)
    bounds={}
    for name,(oc,half) in obstacles.items():
        fixed=oc+corners*half
        edges=np.concatenate([np.roll(moving,-1,axis=0)-moving,np.roll(fixed,-1,axis=0)-fixed])
        lengths=np.linalg.norm(edges,axis=1);edges=edges[lengths>1e-14];lengths=lengths[lengths>1e-14]
        axes=np.stack([-edges[:,1],edges[:,0]],axis=1)/lengths[:,None]
        pa=moving@axes.T;pb=fixed@axes.T
        gap=np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0)).max()
        bounds[name]=float(gap-extrusion)
    return bounds


def author_roll_joint(stage, root):
    """Author the actual USD revolute drive after both rigid links exist."""
    from pxr import UsdPhysics
    for name in ('SlideZ','Tool'):
        link=stage.GetPrimAtPath(root+'/'+name)
        if not link.IsValid() or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Rotating fixture requires rigid '+name)
    joint=UsdPhysics.RevoluteJoint.Define(stage,root+'/DriveRollX')
    joint.CreateAxisAttr('X');joint.CreateBody0Rel().SetTargets([root+'/SlideZ'])
    joint.CreateBody1Rel().SetTargets([root+'/Tool'])
    # USD limits are degrees; tensor drive targets and readback are radians.
    joint.CreateLowerLimitAttr(-10.);joint.CreateUpperLimitAttr(190.)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(20.);drive.CreateDampingAttr(.2)
    drive.CreateTargetPositionAttr(0.)
    return joint
