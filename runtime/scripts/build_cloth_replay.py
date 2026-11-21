"""Create an inspectable USD replay from measured physics, never from video pixels."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Gf,Vt

def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def transform(pos,quat):
    if isinstance(quat,(Gf.Quatf,Gf.Quatd)):quat=np.r_[quat.GetReal(),quat.GetImaginary()]
    matrix=np.eye(4);matrix[:3,:3]=rotation(quat);matrix[:3,3]=pos
    return matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory.resolve()
    verification=json.loads((root/'interior-verification.json').read_text());assert verification['passed']
    data=np.load(root/'trajectory.npz');x=data['cloth_nodes'];faces=data['faces'];fps=verification['fps']
    source=Usd.Stage.Open(str(root/'initial.usda'))
    robot=source.GetPrimAtPath('/World/Franka')
    assert np.allclose(UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),np.eye(4))
    joints=[]
    for prim in Usd.PrimRange(robot):
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim);parent=joint.GetBody0Rel().GetTargets();child=joint.GetBody1Rel().GetTargets()
        if not parent or not child:continue
        kind='rotation' if prim.IsA(UsdPhysics.RevoluteJoint) else 'translation' if prim.IsA(UsdPhysics.PrismaticJoint) else 'fixed'
        joints.append(dict(name=prim.GetName(),parent=str(parent[0]),child=str(child[0]),kind=kind,
            axis=prim.GetAttribute('physics:axis').Get() if kind!='fixed' else None,
            a=transform(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()),
            b=transform(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())))
    names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    destination=root/'replay.usda';assert not destination.exists()
    replay=Usd.Stage.CreateNew(str(destination));replay.GetRootLayer().subLayerPaths=['initial.usda']
    replay.SetTimeCodesPerSecond(fps);replay.SetFramesPerSecond(fps);replay.SetStartTimeCode(0);replay.SetEndTimeCode(len(x)-1)
    replay.SetMetadata('customLayerData',{'mode':'Recorded 3D replay of measured GPU dynamics; physics disabled',
        'trajectory':'trajectory.npz','robot_pose_source':'Forward kinematics from measured joints; measured hand pose overrides FK',
        'simulation_entry':'scripts/tasks.py interior-video'})
    # Remove physics APIs in this overlay so playback cannot overwrite recorded states.
    for prim in list(replay.Traverse()):
        if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene) or 'Attachment' in prim.GetTypeName():
            prim.SetActive(False);continue
        applied=prim.GetMetadata('apiSchemas')
        if not applied:continue
        schemas=applied.GetAppliedItems()
        if any(name.startswith(('Physics','Physx','OmniPhysics')) for name in schemas):
            prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit([name for name in schemas if not name.startswith(('Physics','Physx','OmniPhysics'))]))
    replay.GetPrimAtPath('/World/Cloth').SetActive(False)
    mesh=UsdGeom.Mesh(replay.GetPrimAtPath('/World/ClothDisplay'))
    ops={}
    def set_pose(path,matrix,index):
        if path not in ops:
            xf=UsdGeom.Xformable(replay.GetPrimAtPath(path));xf.ClearXformOpOrder();ops[path]=xf.AddTransformOp(opSuffix='recorded')
        value=Gf.Matrix4d(matrix.T.tolist())
        if index==0:ops[path].Set(value)
        ops[path].Set(value,index)
    max_hand_fk_error=0.
    for i in range(len(x)):
        positions=x[i];q=dict(zip(names,data['joint_positions'][i]));poses={'/World/Franka/panda_link0':np.eye(4)}
        for _ in range(len(joints)+1):
            changed=False
            for joint in joints:
                if joint['parent'] not in poses or joint['child'] in poses:continue
                movement=np.eye(4);value=q.get(joint['name'],0.)
                if joint['kind']!='fixed':
                    axis={'X':0,'Y':1,'Z':2}[joint['axis']]
                    if joint['kind']=='rotation':movement[:3,:3]=rotation(np.r_[np.cos(value/2),np.eye(3)[axis]*np.sin(value/2)])
                    else:movement[axis,3]=value
                poses[joint['child']]=poses[joint['parent']]@joint['a']@movement@np.linalg.inv(joint['b']);changed=True
            if not changed:break
        hand=data['hand_poses'][i]
        max_hand_fk_error=max(max_hand_fk_error,float(np.linalg.norm(poses['/World/Franka/panda_hand'][:3,3]-hand[:3])))
        poses['/World/Franka/panda_hand']=transform(hand[:3],hand[3:])
        for path,matrix in poses.items():set_pose(path,matrix,i)
        camera=data['camera_poses'][i];set_pose('/World/Camera',transform(camera[:3],camera[3:]),i)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions),i)
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([positions.min(0),positions.max(0)],np.float32)),i)
        tri=positions[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=np.zeros_like(positions)
        for corner in range(3):np.add.at(normals,faces[:,corner],normal)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals),i)
        if i==0:
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([positions.min(0),positions.max(0)],np.float32)))
    mesh.SetNormalsInterpolation('vertex')
    assert max_hand_fk_error<.003,max_hand_fk_error
    replay.GetRootLayer().Save()
    reopened=Usd.Stage.Open(str(destination));reloaded=UsdGeom.Mesh(reopened.GetPrimAtPath('/World/ClothDisplay'))
    errors=[]
    for i in (0,len(x)//2,len(x)-1):errors.append(float(np.max(np.abs(np.asarray(reloaded.GetPointsAttr().Get(i))-x[i]))))
    assert max(errors)<1e-7
    active_physics=[]
    for prim in reopened.Traverse():
        schemas=prim.GetMetadata('apiSchemas')
        if schemas and any(n.startswith(('Physics','Physx','OmniPhysics')) for n in schemas.GetAppliedItems()):active_physics.append(str(prim.GetPath()))
    assert not active_physics,active_physics
    result=dict(passed=True,file=str(destination),frames=len(x),fps=fps,cloth_node_readback_errors_m=errors,
                max_hand_forward_kinematics_error_m=max_hand_fk_error,active_physics_prims=active_physics,
                scope='USD replay from recorded cloth vertices, measured hand poses and joint FK. Free camera inspection is supported by scene geometry. Live simulation uses the recording command, not this replay.')
    (root/'replay-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
