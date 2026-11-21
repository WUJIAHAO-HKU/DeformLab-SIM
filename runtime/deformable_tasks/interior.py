"""Compose real architectural meshes around a meter-scale physical robot task.

All overrides are authored in the task layer; source NVIDIA assets stay intact.
The tabletop collision uses the actual visible mesh, not a background plane.
"""
from pathlib import Path
import json

COUNTER='/World/Brownstone01/Geometry/Casework/Counter_Top_no_lip_36__Depth/Casework_463451_1313009'

def compose(stage, output):
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    root=Path(__file__).resolve().parents[1]
    source=root/'assets/architecture/brownstone-interior.usda'
    if not source.is_file():raise FileNotFoundError(source)
    stage.GetRootLayer().subLayerPaths.append(str(source))
    # Target tabletop is z=0. The robot base is 30 cm left of its center.
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    bounds=cache.ComputeWorldBound(stage.GetPrimAtPath(COUNTER)).ComputeAlignedRange()
    origin=(np.array(bounds.GetMin())+np.array(bounds.GetMax()))/2
    origin[0]-=30.;origin[2]=bounds.GetMax()[2]
    transform=Gf.Matrix4d().SetScale(.01)*Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(-origin*.01)))
    for path in ('/World/Brownstone01','/World/Brownstone01_Stair','/World/Lights','/Environment'):
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():continue
        xf=UsdGeom.Xformable(prim)
        original=xf.GetLocalTransformation()
        xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix='taskPlacement').Set(original*transform)
    for prim in stage.Traverse():
        if prim.GetTypeName()=='DistantLight':
            for key in ('color','inputs:color'):
                attr=prim.GetAttribute(key)
                if attr:attr.Set(Gf.Vec3f(1.,.96,.90))
    counter=stage.GetPrimAtPath(COUNTER)
    UsdPhysics.CollisionAPI.Apply(counter).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(counter).CreateApproximationAttr('none')
    cache.Clear()
    b=cache.ComputeWorldBound(counter).ComputeAlignedRange()
    assert abs(b.GetMax()[2])<1e-5, b
    rows=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Brownstone01'),Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);points=mesh.GetPointsAttr().Get();counts=mesh.GetFaceVertexCountsAttr().Get()
        if points is None or counts is None:continue
        world=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        rows.append(dict(path=str(prim.GetPath()),vertices=len(points),polygons=len(counts),
                         min_m=list(world.GetMin()),max_m=list(world.GetMax())))
    report=dict(source=str(source),source_units_m=.01,task_units_m=1.,source_origin_cm=origin.tolist(),
                mesh_count=len(rows),vertices=sum(r['vertices'] for r in rows),polygons=sum(r['polygons'] for r in rows),
                geometry=rows,physical_tabletop=dict(path=COUNTER,min_m=list(b.GetMin()),max_m=list(b.GetMax()),collision='actual visible triangle mesh'),
                scope='Interior walls, floor, cabinets and countertop are USD geometry. Material maps and sky illumination remain textures, not a room-sized image backdrop.')
    assert len(rows)>100 and report['vertices']>10000
    (Path(output)/'scene-geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def cloth_material(stage, visual, nx, nz, root):
    from .specimen_appearance import cloth_material as apply_textile
    return apply_textile(stage,visual,nx,nz,root)

def camera_pose(t):
    import numpy as np
    from pxr import Gf
    # Continuous dolly across a modeled room gives genuine depth-dependent parallax.
    s=float(np.clip(t/8.,0,1));s=s*s*s*(10+s*(-15+6*s))
    eye=np.array([1.65,-2.65,1.2])*(1-s)+np.array([.60,-.92,.65])*s
    target=Gf.Vec3d(*((1-s)*np.array([.23,0,.38])+s*np.array([.30,0,.41])))
    matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),target,Gf.Vec3d(0,0,1)).GetInverse()
    q=matrix.ExtractRotationQuat()
    return eye,np.r_[q.GetReal(),q.GetImaginary()]
