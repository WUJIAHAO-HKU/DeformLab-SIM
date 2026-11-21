"""Initial USD fixture relocation, strictly before simulation initialization."""
import numpy as np
from side_push_config import PUSHER_INITIAL

def relocate_fixture(stage,runtime,offset,y_scale=1.):
    from pxr import UsdGeom,UsdPhysics,Gf,Vt
    if runtime.steps!=0:raise RuntimeError('Cannot relocate during rollout')
    if not np.isfinite(offset) or abs(offset)>.0025:raise ValueError('Unplanned initial offset')
    if y_scale not in (.75,1.,1.25):raise ValueError('Unplanned geometry scale')
    initial=PUSHER_INITIAL.copy();initial[0]+=offset
    old=runtime.movers[0][1].copy();updated=[];reference_authoring=[]
    for name in ('Base','SlideX','SlideY','Tool','ToolDisplay'):
        prim=stage.GetPrimAtPath('/World/Press/'+name)
        attr=prim.GetAttribute('xformOp:translate')
        if not attr or not attr.IsValid():raise RuntimeError('Missing initial transform '+name)
        before=attr.Get();attr.Set(type(before)(*[float(x)for x in initial]));updated.append(dict(prim=str(prim.GetPath()),before=list(before),after=list(attr.Get())))
    anchor=UsdPhysics.FixedJoint(stage.GetPrimAtPath('/World/Press/WorldAnchor'))
    anchor.GetLocalPos0Attr().Set(Gf.Vec3f(*initial))
    rigid,_,path=runtime.movers[0];runtime.movers[0]=(rigid,initial.astype(np.float32),path)
    for b in runtime.bodies:
        shifted=np.asarray(b['rest']).copy();shifted[:,0]+=offset
        shifted[:,1]*=y_scale
        prim=stage.GetPrimAtPath(b['path'])
        points=prim.GetAttribute('points')
        if not points or not points.IsValid():raise RuntimeError('Missing native volume points')
        native_rest=prim.GetAttribute('omniphysics:restShapePoints')
        if not native_rest or not native_rest.IsValid():raise RuntimeError('Missing native reference-shape points')
        old_native_rest=np.asarray(native_rest.Get(),dtype=np.float32)
        if old_native_rest.shape!=shifted.shape:raise RuntimeError('Native reference-shape size mismatch')
        values=Vt.Vec3fArray.FromNumpy(shifted.astype(np.float32))
        points.Set(values);native_rest.Set(values)
        bounds=Vt.Vec3fArray.FromNumpy(np.asarray([shifted.min(0),shifted.max(0)],dtype=np.float32))
        extent=prim.GetAttribute('extent')
        if not extent or not extent.IsValid():raise RuntimeError('Missing native extent')
        extent.Set(bounds)
        b['visual'].GetPointsAttr().Set(values)
        b['visual'].GetExtentAttr().Set(bounds)
        assert np.array_equal(np.asarray(points.Get()),np.asarray(native_rest.Get()))
        reference_authoring.append(dict(path=b['path'],native_rest_attribute='omniphysics:restShapePoints',before_extents_m=np.ptp(old_native_rest,axis=0).tolist(),after_extents_m=np.ptp(np.asarray(native_rest.Get()),axis=0).tolist(),authored_current_equals_reference=True))
        b['rest']=shifted
    material_attributes={}
    for path,prefix in (('/World/Looks/GripPhysics','physics:'),('/World/SoftAPhysics','physxDeformableMaterial:')):
        prim=stage.GetPrimAtPath(path)
        material_attributes[path]={str(a.GetName()):a.Get()for a in prim.GetAttributes() if 'riction' in str(a.GetName()) or 'estitution' in str(a.GetName())}
    # Native SDK fixture poses are checked again after physics initialization.
    return dict(old_tool_center_m=old.tolist(),new_tool_center_m=initial.tolist(),initial_object_offset_m=offset,geometry_y_scale=y_scale,
        authored_transforms=updated,reference_shape_authoring=reference_authoring,material_usd_readback=material_attributes,
        scope='prephysics USD authoring only; actual collider and gravity SDK readback required separately')
