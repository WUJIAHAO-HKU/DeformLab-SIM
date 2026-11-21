"""Solid-model receiving gauge and an exact full-object aperture fit criterion."""
import numpy as np

def specification(side_clearance_m,body_y_scale=1.,allowed_clearances_m=None):
    allowed=(.00025,.0005,.001) if allowed_clearances_m is None else tuple(allowed_clearances_m)
    if not allowed or not all(np.isfinite(c) and 0<c<=.001 for c in allowed):
        raise ValueError('Invalid frozen clearance list')
    if side_clearance_m not in allowed:
        raise ValueError('Use the prospectively specified clearances')
    if body_y_scale not in (.75,1.,1.25):raise ValueError('Unsupported initial body size')
    return dict(center_xy_m=[.04,0.],opening_xy_m=[.04+2*side_clearance_m,.04*body_y_scale+.002],
        bar_width_m=.004,bar_height_m=.025,initial_center_z_m=.23,final_center_z_m=.15,
        side_clearance_m=side_clearance_m,body_nominal_size_m=[.04,.04*body_y_scale,.04],
        geometry_units='metres',criterion='All material points fit within the convex aperture in horizontal projection before insertion.',
        dynamic_validation='Required separately: actual collider contact, commanded/actual descent, deformation and reaction. Aperture fit alone is not a completed physical insertion.')

def bars(spec):
    cx,cy=spec['center_xy_m'];lx,ly=spec['opening_xy_m'];w=spec['bar_width_m'];h=spec['bar_height_m']
    return [
        dict(name='Left',center=[cx-lx/2-w/2,cy,0.],size=[w,ly+2*w,h]),
        dict(name='Right',center=[cx+lx/2+w/2,cy,0.],size=[w,ly+2*w,h]),
        dict(name='Back',center=[cx,cy-ly/2-w/2,0.],size=[lx,w,h]),
        dict(name='Front',center=[cx,cy+ly/2+w/2,0.],size=[lx,w,h])]

def aperture_fit(points,spec):
    x=np.asarray(points,dtype=float)
    if x.ndim!=2 or x.shape[1]!=3 or not len(x)or not np.isfinite(x).all():raise ValueError('Invalid actual material state')
    c=np.asarray(spec['center_xy_m']);half=np.asarray(spec['opening_xy_m'])/2
    minimum=x[:,:2].min(0);maximum=x[:,:2].max(0)
    lower=minimum-(c-half);upper=(c+half)-maximum
    margins=np.r_[lower,upper]
    return dict(fits=bool(np.all(margins>=0)),minimum_clearance_m=float(margins.min()),
        lower_clearance_xy_m=lower.tolist(),upper_clearance_xy_m=upper.tolist(),
        measured_extent_xy_m=(maximum-minimum).tolist(),measured_center_xy_m=((maximum+minimum)/2).tolist(),
        mechanism='Vertex containment is sufficient and necessary for a piecewise-linear volume to lie in this convex, vertically extruded aperture. This is pre-insertion geometry, not a collision-force measurement.')

def author_gauge(stage,spec,path='/World/PrecisionGauge'):
    """Author four actual rigid collision shapes; no texture-only geometry."""
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics
    root=UsdGeom.Xform.Define(stage,path)
    root.AddTranslateOp().Set(Gf.Vec3d(0.,0.,spec['initial_center_z_m']))
    rigid=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True);rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.2)
    material=UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(path+'/ContactMaterial','Material'))
    material.CreateStaticFrictionAttr(.4);material.CreateDynamicFrictionAttr(.4);material.CreateRestitutionAttr(0.)
    from pxr import UsdShade
    authored=[]
    for bar in bars(spec):
        cube=UsdGeom.Cube.Define(stage,path+'/'+bar['name']);cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*bar['center']))
        cube.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*bar['size']))
        cube.CreateDisplayColorAttr([Gf.Vec3f(.34,.36,.39)])
        collision=UsdPhysics.CollisionAPI.Apply(cube.GetPrim());collision.CreateCollisionEnabledAttr(spec.get('collisions_enabled',True))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(UsdShade.Material(material.GetPrim()),materialPurpose='physics')
        authored.append(str(cube.GetPath()))
    return dict(root=path,solid_collision_shapes=authored,specification=spec,
                validated_in_native_runtime=False,requires_kinematic_target_publication=True)
