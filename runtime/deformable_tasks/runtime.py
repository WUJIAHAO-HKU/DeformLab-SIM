"""Shared native Isaac task runtime. Driven fixtures, genuine deformable contact.

No node positions are prescribed. Display meshes copy measured GPU state.
Fixtures are servo-driven Cartesian articulations, not robot-arm policies.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
from .config import validate, digest

ROOT=Path(__file__).resolve().parents[1]

def smooth(x):
    x=float(np.clip(x,0,1));return x*x*x*(10+x*(-15+6*x))

class TaskRuntime:
    def __init__(self, config, external_every_iteration=False, contact_schedule=None):
        self.contact_schedule=contact_schedule
        self.observe_failed_horizon=False
        self.external_every_iteration=external_every_iteration;self.update_visuals=True
        self.config=validate(config);self.bodies=[];self.movers=[];self.samples=[];self.frames=[]
        self.steps=0;self.error=None;self.callback_ids=[];self.latest={};self.rest_volumes=[]
        self.minimum_volume_ratio=1.;self.maximum_penetration=0.;self.maximum_speed=0.
        self.maximum_actuator_error=0.;self.maximum_press_penetration=0.
        self.maximum_attachment_error=0.;self.state_finite=True
        self.fold_support_geometry=None;self.maximum_fold_support_penetration=0.
        self.fold_pose_history=[];self.maximum_angular_error=0.;self.maximum_roll_error=0.
        self.minimum_fixture_clearance=None;self.maximum_visual_angular_error=0.;self.moving_edge_local=None
        self.angular_velocity_target_updates=0;self.maximum_velocity_target_error=0.

    def build(self,stage):
        from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,UsdShade,Gf,Vt,Sdf
        from omni.physx.scripts import deformableUtils
        from isaacsim.core.experimental.prims import DeformablePrim,GeomPrim,RigidPrim,Articulation
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.materials import SurfaceDeformableMaterial,VolumeDeformableMaterial,PreviewSurfaceMaterial
        from isaacsim.core.simulation_manager import SimulationManager,PhysxScene
        self.stage=stage;self.Vt=Vt;self.Gf=Gf;self.UsdGeom=UsdGeom
        c=self.config;m=c['material'];g=c['geometry'];s=c['solver'];task=c['task']
        spec=importlib.util.spec_from_file_location('task_mesh',ROOT/'scripts/isaac_deformable_baseline.py')
        geom=importlib.util.module_from_spec(spec);spec.loader.exec_module(geom);self.geom=geom
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim());stage.SetTimeCodesPerSecond(s['hz'])
        stage.SetMetadata('customLayerData',{'taskConfigJSON':json.dumps(c),'taskConfigSHA256':digest(c)})
        scene=PhysxScene('/World/PhysicsScene');scene.set_steps_per_second(s['hz']);scene.set_gravity([0,0,-9.81])
        SimulationManager.set_device('cuda:0');SimulationManager.enable_fabric(False)
        sceneprim=stage.GetPrimAtPath('/World/PhysicsScene')
        sceneprim.GetAttribute('physxScene:solverType').Set('TGS')
        sceneprim.GetAttribute('physxScene:enableExternalForcesEveryIteration').Set(self.external_every_iteration)
        def material(name,color,rough=.5,metal=0.):
            mat=PreviewSurfaceMaterial('/World/Looks/'+name)
            for k,v in [('diffuseColor',color),('roughness',rough),('metallic',metal)]:mat.set_input_values(k,v)
            return mat
        steel=material('BrushedSteel',[.24,.28,.33],.28,.82)
        rubber=material('Rubber',[.023,.028,.033],.85)
        tablemat=material('LabBench',[.32,.28,.22],.6)
        floor=material('Floor',[.19,.22,.25],.8)
        bodymats=[material('TealSilicone',[.025,.47,.43],.38),material('AmberSilicone',[.87,.25,.035],.4)]
        pm=UsdPhysics.MaterialAPI.Apply(stage.DefinePrim('/World/Looks/GripPhysics','Material'))
        pm.CreateStaticFrictionAttr(m['friction']);pm.CreateDynamicFrictionAttr(m['friction']);pm.CreateRestitutionAttr(0.)
        self.drives=[];self.mover_targets={};self.mover_visuals=[]
        def box(path,pos,scale,mat,collision=True,moving=False):
            root=path
            rotating=moving and c['schema_version'] in (4,5) and task=='cloth_fold' and root=='/World/FoldBar'
            if moving:
                UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage,root).GetPrim())
                # Three driven prismatic joints form a physical Cartesian fixture.
                # Tensor kinematic targets are not implemented in this pinned GPU build.
                previous=None
                for name in (('Base','SlideX','SlideY','SlideZ') if rotating else ('Base','SlideX','SlideY')):
                    link=UsdGeom.Xform.Define(stage,root+'/'+name)
                    link.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
                    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
                    mass=UsdPhysics.MassAPI.Apply(link.GetPrim());mass.CreateMassAttr(1.)
                    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001,.001,.001))
                fixed=UsdPhysics.FixedJoint.Define(stage,root+'/WorldAnchor')
                fixed.CreateBody1Rel().SetTargets([root+'/Base']);fixed.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos]))
                path=root+'/Tool';self.mover_targets[root]=path
            geometry_path=path+'/Shape' if moving else path
            if moving:
                frame=UsdGeom.Xform.Define(stage,path);frame.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
            Cube(geometry_path,sizes=1.,positions=pos,scales=scale)
            gp=GeomPrim(geometry_path,apply_collision_apis=collision);gp.apply_visual_materials(mat)
            prim=stage.GetPrimAtPath(path)
            shape_prim=stage.GetPrimAtPath(geometry_path)
            if collision:
                shape_prim.ApplyAPI('PhysxCollisionAPI')
                offset=float((self.contact_schedule or {}).get('rigid_contact_offset_m',.0015))
                if offset not in (.0015,.02) or (offset==.02 and not task.startswith('cloth')):raise ValueError('Unsupported rigid candidate envelope')
                if not shape_prim.GetAttribute('physxCollision:contactOffset').Set(offset):raise RuntimeError('Rigid candidate envelope authoring failed')
                shape_prim.GetAttribute('physxCollision:restOffset').Set(0.)
                UsdShade.MaterialBindingAPI.Apply(shape_prim).Bind(UsdShade.Material(pm.GetPrim()),materialPurpose='physics')
            if moving:
                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(.2)
                chain=['Base','SlideX','SlideY','SlideZ' if rotating else 'Tool']
                for j,axis in enumerate(('X','Y','Z')):
                    joint=UsdPhysics.PrismaticJoint.Define(stage,root+'/Drive'+axis)
                    joint.CreateAxisAttr(axis);joint.CreateBody0Rel().SetTargets([root+'/'+chain[j]])
                    joint.CreateBody1Rel().SetTargets([root+'/'+chain[j+1]])
                    joint.CreateLowerLimitAttr(-.5);joint.CreateUpperLimitAttr(.5)
                    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
                    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(1e6);drive.CreateDampingAttr(4000.)
                    drive.CreateTargetPositionAttr(0.)
                if rotating:
                    from .fold_rotation import author_roll_joint
                    author_roll_joint(stage,root)
                self.drives.append(Articulation(root,reset_xform_op_properties=False))
                rigid=RigidPrim(path,reset_xform_op_properties=False)
                self.movers.append((rigid,np.asarray(pos,dtype=np.float32),path))
                UsdGeom.Imageable(prim).MakeInvisible()
                display=root+'/ToolDisplay'
                Cube(display,sizes=1.,positions=pos,scales=scale)
                GeomPrim(display).apply_visual_materials(mat)
                visual_prim=stage.GetPrimAtPath(display)
                if rotating and not visual_prim.GetAttribute('xformOp:orient').IsValid():
                    UsdGeom.Xformable(visual_prim).AddOrientOp().Set(Gf.Quatf(1.))
                self.mover_visuals.append(visual_prim)
            return prim
        box('/World/Floor',[0,0,-.08],[20,20,.1],floor)
        self.table_z=.12
        box('/World/Bench',[0,0,.085],[.72,.62,.07],tablemat)
        for x in (-.29,.29):
            for y in (-.24,.24):box('/World/Leg'+str(x).replace('.','_').replace('-','n')+str(y).replace('.','_').replace('-','n'),[x,y,.01],[.026,.026,.08],steel,False)
        # Area lights produce finite soft shadows and distinct material highlights.
        for name,pos,power,color,size in [('Key',(.5,-.5,1.2),2200,(1.,.88,.76),(.8,.55)),('Fill',(-.6,-.1,.8),1300,(.73,.85,1.),(.65,.65)),('Rim',(.2,.65,1.),1800,(1.,1.,1.),(.7,.3))]:
            light=UsdLux.RectLight.Define(stage,'/World/'+name);light.CreateIntensityAttr(power)
            light.CreateWidthAttr(size[0]);light.CreateHeightAttr(size[1]);light.CreateColorAttr(Gf.Vec3f(*color))
            view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos),Gf.Vec3d(0,0,.2),Gf.Vec3d(0,0,1)).GetInverse()
            light.MakeMatrixXform().Set(view)
        dome=UsdLux.DomeLight.Define(stage,'/World/Dome');dome.CreateIntensityAttr(450.)
        dome.CreateColorAttr(Gf.Vec3f(.75,.82,1.))
        sun=UsdLux.DistantLight.Define(stage,'/World/StudioSun');sun.CreateIntensityAttr(1600.)
        sun.CreateAngleAttr(12.);sun.AddRotateXYZOp().Set(Gf.Vec3f(25,-35,-20))
        # Real authored camera, reusable by viewport and offscreen renderer.
        cam=UsdGeom.Camera.Define(stage,'/World/Camera')
        cam.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.68,-.88,.64),Gf.Vec3d(0,0,.25),Gf.Vec3d(0,0,1)).GetInverse())
        cam.CreateFocalLengthAttr(38);cam.CreateHorizontalApertureAttr(36)
        cam.CreateVerticalApertureAttr(36*c['rendering']['height']/c['rendering']['width']);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
        def body(name,points,elements,kind,index=0,uv=None):
            path='/World/'+name
            mesh=geom.author_mesh(stage,path,points,elements,kind,Usd,UsdGeom,Vt)
            setup=deformableUtils.set_physics_surface_deformable_body if kind=='surface' else deformableUtils.set_physics_volume_deformable_body
            if not setup(stage,path):raise RuntimeError('Deformable authoring failed')
            prim=mesh.GetPrim();prim.ApplyAPI('PhysxBaseDeformableBodyAPI');prim.ApplyAPI('PhysxCollisionAPI')
            for k,v in [('solverPositionIterationCount',s['position_iterations']),('linearDamping',s['linear_damping']),('selfCollision',s['self_collision']),('enableSpeculativeCCD',True)]:
                if not prim.GetAttribute('physxDeformableBody:'+k).Set(v):raise RuntimeError('Unsupported solver attribute '+k)
            prim.GetAttribute('physxCollision:contactOffset').Set(.0015)
            prim.GetAttribute('physxCollision:restOffset').Set(.0008 if kind=='surface' else 0.)
            if self.contact_schedule and 'speculative_ccd' in self.contact_schedule:
                prim.GetAttribute('physxDeformableBody:enableSpeculativeCCD').Set(self.contact_schedule['speculative_ccd'])
            if self.contact_schedule and 'pair_updates' in self.contact_schedule:
                if kind!='surface':raise ValueError('Surface schedule cannot be applied to volume bodies')
                prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI')
                for key,value in [('collisionPairUpdateFrequency',self.contact_schedule['pair_updates']),
                                  ('collisionIterationMultiplier',self.contact_schedule['collision_multiplier'])]:
                    if not prim.GetAttribute('physxDeformableBody:'+key).Set(value):raise RuntimeError('Unsupported contact setting '+key)
                if 'filter_distance' in self.contact_schedule:
                    prim.GetAttribute('physxDeformableBody:selfCollisionFilterDistance').Set(self.contact_schedule['filter_distance'])
            dp=DeformablePrim(path,deformable_type=kind,reset_xform_op_properties=False)
            mt=SurfaceDeformableMaterial if kind=='surface' else VolumeDeformableMaterial
            mat=mt(path+'Physics',youngs_moduli=m['young_pa'],poissons_ratios=m['poisson'],densities=m['density'],static_frictions=m['friction'],dynamic_frictions=m['friction'])
            dp.apply_physics_materials(mat);mp=stage.GetPrimAtPath(path+'Physics')
            mp.ApplyAPI('PhysxDeformableMaterialAPI');mp.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(m['elasticity_damping'])
            if kind=='surface':mp.GetAttribute('omniphysics:surfaceThickness').Set(m['thickness_m'])
            UsdGeom.Imageable(mesh).MakeInvisible()
            faces=elements if kind=='surface' else geom.boundary_triangles(elements)
            visual=geom.author_mesh(stage,path+'Display',points,faces,'surface',Usd,UsdGeom,Vt)
            visual.CreateDoubleSidedAttr(True);GeomPrim(path+'Display').apply_visual_materials(bodymats[index])
            # Smooth shading only changes display normals, never collision mesh resolution.
            if uv is not None:
                st=UsdGeom.PrimvarsAPI(visual).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex')
                st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv,dtype=np.float32)))
                clothmat=material('WovenCloth',[.035,.42,.44],.92)
                shader=UsdShade.Shader(stage.GetPrimAtPath('/World/Looks/WovenCloth/Shader'))
                # Discover the wrapper's shader path through its surface connection.
                source=UsdShade.Material(stage.GetPrimAtPath('/World/Looks/WovenCloth')).GetSurfaceOutput().GetConnectedSource()
                if source:shader=UsdShade.Shader(source[0].GetPrim())
                texture=ROOT/'assets/task-platform/cloth-weave.png'
                if texture.exists():
                    reader=UsdShade.Shader.Define(stage,'/World/Looks/ClothUV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
                    tex=UsdShade.Shader.Define(stage,'/World/Looks/ClothTexture');tex.CreateIdAttr('UsdUVTexture')
                    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture));tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
                    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
                    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3);shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
                GeomPrim(path+'Display').apply_visual_materials(clothmat)
            restv=geom.signed_tet_volumes(points,elements) if kind=='volume' else None
            self.bodies.append(dict(path=path,view=dp,visual=visual,rest=np.asarray(points).copy(),elements=np.asarray(elements),faces=np.asarray(faces),kind=kind,restv=restv))
            return path
        def attach(name,path,ids,target,points,center):
            target=self.mover_targets.get(target,target)
            at=stage.DefinePrim('/World/'+name,'OmniPhysicsVtxXformAttachment')
            at.GetRelationship('omniphysics:src0').SetTargets([path]);at.GetRelationship('omniphysics:src1').SetTargets([target])
            at.GetAttribute('omniphysics:vtxIndicesSrc0').Set(Vt.IntArray.FromNumpy(np.asarray(ids,dtype=np.int32)))
            inverse=UsdGeom.Xformable(stage.GetPrimAtPath(target)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            local=np.asarray([inverse.Transform(Gf.Vec3d(*[float(v) for v in p])) for p in points[ids]],dtype=np.float32)
            at.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(local))
            if name=='MovingEdge':self.moving_edge_local=local.copy()
        if task.startswith('cloth'):
            n=g['cloth_resolution'];nx=n;ny=2*n
            uv=np.array([[i/nx,j/ny] for j in range(ny+1) for i in range(nx+1)],np.float32)
            if task=='cloth_grip':points=np.c_[(uv[:,0]-.5)*.14,np.zeros(len(uv)),.16+uv[:,1]*.28].astype(np.float32)
            else:points=np.c_[(uv[:,0]-.5)*.16,(uv[:,1]-.5)*.28,np.full(len(uv),.155)].astype(np.float32)
            faces=[]
            for j in range(ny):
                for i in range(nx):
                    a=j*(nx+1)+i;b=a+1;d=a+nx+2;cc=a+nx+1;faces.extend([[a,b,d],[a,d,cc]])
            path=body('Cloth',points,np.asarray(faces,np.int32),'surface',uv=uv)
            self.edge_ids=np.arange(nx+1)+ny*(nx+1);self.nx=nx;self.ny=ny
            if task=='cloth_grip':
                self.jaw_z=.425
                for i,sign in enumerate((-1,1)):
                    box('/World/Jaw'+str(i),[0,sign*(.012+c['action']['grip_gap_m']/2),self.jaw_z],[.18,.024,.04],rubber,moving=True)
                box('/World/FixturePost',[.14,.04,.3],[.025,.04,.36],steel,False)
            else:
                # The sheet rests 0.8 mm above the upper bar surface, not
                # embedded through its center. Attachment offsets are computed
                # from the actual transforms below. This fixture fix is shared
                # by baseline and all candidate schedules.
                if c.get('fixtures',{}).get('fold_support',False):
                    from .fold_fixture import support_geometry
                    self.fold_support_geometry=support_geometry(points,self.table_z)
                    support=self.fold_support_geometry
                    box('/World/FoldSupport',support['center_m'],support['size_m'],tablemat)
                bar_z=.155-.004-.0008
                for name,y in [('FixedBar',-.14),('FoldBar',.14)]:box('/World/'+name,[0,y,bar_z],[.18,.008,.008],steel,moving=name=='FoldBar')
                # Attach to an unscaled world anchor, not scaled visible geometry.
                UsdGeom.Xform.Define(stage,'/World/FixedEdgeAnchor')
                attach('FixedEdge',path,np.arange(nx+1),'/World/FixedEdgeAnchor',points,np.zeros(3))
                attach('MovingEdge',path,self.edge_ids,'/World/FoldBar',points,np.array([0,.14,.155]))
        else:
            size=g['size_m'];self.size=size
            pts,tets=geom.build_mesh('volume',g['resolution'])
            pts=(pts-np.array([0,0,.95]))*(size/.5)
            if task=='soft_squeeze':
                center=np.array([0,0,self.table_z+size/2+.0005]);body('SoftA',(pts+center).astype(np.float32),tets,'volume')
                self.plate_z=self.table_z+size+.012
                box('/World/Press',[0,0,self.plate_z],[size*2,size*2,.02],steel,moving=True)
            else:
                body('SoftA',(pts+[0,0,self.table_z+size/2+.0005]).astype(np.float32),tets,'volume')
                body('SoftB',(pts+[size*.08,0,self.table_z+size*2.3]).astype(np.float32),tets,'volume',1)
                if not s['contacts']:
                    a=UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/World/SoftA'))
                    a.CreateFilteredPairsRel().SetTargets(['/World/SoftB'])
        self.steps=0
        return self

    def target(self,i,t):
        p=self.movers[i][1].copy();task=self.config['task'];a=self.config['action']
        if task=='cloth_grip':
            p[2]+=.07*smooth((t-.6)/1.2)
            p[1]+=(-1 if i==0 else 1)*.045*smooth((t-a['release_s'])/.3)
        elif task=='cloth_fold':
            # Both fixture bars are 8 mm thick. A 4 mm terminal center
            # separation commands rigid-rigid overlap and cannot track.
            if self.config['schema_version'] in (4,5):
                from .fold_rotation import pose
                p=pose(t,self.config['motion'])['center'].astype(np.float32)
            elif self.config['schema_version']==3:
                from .fold_motion import fold_offset
                motion=self.config['motion'];y,z=fold_offset(t,motion['start_s'],motion['duration_s'],motion['terminal_clearance_m'])
                p[1]=y;p[2]+=z
            else:
                q=smooth((t-.3)/2.);ang=np.pi*q;p[1]=.14*np.cos(ang);p[2]+=.14*np.sin(ang)+.010*q
        elif task=='soft_squeeze':
            q=smooth((t-.4)/1.4)*(1-smooth((t-2.7)/.7));p[2]-=self.size*a['compression']*q
        return p

    def initialize_fixture_drives(self):
        """Reset all authored fixture DOFs once, before recording the initial state."""
        from isaacsim.core.experimental.utils.backend import use_backend
        if self.steps!=0 or self.latest:raise RuntimeError('Fixture initialization is forbidden during a measured rollout')
        expected=['DriveX','DriveY','DriveZ']+(['DriveRollX'] if self.config['schema_version'] in (4,5) else [])
        prepared=[]
        for drive in self.drives:
            names=list(drive.dof_names)
            if len(names)!=len(expected) or set(names)!=set(expected):raise ValueError('Unexpected fixture degrees of freedom: '+str(names))
            prepared.append((drive,[names.index(name) for name in expected]))
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            for drive,indices in prepared:
                zeros=np.zeros(len(indices),dtype=np.float32)
                drive.set_dof_positions(zeros,dof_indices=indices)
                drive.set_dof_velocities(zeros,dof_indices=indices)
                drive.set_dof_position_targets(zeros,dof_indices=indices)

    def before_step(self,*args):
        if self.error and not self.observe_failed_horizon:return
        for i,(view,_,_) in enumerate(self.movers):
            target=self.target(i,(self.steps+1)/self.config['solver']['hz'])
            drive=self.drives[i]
            indices=[drive.dof_names.index('Drive'+axis) for axis in ('X','Y','Z')]
            if self.config['schema_version'] in (4,5):
                from .fold_rotation import pose
                from isaacsim.core.experimental.utils.backend import use_backend
                indices.append(drive.dof_names.index('DriveRollX'))
                command=np.r_[target-self.movers[i][1],pose((self.steps+1)/self.config['solver']['hz'],self.config['motion'])['angle_rad']]
                with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                    drive.set_dof_position_targets(command,dof_indices=indices)
                    if self.config['schema_version']==5 and self.config['motion']['angular_velocity_feedforward']:
                        from .fold_rotation import angular_velocity
                        velocity=angular_velocity((self.steps+1)/self.config['solver']['hz'],self.config['motion'])
                        drive.set_dof_velocity_targets([velocity],dof_indices=[indices[-1]])
                        self.angular_velocity_target_updates+=1
            else:
                drive.set_dof_position_targets(target-self.movers[i][1],dof_indices=indices)

    def after_step(self,*args,advance=True):
        from isaacsim.core.experimental.utils.backend import use_backend
        self.steps+=int(advance);t=self.steps/self.config['solver']['hz'];states=[];sync=0.
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            actuator_positions=[]
            for i,(view,_,_) in enumerate(self.movers):
                world_pose=view.get_world_poses();actual=world_pose[0].numpy()[0]
                if self.config['schema_version'] in (4,5):
                    from .fold_rotation import pose,angular_error,rigid_clearance_bounds
                    actual_orientation=world_pose[1].numpy()[0]
                    expected=pose(t,self.config['motion']);drive=self.drives[i]
                    roll=float(drive.get_dof_positions(dof_indices=[drive.dof_names.index('DriveRollX')]).numpy().reshape(-1)[0])
                    if not np.isfinite(roll):raise RuntimeError('Non-finite rotational DOF')
                    error=angular_error(actual_orientation,expected['orientation_wxyz'])
                    clearance=rigid_clearance_bounds(actual,actual_orientation,self.fold_support_geometry)
                    self.maximum_angular_error=max(self.maximum_angular_error,error)
                    self.maximum_roll_error=max(self.maximum_roll_error,abs(roll-expected['angle_rad']))
                    gap=min(clearance.values())
                    self.minimum_fixture_clearance=gap if self.minimum_fixture_clearance is None else min(self.minimum_fixture_clearance,gap)
                    rotation_record=dict(step=self.steps,time_s=t,center_m=actual.tolist(),orientation_wxyz=actual_orientation.tolist(),roll_rad=roll,target_roll_rad=expected['angle_rad'],angular_error_rad=error,rigid_clearance_lower_bounds_m=clearance)
                    if self.config['schema_version']==5:
                        from .fold_rotation import angular_velocity
                        roll_index=drive.dof_names.index('DriveRollX')
                        velocity_target=float(drive.get_dof_velocity_targets(dof_indices=[roll_index]).numpy().reshape(-1)[0])
                        velocity_actual=float(drive.get_dof_velocities(dof_indices=[roll_index]).numpy().reshape(-1)[0])
                        velocity_command=angular_velocity(t,self.config['motion']) if self.config['motion']['angular_velocity_feedforward'] else 0.
                        if not np.isfinite([velocity_target,velocity_actual]).all():raise RuntimeError('Non-finite angular velocity state/target')
                        self.maximum_velocity_target_error=max(self.maximum_velocity_target_error,abs(velocity_target-velocity_command))
                        rotation_record.update(angular_velocity_command_rad_s=velocity_command,angular_velocity_target_readback_rad_s=velocity_target,angular_velocity_actual_rad_s=velocity_actual)
                    if self.fold_pose_history and self.fold_pose_history[-1]['step']==self.steps:self.fold_pose_history[-1]=rotation_record
                    else:self.fold_pose_history.append(rotation_record)
                actuator_positions.append(actual.tolist())
                # The GPU pipeline suppresses rigid USD readback. Keep render-only
                # tools synchronized without writing transforms to physics bodies.
                if self.update_visuals:
                    self.mover_visuals[i].GetAttribute('xformOp:translate').Set(self.Gf.Vec3d(*[float(v) for v in actual]))
                    if self.config['schema_version'] in (4,5):
                        attr=self.mover_visuals[i].GetAttribute('xformOp:orient')
                        quat_type=self.Gf.Quatf if str(attr.GetTypeName())=='quatf' else self.Gf.Quatd
                        attr.Set(quat_type(*[float(v) for v in actual_orientation]))
                        rendered_q=attr.Get();rendered_orientation=np.r_[rendered_q.GetReal(),np.asarray(rendered_q.GetImaginary())]
                        self.maximum_visual_angular_error=max(self.maximum_visual_angular_error,angular_error(rendered_orientation,actual_orientation))
                    rendered=np.asarray(self.mover_visuals[i].GetAttribute('xformOp:translate').Get())
                    sync=max(sync,float(np.max(np.abs(rendered-actual))))
                self.maximum_actuator_error=max(self.maximum_actuator_error,float(np.linalg.norm(actual-self.target(i,t))))
            for b in self.bodies:
                arrays=b['view'].get_nodal_positions();raw=arrays[0]
                if str(raw.device)!='cuda:0':raise RuntimeError('GPU backend required')
                p=raw.numpy()[0];v=b['view'].get_nodal_velocities().numpy()[0]
                if not np.isfinite(p).all() or not np.isfinite(v).all():
                    self.state_finite=False;raise RuntimeError('Non-finite deformable state')
                if b['kind']=='volume' and 'sim_elements' not in b:
                    rest=arrays[2].numpy()[0];elements=b['view'].get_element_indices()[0].numpy()[0]
                    b['sim_elements']=elements;b['sim_restv']=self.geom.signed_tet_volumes(rest,elements)
                    b['gpu_rest_difference_m']=float(np.max(np.abs(rest-b['rest'])))
                    if not np.all(np.abs(b['sim_restv'])>1e-20):raise RuntimeError('Degenerate GPU reference tetrahedra')
                if self.update_visuals:
                    b['visual'].GetPointsAttr().Set(self.Vt.Vec3fArray.FromNumpy(p))
                    b['visual'].GetExtentAttr().Set(self.Vt.Vec3fArray.FromNumpy(np.asarray([p.min(0),p.max(0)],dtype=np.float32)))
                    f=b['faces'];normals=np.zeros_like(p);fn=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
                    for j in range(3):np.add.at(normals,f[:,j],fn)
                    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
                    b['visual'].GetNormalsAttr().Set(self.Vt.Vec3fArray.FromNumpy(normals));b['visual'].SetNormalsInterpolation('vertex')
                    sync=max(sync,float(np.max(np.abs(np.asarray(b['visual'].GetPointsAttr().Get())-p))))
                vr=float(np.min(self.geom.signed_tet_volumes(p,b['sim_elements'])/b['sim_restv'])) if b['kind']=='volume' else None
                if vr is not None:self.minimum_volume_ratio=min(self.minimum_volume_ratio,vr)
                speed=float(np.max(np.linalg.norm(v,axis=1)));self.maximum_speed=max(speed,self.maximum_speed)
                pen=max(0.,self.table_z-float(p[:,2].min()));self.maximum_penetration=max(self.maximum_penetration,pen)
                if self.config['task']=='soft_squeeze':
                    inside=(np.abs(p[:,0])<self.size)&(np.abs(p[:,1])<self.size)
                    if inside.any():self.maximum_press_penetration=max(self.maximum_press_penetration,float(p[inside,2].max()-(actuator_positions[0][2]-.01)),0.)
                if self.fold_support_geometry is not None:
                    from .fold_fixture import penetration_depth
                    self.maximum_fold_support_penetration=max(self.maximum_fold_support_penetration,penetration_depth(p,self.fold_support_geometry))
                states.append(dict(centroid=p.mean(0).tolist(),min_z=float(p[:,2].min()),max_z=float(p[:,2].max()),min_volume_ratio=vr,max_speed=speed))
                b['last']=p.copy()
        self.latest=dict(step=self.steps,time_s=t,bodies=states,display_error_m=sync,actuator_positions=actuator_positions)
        if self.config['task']=='cloth_grip':
            self.latest['gripped_edge_z']=float(self.bodies[0]['last'][self.edge_ids,2].mean())
            self.latest['jaw_z']=float(self.target(0,t)[2])
        if self.config['task']=='cloth_fold':
            b=self.bodies[0];actual=b['last'];rest=b['rest']
            bottom=np.arange(self.nx+1)
            if self.config['schema_version'] in (4,5):
                from .fold_rotation import attachment_targets
                top_target=attachment_targets(self.moving_edge_local,np.zeros(3),np.asarray(actuator_positions[0]),actual_orientation)
            else:
                top_target=rest[self.edge_ids]-self.movers[0][1]+np.asarray(actuator_positions[0])
            error=max(float(np.linalg.norm(actual[bottom]-rest[bottom],axis=1).max()),float(np.linalg.norm(actual[self.edge_ids]-top_target,axis=1).max()))
            self.maximum_attachment_error=max(self.maximum_attachment_error,error)
            self.latest['attachment_error_m']=error
            if self.config['schema_version'] in (4,5):
                self.fold_pose_history[-1]['attachment_error_m']=error
                self.latest['fold_pose']=self.fold_pose_history[-1].copy()
        if self.steps%max(1,self.config['solver']['hz']//60)==0:self.samples.append(self.latest.copy())
        if self.minimum_volume_ratio<=0: self.error='Inverted volume elements; state retained'

    def report(self):
        from .config import digest
        finite=bool(self.latest) and self.state_finite
        result=dict(task=self.config['task'],config=self.config,config_sha256=digest(self.config),steps=self.steps,
                    scope='Driven Cartesian articulation fixture; no robot-arm policy or real-material calibration claim.',
                    checks=dict(finite_state=finite,display_matches_physics=bool(self.latest) and self.latest['display_error_m']<1e-6,
                        positive_volume=self.minimum_volume_ratio>self.config['validation']['min_volume_ratio'],
                        bench_penetration=self.maximum_penetration<=self.config['validation']['penetration_tolerance_m'],
                        actuator_tracks_command=self.maximum_actuator_error<.003,
                        attachment_consistency=self.maximum_attachment_error<.003,
                        press_penetration=self.maximum_press_penetration<=self.config['validation']['penetration_tolerance_m']),
                    minimum_volume_ratio=self.minimum_volume_ratio,maximum_bench_penetration_m=self.maximum_penetration,
                    maximum_actuator_error_m=self.maximum_actuator_error,maximum_press_penetration_m=self.maximum_press_penetration,
                    maximum_attachment_error_m=self.maximum_attachment_error,
                    gpu_rest_differences_m=[b.get('gpu_rest_difference_m') for b in self.bodies],
                    maximum_speed_m_s=self.maximum_speed,error=self.error,samples=self.samples)
        if self.fold_support_geometry is not None:
            result['fold_support_geometry']=self.fold_support_geometry
            result['maximum_fold_support_penetration_m']=self.maximum_fold_support_penetration
            result['checks']['fold_support_penetration']=self.maximum_fold_support_penetration<=self.config['validation']['penetration_tolerance_m']
        if self.config['schema_version'] in (4,5):
            history=self.fold_pose_history
            result['fold_rotation']=dict(joint='DriveRollX',angular_units='radians',quaternion_order='wxyz',
                maximum_angular_error_rad=self.maximum_angular_error,maximum_roll_error_rad=self.maximum_roll_error,
                minimum_saved_pose_rigid_clearance_lower_bound_m=self.minimum_fixture_clearance,
                maximum_visual_angular_error_rad=self.maximum_visual_angular_error,poses=history,
                scope='Actual tensor poses and revolute DOF at each observed endpoint; no continuous cloth collision certificate.')
            result['checks'].update(fold_rotation_tracks_command=bool(history) and self.maximum_angular_error<.02 and self.maximum_roll_error<.02,
                fold_rigid_fixtures_separated=self.minimum_fixture_clearance is not None and self.minimum_fixture_clearance>0,
                fold_pose_recording_complete=[h['step'] for h in history]==list(range(self.steps+1)))
            if self.update_visuals:result['checks']['fold_visual_rotation_matches_physics']=bool(history) and self.maximum_visual_angular_error<1e-6
            if self.config['schema_version']==5:
                feedforward=self.config['motion']['angular_velocity_feedforward']
                result['fold_rotation'].update(angular_velocity_feedforward=feedforward,angular_velocity_target_updates=self.angular_velocity_target_updates,maximum_velocity_target_error_rad_s=self.maximum_velocity_target_error)
                result['checks']['fold_velocity_targets_verified']=bool(history) and self.maximum_velocity_target_error<1e-5 and self.angular_velocity_target_updates==(self.steps if feedforward else 0)
        result['display_sync_checked']=self.update_visuals
        result['external_every_iteration']=self.external_every_iteration
        if not self.update_visuals:result['checks'].pop('display_matches_physics')
        return result

    def release(self):
        from isaacsim.core.simulation_manager import SimulationManager
        for ident in self.callback_ids:
            try:SimulationManager.deregister_callback(ident)
            except Exception:pass
        self.callback_ids=[];self.bodies=[];self.movers=[];self.drives=[];self.mover_visuals=[];self.fold_pose_history=[];self.moving_edge_local=None;self.stage=None
