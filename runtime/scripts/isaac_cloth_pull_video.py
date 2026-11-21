"""Recorded Franka tensile task: GPU surface cloth and explicit two-way attachments.

No nodal positions are prescribed during simulation. Display points and grid
lines only copy tensor readbacks; the attached edge is driven by the articulation.
"""
import argparse, importlib.util, json, os, sys, traceback, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',required=True)
    p.add_argument('--build-receipt',required=True)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--native-display',action='store_true')
    p.add_argument('--interior',action='store_true')
    p.add_argument('--physics-only',action='store_true')
    p.add_argument('--render-samples',type=int,choices=(16,32,64,128,256),default=32)
    args=p.parse_args();sys.argv=[sys.argv[0]]
    out=Path(args.output_dir).resolve()
    assert out.is_relative_to(ROOT/'results') and out.is_dir()
    common=load('pull_common',ROOT/'scripts/isaac_platform_smoke.py')
    geometry=load('pull_geometry',ROOT/'scripts/isaac_deformable_baseline.py')
    reference=load('pull_robot',ROOT/'scripts/isaac_volume_grasp.py')
    rotation=reference.rotation
    record=dict(status='starting',checks={},scope='Elastic cloth demonstration with explicit attachment grip; not friction-only grasp or material calibration.',script_sha256=common.sha256(Path(__file__)))
    app=None
    def save(): common.write_json(out/'report.json',record)
    def check(name,condition,**details):
        record['checks'][name]=dict(passed=bool(condition),**details);save()
        if not condition: raise RuntimeError('Failed: '+name)
    try:
        binding=common.runtime_binding(ROOT/'upstream/IsaacSim',os.environ.get('ISAAC_PATH'),sys.executable,'')
        check('runtime',binding['passed'])
        receipt=common.verify_build_receipt(Path(args.build_receipt),ROOT/'upstream/IsaacSim',binding)
        check('receipt',receipt['passed'])
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=not args.native_display,fast_shutdown=True,multi_gpu=False,renderer='PathTracing' if args.interior else 'RaytracedLighting',width=1280,height=720,
            extra_args=['--/app/useFabricSceneDelegate=1','--/app/extensions/pythonFullLeakCheck=false','--/crashreporter/enabled=false']),experience=binding['experience'])
        import numpy as np
        from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,Gf,Vt,Sdf
        from omni.physx.scripts import deformableUtils
        from isaacsim.core.experimental.utils import stage as stage_utils,app as app_utils
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.experimental.prims import Articulation,DeformablePrim,GeomPrim,RigidPrim
        from isaacsim.core.experimental.objects import Cube,DistantLight
        from isaacsim.core.experimental.materials import SurfaceDeformableMaterial,PreviewSurfaceMaterial
        from isaacsim.core.simulation_manager import SimulationManager,PhysxScene
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.sensors.experimental.rtx import RtxCamera,CameraSensor
        from isaacsim.storage.native import get_assets_root_path
        if args.physics_only:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().updates_enabled=False
        for _ in range(60):app.update()
        stage_utils.create_new_stage();stage=stage_utils.get_current_stage(backend='usd')
        UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
        stage.SetDefaultPrim(stage_utils.define_prim('/World','Xform'))
        hz=240;stage.SetTimeCodesPerSecond(hz)
        scene=PhysxScene('/World/PhysicsScene');scene.set_steps_per_second(hz);scene.set_gravity([0,0,-9.81])
        SimulationManager.set_device('cuda:0')
        def material(path,color,roughness=.45,metallic=0.):
            m=PreviewSurfaceMaterial(path);m.set_input_values('diffuseColor',color)
            m.set_input_values('roughness',roughness);m.set_input_values('metallic',metallic)
            return m
        graphite=material('/World/Graphite',[.055,.07,.09],.3,.5)
        floor_mat=material('/World/FloorMat',[.22,.25,.30],.65)
        gold=material('/World/Gold',[.95,.48,.05],.28,.6)
        def box(path,pos,scale,mat,collision=False):
            Cube(path,sizes=1.,scales=scale,positions=pos)
            gp=GeomPrim(path,apply_collision_apis=collision);gp.apply_visual_materials(mat)
            return stage.GetPrimAtPath(path)
        if args.interior:
            from deformable_tasks import interior
            record.update(status='composing_interior');save()
            audit=interior.compose(stage,out)
            record['interior_geometry']={k:v for k,v in audit.items() if k!='geometry'};save()
            import carb
            settings=carb.settings.get_settings()
            settings.set('/rtx/pathtracing/spp',4);settings.set('/rtx/pathtracing/totalSpp',args.render_samples)
            settings.set('/rtx/pathtracing/optixDenoiser/enabled',True)
            settings.set('/rtx/pathtracing/maxBounces',12)
            settings.set('/rtx/post/histogram/enabled',True)
            settings.set('/app/viewport/grid/enabled',False)
        else:
            box('/World/Floor',[0,0,-.045],[20,20,.08],floor_mat,True)
            DistantLight('/World/Key').set_intensities(1800.)
            light=UsdLux.DistantLight.Define(stage,'/World/Fill');light.CreateIntensityAttr(1100.);light.AddRotateXYZOp().Set(Gf.Vec3f(35,-45,30))
            dome=UsdLux.DomeLight.Define(stage,'/World/Dome');dome.CreateIntensityAttr(300.);dome.CreateColorAttr(Gf.Vec3f(.65,.78,1.))
        asset=get_assets_root_path()+'/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda'
        record['status']='loading_robot';save()
        stage_utils.add_reference_to_stage(usd_path=asset,path='/World/Franka',variants=[('Gripper','alternatefinger'),('Mesh','quality')])
        for _ in range(600):
            app.update()
            if not stage_utils.is_stage_loading():break
        for _ in range(20):
            instances=[p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')) if p.IsInstance()]
            if not instances:break
            for prim in instances:prim.SetInstanceable(False)
        robot=Articulation('/World/Franka')
        q0=np.array([0.,-.785,0.,-2.356,0.,1.571,.785,.010,.010],dtype=np.float32)
        robot.set_default_state(dof_positions=q0)
        # Derive initial object placement from asset joint frames, then verify against live physics.
        def transform(pos, quat):
            q = np.r_[float(quat.GetReal()), np.asarray(quat.GetImaginary(), dtype=float)]
            T = np.eye(4); T[:3,:3] = rotation(q); T[:3,3] = np.asarray(pos); return T
        poses = {'/World/Franka/panda_link0': np.eye(4)}
        joint_values = {f'panda_joint{i+1}':float(q0[i]) for i in range(7)}
        joint_values.update(panda_finger_joint1=.010, panda_finger_joint2=.010)
        pending = [UsdPhysics.Joint(prim) for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')) if prim.IsA(UsdPhysics.Joint)]
        for _ in range(20):
            for joint in pending:
                parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if not parents or not children or str(parents[0]) not in poses: continue
                move=np.eye(4); value=joint_values.get(joint.GetPrim().GetName(),0.)
                if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                    axis={'X':0,'Y':1,'Z':2}[joint.GetPrim().GetAttribute('physics:axis').Get()]
                    move[:3,:3]=rotation(np.r_[np.cos(value/2),np.eye(3)[axis]*np.sin(value/2)])
                elif joint.GetPrim().IsA(UsdPhysics.PrismaticJoint):
                    move[{'X':0,'Y':1,'Z':2}[joint.GetPrim().GetAttribute('physics:axis').Get()],3]=value
                poses[str(children[0])]=poses[str(parents[0])] @ transform(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()) @ move @ np.linalg.inv(transform(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get()))
        hand_expected=poses['/World/Franka/panda_hand']
        center=(hand_expected @ [0,0,.1,1])[:3]
        # Author the same rest pose BEFORE PhysX creates attachments. Resetting an
        # already running articulation would otherwise shock the elastic sheet.
        for path,T in poses.items():
            body=stage.GetPrimAtPath(path)
            assert str(body.GetParent().GetPath())=='/World/Franka'
            UsdGeom.Xformable(body).MakeMatrixXform().Set(Gf.Matrix4d(T.T.tolist()))
        for joint in pending:
            jp=joint.GetPrim();name=jp.GetName()
            if name not in joint_values:continue
            angular=jp.IsA(UsdPhysics.RevoluteJoint)
            instance='angular' if angular else 'linear'
            assert jp.ApplyAPI('PhysicsJointStateAPI',instance)
            value=joint_values[name]*(180/np.pi if angular else 1)
            assert jp.GetAttribute('state:'+instance+':physics:position').Set(value)
            assert jp.GetAttribute('state:'+instance+':physics:velocity').Set(0.)
            drive=UsdPhysics.DriveAPI(jp,instance)
            if drive:drive.GetTargetPositionAttr().Set(value)
        record['hand_expected']=hand_expected.tolist()
        # Vertical ribbon, wide clamp jaws along X, robot pulls upward.
        nx,nz=24,48;length=.28;width=.14
        fixed=center-np.array([0.,0.,length])
        points=np.array([[center[0]+width*(i/nx-.5),center[1]+.012*np.sin(np.pi*j/nz)*np.sin(2*np.pi*i/nx),fixed[2]+length*j/nz] for j in range(nz+1) for i in range(nx+1)],dtype=np.float32)
        faces=[]
        for j in range(nz):
            for i in range(nx):
                a=j*(nx+1)+i;b=a+1;c=a+nx+1;d=c+1
                faces.extend([[a,b,d],[a,d,c]])
        faces=np.asarray(faces,dtype=np.int32)
        bottom=np.arange(nx+1,dtype=np.int32);top=bottom+nz*(nx+1)
        mesh=geometry.author_mesh(stage,'/World/Cloth',points,faces,'surface',Usd,UsdGeom,Vt)
        assert deformableUtils.set_physics_surface_deformable_body(stage,'/World/Cloth')
        prim=mesh.GetPrim();assert prim.ApplyAPI('PhysxBaseDeformableBodyAPI')
        prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Set(64)
        prim.GetAttribute('physxDeformableBody:linearDamping').Set(.1)
        if args.interior:
            prim.GetAttribute('physxDeformableBody:selfCollision').Set(True)
            prim.ApplyAPI('PhysxCollisionAPI')
            prim.GetAttribute('physxCollision:contactOffset').Set(.001)
            prim.GetAttribute('physxCollision:restOffset').Set(.0005)
        cloth=DeformablePrim('/World/Cloth',deformable_type='surface',reset_xform_op_properties=False)
        m=SurfaceDeformableMaterial('/World/ClothMaterial',youngs_moduli=20000.,poissons_ratios=.3,densities=500.,dynamic_frictions=.5,static_frictions=.5)
        cloth.apply_physics_materials(m)
        matprim=stage.GetPrimAtPath('/World/ClothMaterial')
        assert matprim.GetAttribute('omniphysics:surfaceThickness').Set(.001)
        assert matprim.ApplyAPI('PhysxDeformableMaterialAPI')
        matprim.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(.05)
        UsdGeom.Imageable(mesh).MakeInvisible()
        visual=geometry.author_mesh(stage,'/World/ClothDisplay',points,faces,'surface',Usd,UsdGeom,Vt)
        visual.CreateDoubleSidedAttr(True)
        GeomPrim('/World/ClothDisplay').apply_visual_materials(material('/World/ClothVisual',[.025,.58,.69],.8))
        if args.interior:
            from deformable_tasks import specimen_appearance
            specimen_appearance.cloth_material(stage,visual,nx,nz,ROOT)
        # Display-only grid follows actual simulated nodes; it makes local strain visible.
        grid_indices=[];counts=[]
        for j in range(0,nz+1,4):
            ids=list(range(j*(nx+1),(j+1)*(nx+1)));grid_indices.extend(ids);counts.append(len(ids))
        for i in range(0,nx+1,4):
            ids=[j*(nx+1)+i for j in range(nz+1)];grid_indices.extend(ids);counts.append(len(ids))
        grid_indices=np.array(grid_indices)
        grid=UsdGeom.BasisCurves.Define(stage,'/World/MaterialGrid')
        grid.CreateTypeAttr('linear');grid.CreateCurveVertexCountsAttr(counts)
        grid.CreateWidthsAttr([.0007]);grid.SetWidthsInterpolation('constant')
        grid.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[grid_indices]))
        GeomPrim('/World/MaterialGrid').apply_visual_materials(material('/World/GridMaterial',[.012,.13,.19],.8))
        if args.interior:UsdGeom.Imageable(grid).MakeInvisible()
        box('/World/FixtureBase',[fixed[0],fixed[1],.022],[.25,.18,.044],graphite,True)
        box('/World/FixturePost',[fixed[0],fixed[1]+.045,(fixed[2]-.015)/2],[.045,.05,fixed[2]-.015],graphite)
        anchor=UsdGeom.Xform.Define(stage,'/World/FixedClamp')
        for side in [-1,1]:
            box('/World/FixedClamp/Jaw'+str(side+1),fixed+[0,side*.011,0],[width+.035,.02,.028],gold)
        # Wide pads are rigid children of the hand; the prescribed binding is stated in the report.
        for side in [-1,1]:
            pad_path='/World/Franka/panda_hand/PullPad'+str(side+1)
            pad=UsdGeom.Cube.Define(stage,pad_path);pad.CreateSizeAttr(1.)
            pad.AddTranslateOp().Set(Gf.Vec3d(0,side*.010,.10));pad.AddScaleOp().Set(Gf.Vec3f(width+.02,.017,.024))
            GeomPrim(pad_path,reset_xform_op_properties=False).apply_visual_materials(gold)
        if args.interior:record['appearance']=specimen_appearance.fixture(stage,fixed,width,out)
        def attach(path,indices,target,local):
            a=stage.DefinePrim(path,'OmniPhysicsVtxXformAttachment')
            assert a.GetRelationship('omniphysics:src0').SetTargets(['/World/Cloth'])
            assert a.GetRelationship('omniphysics:src1').SetTargets([target])
            assert a.GetAttribute('omniphysics:vtxIndicesSrc0').Set(Vt.IntArray.FromNumpy(indices))
            assert a.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        attach('/World/BottomBinding',bottom,'/World/FixedClamp',points[bottom])
        local_top=(points[top]-hand_expected[:3,3])@hand_expected[:3,:3]
        attach('/World/TopBinding',top,'/World/Franka/panda_hand',local_top)
        hand=RigidPrim('/World/Franka/panda_hand',reset_xform_op_properties=False)
        # Oblique front view, using USD camera look-at convention (-Z forward, +Y up).
        camera_pos=Gf.Vec3d(1.0,-1.65,.95);look=Gf.Vec3d(.21,0,.43)
        view=Gf.Matrix4d().SetLookAt(camera_pos,look,Gf.Vec3d(0,0,1)).GetInverse()
        cq=view.ExtractRotationQuat();quat=np.r_[cq.GetReal(),cq.GetImaginary()]
        if args.interior:camera_pos,quat=interior.camera_pose(0.)
        camera=RtxCamera('/World/Camera',translations=np.array(camera_pos),orientations=quat)
        cam_usd=UsdGeom.Camera(stage.GetPrimAtPath('/World/Camera'))
        cam_usd.GetFocalLengthAttr().Set(23.);cam_usd.GetHorizontalApertureAttr().Set(20.955)
        cam_usd.GetVerticalApertureAttr().Set(20.955*720/1280)
        if args.interior:cam_usd.GetFocalLengthAttr().Set(20.)
        camera.camera.set_clipping_ranges(.01,100.)
        sensor=None if args.physics_only else CameraSensor(camera,resolution=(720,1280),annotators=['rgb','distance_to_image_plane','normals'] if args.interior else ['rgb'])
        if args.interior:
            common.write_json(out/'camera-calibration.json',dict(width=1280,height=720,
                fx=1280*20./20.955,fy=720*20./(20.955*720/1280),cx=640.,cy=360.,
                depth='distance_to_image_plane, float32 meters; invalid values retained in NPZ',
                camera_pose='world position xyz meters, then quaternion wxyz; USD camera looks along -Z, +Y up',
                normals='Native RTX normals annotator, original channels preserved; visualization is signed-normal RGB',
                timestamps='time_s is elapsed recording time; simulation_time_s is actual physics clock'))
        if args.native_display:
            from omni.kit.viewport.utility import get_active_viewport
            vp=get_active_viewport()
            if vp:vp.camera_path='/World/Camera'
        stage.Export(str(out/'initial.usda'))
        app_utils.play();app.update();robot.reset_to_default_state();robot.set_dof_position_targets(q0)
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            robot.set_dof_gains(stiffnesses=8000.,dampings=200.,dof_indices=list(range(7)),update_default_gains=False)
            robot.set_dof_gains(stiffnesses=1500.,dampings=80.,dof_indices=[7,8],update_default_gains=False)
        SimulationManager.step(update_fabric=True)
        for _ in range(120):SimulationManager.step(update_fabric=True)
        hi=robot.link_names.index('panda_hand')
        record.update(status='simulating',material=dict(young_modulus_pa=20000,poisson_ratio=.3,density_kg_m3=500,thickness_m=.001),physics_hz=hz,position_iterations=64,cloth_nodes=len(points),cloth_triangles=len(faces),grip='Explicit vertex-to-rigid-hand attachment with wide-jaw adapter; bottom vertex-to-world attachment',pull_distance_m=.11)
        if args.interior:record['contact_settings']=dict(self_collision=True,contact_offset_m=.001,rest_offset_m=.0005)
        save()
        frames=out/'frames';frames.mkdir()
        states=[];times=[];hands=[];metrics=[];joint_states=[];camera_poses=[];simulation_times=[]
        quality=dict(physics_steps_checked=0,min_triangle_area_m2=1.,max_node_speed_m_s=0.,max_fixed_edge_error_m=0.,max_hand_binding_error_m=0.,min_cloth_z_m=1.)
        def read():
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                x=cloth.get_nodal_positions()[0].numpy()[0].copy()
                hp,hq=hand.get_world_poses();pos=hp.numpy()[0].copy();q=hq.numpy()[0].copy()
                joints=robot.get_dof_positions().numpy()[0].copy()
            assert x.shape==points.shape and np.isfinite(x).all()
            return x,pos,q,joints
        x,pos,hq,joints=read();origin=pos.copy()
        check('hand_initial_alignment',np.linalg.norm(origin-hand_expected[:3,3])<.005,error_m=float(np.linalg.norm(origin-hand_expected[:3,3])))
        if args.interior and args.native_display:
            # The recording sensor has its own render product. Avoid rendering
            # the identical live viewport a second time during offline export.
            vp.updates_enabled=False
        def capture(index,t):
            x,pos,hq,joints=read()
            visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x))
            visual.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([x.min(0),x.max(0)],dtype=np.float32)))
            grid.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x[grid_indices]))
            if args.interior:
                normals=np.zeros_like(x);tri=x[faces];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                for j in range(3):np.add.at(normals,faces[:,j],fn)
                normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
                visual.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals));visual.SetNormalsInterpolation('vertex')
                eye,cquat=interior.camera_pose(t)
                camera.camera.set_world_poses(positions=eye,orientations=cquat)
                camera_poses.append(np.r_[eye,cquat])
            before=SimulationManager.get_simulation_time()
            simulation_times.append(before)
            for _ in range(0 if args.physics_only else (max(12,args.render_samples//4+4) if args.interior else (24 if index==0 else 3))):RenderingManager.render()
            assert abs(SimulationManager.get_simulation_time()-before)<1e-8
            if sensor:
                rgb,_=sensor.get_data('rgb')
                common.write_rgb_png(frames/f'{index:05d}.png',np.ascontiguousarray(rgb.numpy()[:,:,:3]))
            if args.interior and sensor:
                depth,_=sensor.get_data('distance_to_image_plane');d=depth.numpy().copy()
                np.savez_compressed(out/f'depth-{index:05d}.npz',depth_m=d,camera_pose=camera_poses[-1],time_s=t,simulation_time_s=before)
                if index in (0,5*fps,10*fps):
                    normal,_=sensor.get_data('normals');n=normal.numpy().copy()
                    np.savez_compressed(out/f'normals-{index:05d}.npz',normals=n,camera_pose=camera_poses[-1],time_s=t)
                    common.write_rgb_png(out/f'normals-{index:05d}.png',np.ascontiguousarray(np.clip((n[:,:,:3]*.5+.5)*255,0,255).astype(np.uint8)))
            strips=x.reshape(nz+1,nx+1,3)
            rest=points.reshape(nz+1,nx+1,3)
            arclength=np.linalg.norm(np.diff(strips,axis=0),axis=2).sum(0)
            rest_arc=np.linalg.norm(np.diff(rest,axis=0),axis=2).sum(0)
            target_top=local_top@rotation(hq).T+pos
            data=dict(time_s=t,edge_separation_m=float(np.linalg.norm(x[top].mean(0)-x[bottom].mean(0))),mean_material_extension_percent=float(100*np.mean(arclength/rest_arc-1)),bottom_error_m=float(np.max(np.linalg.norm(x[bottom]-points[bottom],axis=1))),top_error_m=float(np.max(np.linalg.norm(x[top]-target_top,axis=1))),midwidth_m=float(np.linalg.norm(strips[nz//2,-1]-strips[nz//2,0])))
            metrics.append(data);states.append(x);times.append(t);hands.append(np.r_[pos,hq]);joint_states.append(joints)
            if index%30==0:
                record['latest_frame']=index;record['latest_metrics']=data;save();print(json.dumps(data),flush=True)
        fps=6 if args.preview else (24 if args.interior else 30)
        duration=10.;total=int(duration*hz);stride=hz//fps
        capture(0,0.)
        for step in range(1,total+1):
            t=step/hz
            s=np.clip((t-1.5)/5,0,1);rise=.11*(10*s**3-15*s**4+6*s**5)
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                current=robot.get_dof_positions().numpy()[0]
                jac=robot.get_jacobian_matrices().numpy()[0,hi-1,:,:7]
                hp,hq=hand.get_world_poses()
            pos=hp.numpy()[0];R=rotation(hq.numpy()[0]);goal=hand_expected[:3,:3]
            angular=.5*sum((np.cross(R[:,i],goal[:,i]) for i in range(3)))
            error=np.r_[origin+[0,0,rise]-pos,angular]
            dq=jac.T@np.linalg.solve(jac@jac.T+.0025*np.eye(6),error)
            target=q0.copy();target[:7]=current[:7]+np.clip(dq,-.04,.04)
            robot.set_dof_position_targets(target)
            SimulationManager.step(update_fabric=True)
            if args.interior:
                xx,pp,qq,_=read()
                with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):vv=cloth.get_nodal_velocities().numpy()[0]
                triangles=xx[faces];area=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
                quality['physics_steps_checked']+=1
                quality['min_triangle_area_m2']=min(quality['min_triangle_area_m2'],float(area.min()))
                quality['max_node_speed_m_s']=max(quality['max_node_speed_m_s'],float(np.linalg.norm(vv,axis=1).max()))
                quality['max_fixed_edge_error_m']=max(quality['max_fixed_edge_error_m'],float(np.linalg.norm(xx[bottom]-points[bottom],axis=1).max()))
                quality['max_hand_binding_error_m']=max(quality['max_hand_binding_error_m'],float(np.linalg.norm(xx[top]-(local_top@rotation(qq).T+pp),axis=1).max()))
                quality['min_cloth_z_m']=min(quality['min_cloth_z_m'],float(xx[:,2].min()))
                if not np.isfinite(vv).all() or area.min()<1e-10:raise RuntimeError('Invalid cloth state during physics substep')
            if step%stride==0:capture(step//stride,t)
        np.savez_compressed(out/'trajectory.npz',time_s=times,simulation_time_s=simulation_times,cloth_nodes=states,hand_poses=hands,joint_positions=joint_states,rest_nodes=points,faces=faces,bottom_indices=bottom,top_indices=top,camera_poses=camera_poses)
        common.write_json(out/'metrics.json',metrics)
        check('attachments_hold',max(m['bottom_error_m'] for m in metrics)<.005 and max(m['top_error_m'] for m in metrics)<.005,max_bottom_m=max(m['bottom_error_m'] for m in metrics),max_top_m=max(m['top_error_m'] for m in metrics))
        check('real_material_stretch',metrics[-1]['mean_material_extension_percent']>20,initial=metrics[0],final=metrics[-1])
        check('robot_moves',hands[-1][2]-hands[0][2]>.09,hand_rise_m=float(hands[-1][2]-hands[0][2]))
        if args.interior:
            record['all_step_quality']=quality
            check('all_step_stability',quality['min_triangle_area_m2']>1e-10 and quality['max_fixed_edge_error_m']<.005 and quality['max_hand_binding_error_m']<.005 and quality['max_node_speed_m_s']<2.,**quality)
            check('countertop_clearance',quality['min_cloth_z_m']>.02,min_clearance_m=quality['min_cloth_z_m'])
            record['rendering']=dict(mode='physics_only' if args.physics_only else 'native_path_tracing',requested_samples=args.render_samples,camera='continuous physical dolly',resolution=[1280,720],source='live GPU simulation, no precomputed mesh animation')
        record.update(frame_count=len(metrics),fps=fps,duration_s=duration)
        save()
        if args.interior and not args.preview and not args.physics_only:
            # A separate native square camera resolves the textile and machining.
            # Physics stays paused at the final measured state throughout capture.
            detail_eye=Gf.Vec3d(.40,-.59,.46);detail_target=Gf.Vec3d(.307,0,.405)
            transform=Gf.Matrix4d().SetLookAt(detail_eye,detail_target,Gf.Vec3d(0,0,1)).GetInverse()
            dq=transform.ExtractRotationQuat();detail_quat=np.r_[dq.GetReal(),dq.GetImaginary()]
            detail_camera=RtxCamera('/World/DetailCamera',translations=np.array(detail_eye),orientations=detail_quat)
            detail_usd=UsdGeom.Camera(stage.GetPrimAtPath('/World/DetailCamera'))
            detail_usd.GetFocalLengthAttr().Set(25.);detail_usd.GetHorizontalApertureAttr().Set(20.955);detail_usd.GetVerticalApertureAttr().Set(20.955)
            detail_camera.camera.set_clipping_ranges(.01,100.)
            detail_sensor=CameraSensor(detail_camera,resolution=(1920,1920),annotators=['rgb','distance_to_image_plane','normals'])
            settings.set('/rtx/pathtracing/totalSpp',128)
            before=SimulationManager.get_simulation_time()
            for _ in range(40):RenderingManager.render()
            assert abs(SimulationManager.get_simulation_time()-before)<1e-8
            rgb,_=detail_sensor.get_data('rgb');depth,_=detail_sensor.get_data('distance_to_image_plane');normal,_=detail_sensor.get_data('normals')
            common.write_rgb_png(out/'cloth-detail.png',np.ascontiguousarray(rgb.numpy()[:,:,:3]))
            np.savez_compressed(out/'detail-sensors.npz',depth_m=depth.numpy(),normals=normal.numpy(),camera_pose=np.r_[detail_eye,detail_quat],simulation_time_s=before)
            common.write_json(out/'detail-camera.json',dict(width=1920,height=1920,fx=1920*25/20.955,fy=1920*25/20.955,cx=960,cy=960,camera_pose=np.r_[detail_eye,detail_quat].tolist(),simulation_time_s=before,requested_samples=128))
            record['detail_capture']=dict(resolution=[1920,1920],requested_samples=128,source='Native camera at final live physics state; no image synthesis');save()
        if args.native_display:
            if args.interior:
                vp.updates_enabled=True
                # Initialize debug-draw/viewport resources before native shutdown,
                # including dynamics-only runs, so stop callbacks see valid handles.
                for _ in range(4 if args.physics_only else 24):RenderingManager.render()
            # -I prevents Kit's PYTHONHOME/PYTHONPATH from contaminating system Python.
            if not args.physics_only:subprocess.run(['/usr/bin/python3','-I',str(ROOT/'scripts/capture_task_display.py'),'--output',str(out/'native-desktop.png')],check=True)
        record.update(status='pending_official_fast_exit',exit_code=0,frame_count=len(metrics),fps=fps,duration_s=duration)
    except Exception as e:
        record.update(status='failed',exit_code=1,error=str(e),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        save()
        if app is not None:app.close(exit_code=record['exit_code'])
    return record['exit_code']
if __name__=='__main__':raise SystemExit(main())
