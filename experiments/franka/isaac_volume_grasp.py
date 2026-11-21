"""Actuated Franka volume pinch/lift/release and matched negative controls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import time
from grasp_motion import lift_fraction
from transfer_task import HZ,END,command,fixture,score

ROOT = Path(os.environ.get('DEFORMLABSIM_ROOT', '.')).resolve()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ray_depth(points, faces, pixels, camera, focal, width, height):
    """Independent two-sided ray/triangle intersection in world coordinates."""
    import numpy as np
    triangles = points[faces]
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    rays = np.column_stack(((pixels[:, 1] + .5 - width / 2) / focal,
                           np.ones(len(pixels)), -(pixels[:, 0] + .5 - height / 2) / focal))
    cross = np.cross(rays[:, None], edge2[None])
    det = np.sum(edge1[None] * cross, axis=2)
    inv = np.divide(1, det, out=np.zeros_like(det), where=np.abs(det) > 1e-9)
    origin = camera - triangles[:, 0]
    u = np.sum(origin[None] * cross, axis=2) * inv
    q = np.cross(origin, edge1)
    v = np.sum(rays[:, None] * q[None], axis=2) * inv
    t = np.sum(edge2 * q, axis=1)[None] * inv
    valid = (np.abs(det) > 1e-9) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.min(np.where(valid, t, np.inf), axis=1)


def rotation(q):
    import numpy as np
    w,x,y,z=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    sys.path.insert(0,str(ROOT))
    from deformable_tasks.robot_config import ROBOT_BACKENDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--build-receipt', required=True)
    parser.add_argument('--native-display', action='store_true')
    parser.add_argument('--grip-mode', choices=['close','open'], default='close')
    parser.add_argument('--robot-collision', choices=['on', 'off'], required=True)
    parser.add_argument('--physics-hz',type=int,choices=[240,480,960],default=240)
    parser.add_argument('--position-iterations',type=int,default=64)
    parser.add_argument('--contact-offset',type=float,default=.001)
    parser.add_argument('--speculative-ccd',action='store_true')
    parser.add_argument('--block-yaw-deg',type=float,default=0.)
    parser.add_argument('--block-offset-x',type=float,default=0.)
    parser.add_argument('--lift-height',type=float,default=.1)
    parser.add_argument('--record-finger-wrenches',action='store_true')
    parser.add_argument('--cp-conservative-trial',choices=['none','split-velocity','split-sequential','split-stale','pass-through'],default='none')
    parser.add_argument('--hold-arm-target',choices=['feedback','fixed'],default='feedback',
                        help='Diagnostic: keep the last arm target during hold; dynamics and finger drives remain active')
    parser.add_argument('--trace-hold-kernels',action='store_true')
    parser.add_argument('--elasticity-damping',type=float,choices=[0.,.05],default=.05)
    parser.add_argument('--lift-profile',choices=['linear','quintic'],default=None,
                        help='Default: quintic with the experimental CP bridge, linear on the legacy path')
    parser.add_argument('--unified-task-config')
    parser.add_argument('--unified-backend',choices=ROBOT_BACKENDS)
    parser.add_argument('--no-captures',action='store_true')
    parser.add_argument('--zero-settling',action='store_true')
    parser.add_argument('--surface-runtime',choices=['off','persistent'],required=True)
    parser.add_argument('--visual-controller',choices=['rgbd'],required=True)
    parser.add_argument('--actuation-offset-m',type=float,required=True)
    parser.add_argument('--goal-dir',required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    case=json.loads(Path(args.case).read_text())
    if not args.no_captures:parser.error("Regrasp development records RGB-D without presentation captures")
    if not -.015<=args.actuation_offset_m<=.015:parser.error('Actuation offset outside pilot bounds')
    if args.cp_conservative_trial!='split-velocity' or args.physics_hz!=HZ:parser.error('Transfer requires matched mapping and native physics rate')
    unified=None;backend_meta=None
    side_m=.04;young_pa=50000.;density=1000.;poisson=.3;friction=.5;resolution=4
    if bool(args.unified_task_config)!=bool(args.unified_backend):parser.error('Unified config and backend must be supplied together')
    if args.unified_task_config:
        sys.path.insert(0,str(ROOT))
        from paper_vision_regrasp_config import load_robot_config,robot_plan
        unified,backend_meta=robot_plan(load_robot_config(args.unified_task_config),args.unified_backend)
        if args.cp_conservative_trial!=backend_meta['trial_mode']:parser.error('Robot backend/trial mismatch')
        m=unified['material'];g=unified['geometry'];solver=unified['solver'];action=unified['action']
        args.physics_hz=solver['hz'];args.position_iterations=solver['position_iterations']
        args.contact_offset=solver['contact_offset_m'];args.speculative_ccd=solver['speculative_ccd'];args.zero_settling=solver['zero_settling']
        args.grip_mode=action['grip_mode'];args.robot_collision='on' if action['robot_collision'] else 'off'
        args.lift_height=action['lift_height_m'];args.lift_profile=action['lift_profile'];args.hold_arm_target=action['hold_arm_target']
        args.block_yaw_deg=g['yaw_deg'];args.block_offset_x=g['offset_x_m'];args.elasticity_damping=m['elasticity_damping']
        args.record_finger_wrenches=unified['measurement']['finger_wrenches']
        side_m=g['size_m'];resolution=g['resolution'];young_pa=m['young_pa'];density=m['density'];poisson=m['poisson'];friction=m['friction']
    side_m=case['side_m'];young_pa=case['young_pa']
    trial=args.cp_conservative_trial!='none'
    if args.lift_profile is None:args.lift_profile='quintic' if trial else 'linear'
    if args.trace_hold_kernels and not trial:parser.error('Hold kernel trace requires the explicit CP driver bridge')
    if trial and (args.physics_hz!=960 or args.position_iterations not in [16,32,64]):parser.error('CP experiment supports only the audited 960 Hz / 16, 32 or 64 requested position iterations')
    if not 1<=args.position_iterations<=255 or not 0<args.contact_offset<=.02: parser.error('Invalid contact/iteration setting')
    if not abs(args.block_yaw_deg)<=30 or not abs(args.block_offset_x)<=.008 or not .09<=args.lift_height<=.2:
        parser.error('Perturbation outside the supported grasp benchmark range')
    substeps=args.physics_hz//240
    sys.argv = [sys.argv[0]]
    out = (ROOT / args.output_dir).resolve()
    assert out.is_relative_to(Path('.cache/deformlabsim-ral-priority-20261119/franka')) and out.is_dir()
    common = load('joint_common', ROOT / 'scripts/isaac_platform_smoke.py')
    geometry = load('joint_geometry', ROOT / 'scripts/isaac_deformable_baseline.py')
    record = dict(kind='franka_rgbd_release_reacquire_regrasp', status='starting', checks={}, captures=[],
                  robot_collision=args.robot_collision, full_platform_passed=False,
                  script_sha256=common.sha256(Path(__file__)), geometry_helper_sha256=common.sha256(ROOT/'scripts/isaac_deformable_baseline.py'),
                  force_accuracy_verified=False, original_solver_contribution_verified=False)
    record['exit_code']=1
    app = None
    cp_trial=None
    surface=None
    if unified is not None:
        record.update(task='robot_visual_regrasp',effective_config=unified,backend=backend_meta)
    record['render_validation']=dict(requested=not args.no_captures,status='not_requested' if args.no_captures else 'pending')

    def save():
        common.write_json(out / 'report.json', record)

    def check(name, passed, **evidence):
        record['checks'][name] = dict(passed=bool(passed), **evidence)
        save()
        if not passed and name not in ('no_inverted_or_collapsed_tets','visual_reacquisition_ready','unsupported_regrasp_hold','tabletop_penetration'):
            raise RuntimeError('Failed integration check: ' + name)

    try:
        binding = common.runtime_binding(ROOT / 'upstream/IsaacSim', os.environ.get('ISAAC_PATH'), sys.executable, '')
        check('runtime_binding', binding['passed'])
        receipt = common.verify_build_receipt((ROOT / args.build_receipt).resolve(), ROOT / 'upstream/IsaacSim', binding)
        check('receipt', receipt['passed'])
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=not args.native_display, fast_shutdown=True, multi_gpu=False,
            renderer='RaytracedLighting', extra_args=['--/app/useFabricSceneDelegate=1',
            '--/app/extensions/pythonFullLeakCheck=false', '--/crashreporter/enabled=false']),
            experience=binding['experience'])
        import numpy as np
        import carb.settings
        from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Vt
        from omni.physx.scripts import deformableUtils
        from isaacsim.core.experimental.utils import stage as stage_utils, app as app_utils
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.experimental.prims import Articulation, DeformablePrim, GeomPrim, RigidPrim
        from isaacsim.core.experimental.objects import Cube, DistantLight
        from isaacsim.core.experimental.materials import VolumeDeformableMaterial, PreviewSurfaceMaterial
        from isaacsim.core.simulation_manager import SimulationManager, PhysxScene
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
        from isaacsim.storage.native import get_assets_root_path

        check('official_runtime', SimulationManager.get_active_physics_engine() == 'physx'
              and carb.settings.get_settings().get('/app/fastShutdown') is True)
        stage_utils.create_new_stage()
        stage = stage_utils.get_current_stage(backend='usd')
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        stage.SetDefaultPrim(stage_utils.define_prim('/World', 'Xform'))
        stage.SetTimeCodesPerSecond(args.physics_hz)
        scene = PhysxScene('/World/PhysicsScene')
        scene.set_steps_per_second(args.physics_hz)
        scene.set_gravity([0, 0, -9.81])
        if trial or unified is not None:
            scene_prim=stage.GetPrimAtPath('/World/PhysicsScene')
            assert scene_prim.GetAttribute('physxScene:solverType').Set('TGS')
            assert scene_prim.GetAttribute('physxScene:enableExternalForcesEveryIteration').Set(True)
        if unified is not None:
            scene_prim=stage.GetPrimAtPath('/World/PhysicsScene')
            actual_scene=dict(solver_type=scene_prim.GetAttribute('physxScene:solverType').Get(),external_forces_every_iteration=scene_prim.GetAttribute('physxScene:enableExternalForcesEveryIteration').Get())
            record['physics_scene_settings']=actual_scene
            check('matched_physics_scene',actual_scene['solver_type']==unified['solver']['solver_type'] and actual_scene['external_forces_every_iteration']==unified['solver']['external_forces_every_iteration'])
        SimulationManager.set_device('cuda:0')
        Cube('/World/Floor', sizes=1., scales=[6, 6, .2], positions=[0, 0, -.1])
        GeomPrim('/World/Floor', apply_collision_apis=True)
        light = DistantLight('/World/Light')
        light.set_intensities(1500.)
        fill=UsdLux.DistantLight.Define(stage,'/World/FillLight')
        fill.CreateIntensityAttr(900.);fill.AddRotateXYZOp().Set(Gf.Vec3f(45,0,0))
        asset = get_assets_root_path() + '/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda'
        record['robot_asset'] = asset
        record['status'] = 'loading_franka'
        save()
        stage_utils.add_reference_to_stage(usd_path=asset, path='/World/Franka',
                                         variants=[('Gripper', 'alternatefinger'), ('Mesh', 'quality')])
        for _ in range(600):
            app.update()
            if not stage_utils.is_stage_loading():
                break
        # Expand instance roots so the OFF intervention reaches every collision shape.
        for _ in range(20):
            instances = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')) if prim.IsInstance()]
            if not instances: break
            for prim in instances: assert prim.SetInstanceable(False)
        assert not instances, 'Unexpanded robot instances'
        colliders = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka'))
                     if prim.HasAPI(UsdPhysics.CollisionAPI)]
        check('robot_asset_loaded', len(colliders) > 0 and not stage_utils.is_stage_loading(), collider_count=len(colliders))
        original_flags = {str(prim.GetPath()): bool(prim.GetAttribute('physics:collisionEnabled').Get()) for prim in colliders}
        if args.robot_collision == 'off':
            for prim in colliders: assert prim.GetAttribute('physics:collisionEnabled').Set(False)
        check('collision_flags_authored', all(prim.GetAttribute('physics:collisionEnabled').Get() == (original_flags[str(prim.GetPath())] if args.robot_collision == 'on' else False) for prim in colliders), original_flags=original_flags)
        for prim in colliders:
            assert prim.ApplyAPI('PhysxCollisionAPI')
            assert prim.GetAttribute('physxCollision:contactOffset').Set(args.contact_offset)
            assert prim.GetAttribute('physxCollision:restOffset').Set(0.)
        articulation_roots=[prim for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(articulation_roots)!=1:raise RuntimeError('Expected one articulation root')
        for prim in articulation_roots:
            assert prim.ApplyAPI('PhysxArticulationAPI')
            assert prim.GetAttribute('physxArticulation:solverPositionIterationCount').Set(args.position_iterations)
            assert prim.GetAttribute('physxArticulation:solverVelocityIterationCount').Set(1)
        record['articulated_budget']=dict(requested_position_iterations=args.position_iterations,velocity_iterations=1,
            roots={str(prim.GetPath()):prim.GetAttribute('physxArticulation:solverPositionIterationCount').Get() for prim in articulation_roots})
        check('articulation_budget_authored',all(value==args.position_iterations for value in record['articulated_budget']['roots'].values()))
        robot = Articulation('/World/Franka')
        links = RigidPrim([str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Franka')) if prim.HasAPI(UsdPhysics.RigidBodyAPI)], reset_xform_op_properties=False)
        q0 = np.array([0., -.785, 0., -2.356, 0., 1.571, .785, .04, .04], dtype=np.float32)
        robot.set_default_state(dof_positions=q0)
        # Derive initial object placement from asset joint frames, then verify against live physics.
        def transform(pos, quat):
            q = np.r_[float(quat.GetReal()), np.asarray(quat.GetImaginary(), dtype=float)]
            T = np.eye(4); T[:3,:3] = rotation(q); T[:3,3] = np.asarray(pos); return T
        poses = {'/World/Franka/panda_link0': np.eye(4)}
        joint_values = {f'panda_joint{i+1}':float(q0[i]) for i in range(7)}
        joint_values.update(panda_finger_joint1=.04, panda_finger_joint2=.04)
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
        record['setup']=dict(grip_mode=args.grip_mode, block_center=center.tolist(),side_m=side_m,
                            young_modulus_pa=young_pa,density_kg_m3=density,poisson_ratio=poisson,friction=friction,resolution=resolution,zero_settling=bool(trial or args.zero_settling),
                            physics_hz=args.physics_hz,control_hz=240,position_iterations=args.position_iterations,contact_offset_m=args.contact_offset,rest_offset_m=0.,linear_damping=0.,elasticity_damping=args.elasticity_damping,
                            speculative_ccd=args.speculative_ccd,
                            block_yaw_deg=args.block_yaw_deg,block_offset_x_m=args.block_offset_x,lift_height_m=args.lift_height,
                            lift_profile=args.lift_profile,motion_source_sha256=common.sha256(ROOT/'scripts/grasp_motion.py'),
                            record_finger_wrenches=args.record_finger_wrenches,
                            finger_drive_stiffness=1000.,finger_drive_damping=80.,effort_limit='unchanged asset',
                            hand_expected=hand_expected.tolist())
        table_top=float(center[2]-side_m/2)
        Cube('/World/Table',sizes=1.,scales=[.16,.16,.04],positions=[center[0],center[1],table_top-.02])
        GeomPrim('/World/Table',apply_collision_apis=True)
        for path in ['/World/Table','/World/Floor']:
            prim=stage.GetPrimAtPath(path); assert prim.ApplyAPI('PhysxCollisionAPI')
            assert prim.GetAttribute('physxCollision:contactOffset').Set(args.contact_offset)
            assert prim.GetAttribute('physxCollision:restOffset').Set(0.)
        slot_xy,slot_inner,rails=fixture(center,table_top,side_m,case['opening_extra_m'])
        rail_shader=PreviewSurfaceMaterial('/World/SlotFinish');rail_shader.set_input_values('diffuseColor',[.48,.24,.085])
        for i,(xyz,size) in enumerate(rails):
            path=f'/World/SlotRail{i}'
            Cube(path,sizes=1.,scales=size,positions=xyz)
            g=GeomPrim(path,apply_collision_apis=True)
            p=stage.GetPrimAtPath(path); assert p.ApplyAPI('PhysxCollisionAPI')
            assert p.GetAttribute('physxCollision:contactOffset').Set(args.contact_offset)
            assert p.GetAttribute('physxCollision:restOffset').Set(0.)
            g.apply_visual_materials(rail_shader)
        record['slot']=dict(center_xy=slot_xy.tolist(),half_opening_m=slot_inner.tolist(),support_top_m=table_top,
            rails=[dict(center=x.tolist(),full_size=d.tolist()) for x,d in rails])
        points,tets=geometry.build_mesh('volume',resolution)
        points=(points-[0,0,.95])*(side_m*2)+center
        yaw=np.deg2rad(args.block_yaw_deg)
        points=(points-center) @ rotation([np.cos(yaw/2),0,0,np.sin(yaw/2)]).T+center+[args.block_offset_x,case['offset_y_m'],0]
        faces=geometry.boundary_triangles(tets)
        mesh=geometry.author_mesh(stage,'/World/SoftBlock',points,tets,'volume',Usd,UsdGeom,Vt)
        assert deformableUtils.set_physics_volume_deformable_body(stage,'/World/SoftBlock')
        assert mesh.GetPrim().ApplyAPI('PhysxBaseDeformableBodyAPI')
        assert mesh.GetPrim().GetAttribute('physxDeformableBody:solverPositionIterationCount').Set(args.position_iterations)
        assert mesh.GetPrim().GetAttribute('physxDeformableBody:linearDamping').Set(0.)
        if trial or args.zero_settling:
            for key in ['settlingDamping','settlingThreshold','sleepThreshold']:
                assert mesh.GetPrim().GetAttribute('physxDeformableBody:'+key).Set(0.)
        assert mesh.GetPrim().GetAttribute('physxDeformableBody:enableSpeculativeCCD').Set(args.speculative_ccd)
        assert mesh.GetPrim().ApplyAPI('PhysxCollisionAPI')
        assert mesh.GetPrim().GetAttribute('physxCollision:contactOffset').Set(args.contact_offset)
        assert mesh.GetPrim().GetAttribute('physxCollision:restOffset').Set(0.)
        cloth=DeformablePrim('/World/SoftBlock',deformable_type='volume',reset_xform_op_properties=False)
        material=VolumeDeformableMaterial('/World/SoftMaterial',youngs_moduli=young_pa,poissons_ratios=poisson,densities=density,dynamic_frictions=friction,static_frictions=friction)
        cloth.apply_physics_materials(material)
        mat=stage.GetPrimAtPath('/World/SoftMaterial');assert mat.ApplyAPI('PhysxDeformableMaterialAPI')
        assert mat.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(args.elasticity_damping)
        UsdGeom.Imageable(mesh).MakeInvisible()
        visual=geometry.author_mesh(stage,'/World/SoftDisplay',points,faces,'surface',Usd,UsdGeom,Vt)
        shader=PreviewSurfaceMaterial('/World/SoftVisual');shader.set_input_values('diffuseColor',[.02,.7,.85])
        GeomPrim('/World/SoftDisplay').apply_visual_materials(shader)
        hand=RigidPrim('/World/Franka/panda_hand',reset_xform_op_properties=False)
        fingers=RigidPrim(['/World/Franka/panda_leftfinger','/World/Franka/panda_rightfinger'],reset_xform_op_properties=False)
        camera_yaw=np.pi/4
        camera_yaw_rotation=rotation([np.cos(camera_yaw/2),0,0,np.sin(camera_yaw/2)])
        cam_pos=center+camera_yaw_rotation @ np.array([0.,-.65,.05])
        camera_quaternion=np.array([np.cos(camera_yaw/2),np.cos(camera_yaw/2),np.sin(camera_yaw/2),np.sin(camera_yaw/2)])/np.sqrt(2)
        record['camera']=dict(position=cam_pos.tolist(),quaternion_wxyz=camera_quaternion.tolist(),resolution=[640,480],focal_length=24.,horizontal_aperture=20.955,yaw_rad=float(camera_yaw))
        cam_pos=center+np.array([.65,-.53,.41])
        look=center+np.array([-.047,0.,-.035])
        back=cam_pos-look;back/=np.linalg.norm(back)
        right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
        up=np.cross(back,right)
        camera = RtxCamera('/World/Camera')
        cp=stage.GetPrimAtPath('/World/Camera')
        cam_xf=UsdGeom.Xformable(cp);cam_xf.ClearXformOpOrder()
        cam_xf.AddTransformOp().Set(Gf.Matrix4d(*np.vstack([np.r_[right,0.],np.r_[up,0.],np.r_[back,0.],np.r_[cam_pos,1.]]).ravel().tolist()))
        record['camera']=dict(position=cam_pos.tolist(),look_at=look.tolist(),resolution=[640,480],focal_length=24.)
        camera_usd = UsdGeom.Camera(stage.GetPrimAtPath('/World/Camera'))
        camera_usd.GetFocalLengthAttr().Set(24.)
        camera_usd.GetHorizontalApertureAttr().Set(20.955)
        camera_usd.GetVerticalApertureAttr().Set(20.955 * 480 / 640)
        camera.camera.set_clipping_ranges(.01, 100.)
        sensor = CameraSensor(camera, resolution=(480, 640), annotators=['rgb', 'distance_to_image_plane'])
        if args.native_display:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport: viewport.camera_path = '/World/Camera'
        stage.Export(str(out / 'initial.usda'))
        app_utils.play()
        app.update()
        robot.reset_to_default_state()
        robot.set_dof_position_targets(q0)
        with use_backend('tensor', raise_on_fallback=True, raise_on_unsupported=True):
            robot.set_dof_gains(stiffnesses=1000., dampings=80., dof_indices=[7,8], update_default_gains=False)
        SimulationManager.step(update_fabric=True)
        focal = 640 * 24 / 20.955
        times, states, joints, targets = [], [], [], []
        hand_states, finger_states, qualities, forces = [], [], [], []
        finger_joint_forces,finger_joint_torques=[],[]
        finger_indices=[robot.link_names.index(name) for name in ['panda_leftfinger','panda_rightfinger']]
        if args.record_finger_wrenches:
            child_frames=[]
            for name in ['panda_leftfinger','panda_rightfinger']:
                joint=next(j for j in pending if j.GetBody1Rel().GetTargets() and str(j.GetBody1Rel().GetTargets()[0])=='/World/Franka/'+name)
                q=joint.GetLocalRot1Attr().Get()
                child_frames.append(dict(position=np.asarray(joint.GetLocalPos1Attr().Get()).tolist(),
                    quaternion_wxyz=[float(q.GetReal()),*np.asarray(q.GetImaginary()).tolist()]))
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                finger_masses=robot.get_link_masses(link_indices=finger_indices).numpy()[0].tolist()
            record['finger_wrench_metadata']=dict(frame='joint child frame',child_frames=child_frames,
                link_indices=finger_indices,link_masses_kg=finger_masses,scope='Incoming joint wrench, not direct contact force')
        task_references=[[.04,0.,0.]]
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            hp,hq=hand.get_world_poses()
            rest=cloth.get_nodal_positions()[2].numpy()[0].copy()
            actual_elements=cloth.get_element_indices()
            sim_tets=actual_elements[0].numpy()[0].copy()
            rest_tets=actual_elements[2].numpy()[0].copy()
        check('runtime_topology_matches_authored',np.array_equal(sim_tets,tets) and np.array_equal(rest_tets,tets),
              simulation_shape=list(sim_tets.shape),rest_shape=list(rest_tets.shape))
        check('runtime_rest_matches_authored',np.max(np.abs(rest-points))<1e-6,max_error_m=float(np.max(np.abs(rest-points))))
        check('initial_fk_matches_physics',np.linalg.norm(hp.numpy()[0]-hand_expected[:3,3])<.005,
              predicted=hand_expected[:3,3].tolist(),actual=hp.numpy()[0].tolist())
        hand_index=robot.link_names.index('panda_hand')
        rest_volumes=geometry.signed_tet_volumes(rest,tets)
        check('rest_tets_positive',bool(np.all(rest_volumes>0)))
        native_positions=[];native_velocities=[]
        if trial:
            from cp_trial_bridge import ConservativeCpTrial
            from omni.physx import get_physx_interface
            import warp as wp
            from deformable_tasks.weight_backend import WEIGHT_BACKENDS
            from paper_vision_crosscondition_lifecycle import initialize
            cp_trial=initialize(ConservativeCpTrial.__new__(ConservativeCpTrial), lambda obj: ConservativeCpTrial.__init__(obj,ROOT,get_physx_interface(),'/World/SoftBlock',args.cp_conservative_trial,weight_backend=args.unified_backend if args.unified_backend in WEIGHT_BACKENDS else None),out/'constructor-cleanup-receipt.json')
            if args.surface_runtime=='persistent':
                from robot_surface_live_bridge import RobotSurfaceLive
                surface=RobotSurfaceLive(cp_trial,Path(os.environ['FRANKA_ROWS_PTX']),sim_tets,center,table_top,args.position_iterations)
                record['surface_runtime']=surface.evidence
            else:record['surface_runtime']=dict(mode='off',execution_verified=False)
            record['cp_conservative_trial']=cp_trial.evidence
            record['cp_conservative_trial'].update(external_every_iteration=True,zero_settling=True)
            if args.trace_hold_kernels:
                cp_trial.trace_start(out/'hold-kernels.tsv')
                record['hold_kernel_trace']=dict(steps=list(range(4800,4808)),scope='Read-only host launch boundary snapshots; synchronization changes scheduling. Compare with an untraced control.')

        def sample(target):
            with use_backend('tensor', raise_on_fallback=True, raise_on_unsupported=True):
                nodes = cloth.get_nodal_positions()[0]
                q = robot.get_dof_positions()
                hp,hq=hand.get_world_poses(); fp,fq=fingers.get_world_poses()
                joint_forces=robot.get_dof_projected_joint_forces()
                if args.record_finger_wrenches:
                    incoming_f,incoming_t=robot.get_link_incoming_joint_force(link_indices=finger_indices)
            assert str(nodes.device) == 'cuda:0' and str(q.device) == 'cuda:0'
            x, actual = nodes.numpy()[0].copy(), q.numpy()[0].copy()
            assert np.isfinite(x).all() and np.isfinite(actual).all()
            hand_states.append(np.r_[hp.numpy()[0],hq.numpy()[0]])
            finger_states.append(np.concatenate([fp.numpy(),fq.numpy()],axis=1))
            forces.append(joint_forces.numpy()[0].copy())
            if cp_trial:
                h=cp_trial.handle
                px=wp.array(ptr=h['device_pointer'],shape=(h['node_count'],),dtype=wp.vec4,device='cuda:0').numpy().copy()
                vv=wp.array(ptr=h['velocity_pointer'],shape=(h['node_count'],),dtype=wp.vec4,device='cuda:0').numpy().copy()
                assert np.max(abs(px[:,:3]-x))<1e-6 and np.all(px[:,3]>0) and np.isfinite(vv).all()
                native_positions.append(px);native_velocities.append(vv)
            if args.record_finger_wrenches:
                finger_joint_forces.append(incoming_f.numpy()[0].copy())
                finger_joint_torques.append(incoming_t.numpy()[0].copy())
            qualities.append(float(np.min(geometry.signed_tet_volumes(x,tets)/rest_volumes)))
            visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(x))
            visual.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([x.min(axis=0), x.max(axis=0)], dtype=np.float32)))
            times.append(float(SimulationManager.get_simulation_time()))
            states.append(x)
            joints.append(actual)
            targets.append(target.copy())
            return x

        def capture(index, x):
            before = float(SimulationManager.get_simulation_time())
            with use_backend('tensor', raise_on_fallback=True, raise_on_unsupported=True):
                link_p, link_q = links.get_world_poses()
            with use_backend('fabric', raise_on_fallback=True, raise_on_unsupported=True):
                render_p, render_q = links.get_world_poses()
            link_error = float(np.max(np.abs(link_p.numpy() - render_p.numpy())))
            quaternion_error = float(np.max(1-np.abs(np.sum(link_q.numpy()*render_q.numpy(),axis=1))))
            check('robot_render_pose_'+str(index), link_error < 1e-4 and quaternion_error < 1e-4,
                  position_max_error_m=link_error, quaternion_inner_product_error=quaternion_error)
            for _ in range(30):
                RenderingManager.render()
            rgb_data, rgb_info = sensor.get_data('rgb')
            dep_data, dep_info = sensor.get_data('distance_to_image_plane')
            assert rgb_data is not None and dep_data is not None
            rgb, depth = rgb_data.numpy().copy(), dep_data.numpy().copy().reshape(480, 640)
            assert rgb.shape[:2] == (480, 640) and np.isfinite(rgb).all()
            assert not np.isnan(depth).any() and not np.isneginf(depth).any()
            r = rgb[:, :, :3].astype(float)
            mask = (r[:, :, 1] > r[:, :, 0] * 1.5) & (r[:, :, 2] > r[:, :, 0] * 1.5) & (r[:, :, 2] > 30)
            interior = mask.copy()
            for axis in [0, 1]:
                for offset in [-2, -1, 1, 2]:
                    interior &= np.roll(mask, offset, axis=axis)
            pixels = np.argwhere(interior)
            selected = pixels[::max(1, len(pixels) // 250)]
            predicted = ray_depth((x-cam_pos) @ camera_yaw_rotation + cam_pos, faces, selected, cam_pos, focal, 640, 480)
            measured = depth[selected[:, 0], selected[:, 1]]
            valid = np.isfinite(predicted) & np.isfinite(measured)
            errors = np.abs(measured[valid] - predicted[valid])
            capture = dict(index=index, time_s=before, physics_time_after_render=float(SimulationManager.get_simulation_time()),
                volume_pixels=int(mask.sum()), valid_volume_rays=int(valid.sum()),
                median_depth_error_m=float(np.median(errors)) if len(errors) else None,
                depth_fraction_within_3cm=float(np.mean(errors < .03)) if len(errors) else 0.,
                rgb_info=common.annotator_frame_info(rgb_info), depth_info=common.annotator_frame_info(dep_info))
            capture['passed'] = bool(len(errors) >= 30 and capture['median_depth_error_m'] < .015
                and capture['depth_fraction_within_3cm'] >= .9 and abs(capture['physics_time_after_render']-before) < 1e-9)
            record['captures'].append(capture)
            np.savez_compressed(out / f'capture-{index:03d}.npz', rgb=rgb, depth=depth, volume_nodes=x,
                                tested_pixels=selected, predicted_depth=predicted)
            common.write_rgb_png(out / f'capture-{index:03d}.png', np.ascontiguousarray(rgb[:, :, :3]))
            if args.native_display and index == 1080:
                import subprocess
                subprocess.run(['/usr/bin/python3', str(ROOT / 'work/capture_project_x11.py'),
                                '--output', str(out / 'native-desktop.png')], check=True, timeout=30,
                               env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
            save()

        import warp as wp
        record['status']='simulating';record['case']=case
        record['control']=dict(kind='shared Cartesian reference with native joint feedback', physics_hz=HZ,
           control_hz=240, rendering_hz=10, object_feedback=False, attachment=False)
        save()
        initial=sample(q0);origin=hand_states[0][:3].copy();target=q0.copy()
        frames=[]
        def capture_frame():
            before=SimulationManager.get_simulation_time()
            for _ in range(4):RenderingManager.render()
            rgb,_=sensor.get_data('rgb')
            if rgb is None:raise RuntimeError('No native RGB frame')
            arr=np.ascontiguousarray(rgb.numpy()[:,:,:3])
            assert arr.shape==(480,640,3) and arr.dtype==np.uint8
            assert abs(SimulationManager.get_simulation_time()-before)<1e-9
            frames.append(arr)
        for _ in range(30):RenderingManager.render()
        capture_frame()
        wp.synchronize_device('cuda:0')
        total=int(END*HZ);start=time.perf_counter();capture_s=0.;readback_s=0.;diagnostic_s=0.
        for step in range(1,total+1):
            if (step-1)%substeps==0:
                opening,xyz=command((step-1)/HZ,side_m,case['closed_fraction'])
                with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                    current=robot.get_dof_positions().numpy()[0]
                    jac=robot.get_jacobian_matrices().numpy()[0,hand_index-1,:,:7]
                    hp,hq=hand.get_world_poses()
                pos=hp.numpy()[0];R=rotation(hq.numpy()[0]);Rgoal=hand_expected[:3,:3]
                angular=.5*sum(np.cross(R[:,i],Rgoal[:,i]) for i in range(3))
                error=np.r_[origin+xyz-pos,angular]
                dq=jac.T @ np.linalg.solve(jac @ jac.T+.0025*np.eye(6),error)
                target[:7]=current[:7]+np.clip(dq,-.1,.1);target[7:]=opening
                robot.set_dof_position_targets(target)
            if surface:surface.frame(step)
            SimulationManager.step(update_fabric=True)
            # Common synchronized endpoint splits physics from readback/capture.
            wp.synchronize_device('cuda:0')
            d0=time.perf_counter()
            if surface and (step==1 or step%240==0):surface.after_frame(step)
            diagnostic_s+=time.perf_counter()-d0
            r0=time.perf_counter();sample(target);readback_s+=time.perf_counter()-r0
            if step%96==0:
                r0=time.perf_counter();capture_frame();capture_s+=time.perf_counter()-r0
            if step%960==0:
                record['completed_steps']=step;save()
        wp.synchronize_device('cuda:0');elapsed=time.perf_counter()-start
        count=cp_trial.counts()
        rounds=args.position_iterations+1
        check('cp_all_steps_active',count[:3]==[total*rounds]*3,counts=count,expected=total*rounds)
        if surface:
            v=surface.snapshot()
            check('supplemental_support_execution',v[0]==v[1]==total*rounds and v[9]>0 and v[4]==v[12]==v[13]==0,stats=v)
            record['supplemental_row_rounds']=v[9]
        else:record['supplemental_row_rounds']=0
        check('cp_no_errors',cp_trial.stop()==0);cp_trial=None
        x=np.asarray(states);t=np.asarray(times);hp=np.asarray(hand_states);q=np.asarray(joints);J=np.asarray(qualities)
        record['task_result'],metrics=score(t,x,hp,q,J,center,table_top,slot_xy,slot_inner,rails)
        record['timing']=dict(rollout_s=elapsed,capture_s=capture_s,readback_s=readback_s,diagnostic_s=diagnostic_s,
            actuation_physics_s=elapsed-capture_s-readback_s-diagnostic_s,scope='matched synchronous rollout; startup and disk export excluded')
        assert len(frames)==101 and len(t)==total+1
        np.savez_compressed(out/'trajectory.npz',time_s=t-t[0],volume_nodes=x,tets=sim_tets,rest_nodes=rest,
            joint_positions=q,joint_targets=np.asarray(targets),hand_poses=hp,finger_poses=np.asarray(finger_states),minimum_J=J,**metrics)
        # Preserve every raw frame, using fast PNG compression, outside timing.
        from PIL import Image
        fdir=out/'frames';fdir.mkdir()
        for i,frame in enumerate(frames):Image.fromarray(frame).save(fdir/f'{i:03d}.png',compress_level=2)
        record.update(status='completed_native_rollout',exit_code=0,video_frames=len(frames),all_timesteps=total)
    except Exception as error:
        record.update(status='failed', exit_code=1, error=str(error), traceback=traceback.format_exc())
    finally:
        from paper_vision_crosscondition_lifecycle import native_cleanup
        try:
            native_cleanup(record,locals(),out)
        except BaseException as cleanup_error:
            record.setdefault('cleanup_failures',[]).append(dict(stage='native_cleanup_receipt',error=repr(cleanup_error)))
            record['exit_code']=1
        try:
            save()
        finally:
            if app is not None:app.close(exit_code=record['exit_code'])
    return record['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
