"""Same native volume method on frictional visual side-push alignment.

No new contact/material kernel or per-backend control law is introduced.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback,faulthandler

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def module_inventory(root):
    roots=[(root/name).resolve() for name in ['deformable_tasks','work','scripts','results']]+[Path(__file__).resolve().parent]
    out={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:continue
        p=Path(path).resolve()
        if p.is_file() and any(p.is_relative_to(base) for base in roots):
            out[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();sys.argv=[sys.argv[0]];root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    case=json.loads(Path(a.case).read_text(encoding='utf-8'));base=root/'results/paper-material-reference-v1/current-surface-coupon-v3/source'
    sys.path.insert(0,str(root/'work/paper-generalization-v1/method/bulk-truth-v1'));sys.path.insert(0,str(root/'work/paper-generalization-v1/method'));sys.path.insert(0,str(root));sys.path.insert(0,str(base));sys.path.insert(0,str(root/'work/paper-generalization-v1/vision/v5'));sys.path.insert(0,str(root/'work/paper-generalization-v1/review'));sys.path.insert(0,str(Path(__file__).resolve().parent))
    from learning_hook import validate_case
    validate_case(case,external=True)
    process_started=time.monotonic()
    learning=None
    rec=dict(status='starting',case=case,pid=os.getpid(),errors=[],node_resets_during_rollout=0)
    external=None;app=runtime=app_utils=adapter=sensor=observer_worker=None;positions=[];velocities=[];times=[];native_times=[];tool_positions=[];tool_quats=[];body_positions=[];commands=[];packets=[];actions=[];mass=None;elements=None;geometry=None;gauge=None
    stack_stream=(out/'python-stacks.log').open('w')
    faulthandler.enable(file=stack_stream,all_threads=True)
    faulthandler.dump_traceback_later(60,repeat=True,file=stack_stream)
    dump(out/'process-receipt.json',rec)
    try:
        import numpy as np
        from side_push_config_v3 import CAMERA_POSITION,CAMERA_QUATERNION_WXYZ,schedule,target_pose,settings
        from side_push_observers import observe_color,observe_depth,ROI
        from observer_client import ObserverWorker
        rec['observer_component_implementation']='Persistent image-only CPU worker, same interpreter/NumPy, reusable float64 geometry buffers and ordered three-term norms, order-preserving flat DFS; original tuple observer independently replayed. All IPC and wait costs included.'

        from side_push_geometry import freeze_front,leading_feature
        from side_push_control import PositionController,InitialPositionCalibration
        sys.path.insert(0,str(root/'work/paper-mechanism-v1/vision'))
        from condition_camera import validate_actual_camera
        condition=case['condition'];camera_spec=condition['camera']
        CAMERA_POSITION=camera_spec['position'];CAMERA_QUATERNION_WXYZ=camera_spec['quaternion_wxyz']
        from side_push_fixture import relocate_fixture
        from deformable_tasks.config import validate
        config=validate(case['config']);assert config['material']['friction']==.4;policy=PositionController();scale=InitialPositionCalibration(.02+case['initial_x_offset_m'])
        assert case['arm'] in ['B','D','dense','frozen_first'] and case['observer'] in ['color','depth','truth']
        spec=importlib.util.spec_from_file_location('common',root/'scripts/isaac_platform_smoke.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        binding=common.runtime_binding(root/'upstream/IsaacSim',os.environ.get('ISAAC_PATH'),sys.executable,'')
        build_receipt=common.verify_build_receipt(root/'results/isaac-initial-runtime-egl-v4/receipt.json',root/'upstream/IsaacSim',binding)
        if not binding['passed']or not build_receipt['passed']:raise RuntimeError('Runtime binding')
        rec['build_receipt']=build_receipt
        def checkpoint(stage):
            dump(out/'startup-phase.json',dict(stage=stage,unix=time.time(),monotonic=time.monotonic()))
            print('STARTUP_PHASE',stage,flush=True)
        checkpoint('SimulationApp')
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,fast_shutdown=True,multi_gpu=False,create_new_stage=False,disable_viewport_updates=True,renderer='RaytracedLighting',width=640,height=480,extra_args=['--/app/useFabricSceneDelegate=0','--/app/extensions/pythonFullLeakCheck=false','--/crashreporter/enabled=false']),experience=binding['experience'])
        from omni.kit.viewport.utility import get_active_viewport
        startup_viewport=get_active_viewport()
        rec['render_lifecycle']=dict(default_stage_creation=False,viewport_updates_enabled=bool(startup_viewport.updates_enabled) if startup_viewport else None,camera_render_product='independent CameraSensor',render_calls_per_feedback=4)
        assert startup_viewport is None or not startup_viewport.updates_enabled
        import warp as wp
        from pxr import UsdGeom,Usd,UsdPhysics,PhysxSchema
        from omni.physx import get_physx_interface
        from isaacsim.core.experimental.utils import stage as stage_utils,app as app_utils
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.simulation_manager import SimulationManager,PhysxScene
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.sensors.experimental.rtx import RtxCamera,CameraSensor
        from deformable_tasks.runtime import TaskRuntime
        from deformable_tasks.material_audit import MaterialAudit
        from side_push_adapter import Adapter

        class ControlledTask(TaskRuntime):
            def target(self,i,t):
                return target_pose(t,policy.correction,case['initial_x_offset_m'])

        checkpoint('initial_app_updates')
        for _ in range(60):app.update()
        stage_utils.create_new_stage();stage=stage_utils.get_current_stage(backend='usd')
        checkpoint('build_task')
        runtime=ControlledTask(config,True).build(stage);runtime.observe_failed_horizon=True;runtime.update_visuals=False
        MaterialAudit(runtime,{'mode':'coupon'},out).prepare()
        PhysxScene('/World/PhysicsScene').set_gravity([0.,0.,-9.81])
        rec['fixture_relocation']=relocate_fixture(stage,runtime,case['initial_x_offset_m'],case['geometry_y_scale'])
        from gauge_geometry import specification
        from gauge_articulation import author,NativeGauge,validate_geometry
        gauge_spec=specification(case['gauge_clearance_m'],case['geometry_y_scale'])
        gauge_spec['center_xy_m'][0]+=case.get('gauge_development_x_bias_m',0.)
        gauge_spec['collisions_enabled']=case.get('gauge_collisions_enabled',True)
        gauge_views,rec['gauge_authoring']=author(stage,gauge_spec,config['solver']['position_iterations'])
        dump(out/'gauge-specification.json',gauge_spec)
        rec['articulation_iteration_authoring']=[]
        for drive in runtime.drives:
            for prim in drive.prims:
                explicit=prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
                raw_prior=prim.GetAttribute('physxArticulation:solverPositionIterationCount').Get()
                api=PhysxSchema.PhysxArticulationAPI.Apply(prim)
                before=[[[int(api.GetSolverPositionIterationCountAttr().Get())]],[[int(api.GetSolverVelocityIterationCountAttr().Get())]]]
                api.CreateSolverPositionIterationCountAttr(config['solver']['position_iterations'])
                after=[[[int(api.GetSolverPositionIterationCountAttr().Get())]],[[int(api.GetSolverVelocityIterationCountAttr().Get())]]]
                assert after[0][0][0]==config['solver']['position_iterations'] and after[1]==before[1]
                rec['articulation_iteration_authoring'].append(dict(before=before,after=after,raw_prior_position_attr=raw_prior,schema_previously_explicit=bool(explicit),prim_path=str(prim.GetPath()),scope='Resolved USD PhysxArticulation schema, not native SDK readback; actual dispatch rounds checked each step'))
        assert len(rec['articulation_iteration_authoring'])==1
        camera=RtxCamera('/World/FeedbackCamera',translations=CAMERA_POSITION,orientations=CAMERA_QUATERNION_WXYZ)
        cam=UsdGeom.Camera(stage.GetPrimAtPath('/World/FeedbackCamera'));cam.GetFocalLengthAttr().Set(24.);cam.GetHorizontalApertureAttr().Set(20.955);cam.GetVerticalApertureAttr().Set(20.955*480/640);camera.camera.set_clipping_ranges(.01,100.)
        checkpoint('create_camera_sensor')
        sensor=CameraSensor(camera,resolution=(480,640),annotators=['rgb','distance_to_image_plane'])
        actual=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(cam.GetPrim()),float)
        lens_readback=dict(width=640,height=480,focal_length_mm=float(cam.GetFocalLengthAttr().Get()),horizontal_aperture_mm=float(cam.GetHorizontalApertureAttr().Get()),vertical_aperture_mm=float(cam.GetVerticalApertureAttr().Get()),horizontal_offset_mm=float(cam.GetHorizontalApertureOffsetAttr().Get()),vertical_offset_mm=float(cam.GetVerticalApertureOffsetAttr().Get()))
        CALIBRATION=validate_actual_camera(camera_spec,actual,lens_readback);expected=CALIBRATION.matrix()
        dump(out/'condition.json',condition)
        attachments=[str(v.GetPath())for v in stage.Traverse()if 'Attachment' in v.GetTypeName()];assert not attachments
        rec['attachments']=attachments;observer_settings=settings();observer_settings['calibration']=vars(CALIBRATION);observer_settings['condition']=condition;observer_settings['camera_position']=CAMERA_POSITION;observer_settings['camera_quaternion_wxyz']=CAMERA_QUATERNION_WXYZ;dump(out/'observer-settings.json',observer_settings)
        stage.Export(str(out/'initial.usda'))
        tool_prim=stage.GetPrimAtPath('/World/Press/Tool/Shape');bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_],useExtentsHint=False).ComputeWorldBound(tool_prim).ComputeAlignedRange()
        tool_min=np.asarray(bounds.GetMin());tool_max=np.asarray(bounds.GetMax());tool_half=(tool_max-tool_min)/2;shape_center=(tool_max+tool_min)/2
        assert np.allclose(tool_half,[.04,.04,.01],rtol=0,atol=2e-7)
        checkpoint('scene_app_updates')
        for _ in range(20):app.update()
        checkpoint('play');app_utils.play()
        checkpoint('app_update');app.update()
        checkpoint('first_render');RenderingManager.render()
        checkpoint('first_physics_step');SimulationManager.step(update_fabric=False)
        checkpoint('node_reset')
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            b=runtime.bodies[0];b['view'].set_nodal_positions(np.asarray([b['rest']],dtype=np.float32));b['view'].set_nodal_velocities(np.zeros((1,len(b['rest']),3),dtype=np.float32));runtime.initialize_fixture_drives()
        wp.synchronize_device('cuda:0');runtime.after_step(advance=False)
        elements=np.asarray(b['sim_elements']);rest=np.asarray(b['rest']);feature=freeze_front(rest,elements);dump(out/'oracle-feature.json',feature)
        sys.path.insert(0,str(root/'work/native-readback'));import physx_deformable_readback as rb
        sys.path.insert(0,str(root/'work/external-support-v3-build'));import physx_support_geometry_v3 as sdk
        interface=get_physx_interface();handle=rb.simulation_buffer(interface,b['path']);raw=wp.array(ptr=handle['device_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0').numpy();mass=1/raw[:,3].astype(float)
        expected_extents=np.asarray([.04,.04*case['geometry_y_scale'],.04]);actual_extents=np.ptp(rest,axis=0)
        assert np.allclose(actual_extents,expected_extents,rtol=0,atol=2e-7),(actual_extents,expected_extents)
        assert b['gpu_rest_difference_m']<2e-7,b['gpu_rest_difference_m']
        expected_mass=float(np.prod(expected_extents)*config['material']['density'])
        assert np.isclose(mass.sum(),expected_mass,rtol=2e-5,atol=1e-8),(mass.sum(),expected_mass)
        rec['actual_geometry_mass']=dict(extents_m=actual_extents.tolist(),expected_extents_m=expected_extents.tolist(),gpu_rest_difference_m=b['gpu_rest_difference_m'],mass_kg=float(mass.sum()),expected_mass_kg=expected_mass)
        geo=sdk.read(interface,b['path']);assert np.allclose(geo['gravity_m_s2'],[0.,0.,-9.81],atol=1e-6,rtol=0)
        assert geo['solver_is_tgs']and geo['external_forces_every_iteration_tgs']and (int(geo['scene_flags'])&(1<<16))
        bench=[q for q in geo['rigid_shapes']if q['simulation_shape']and q['is_static']and q['is_box']and np.allclose(q['shape_world_pose']['position'],[0,0,.085],rtol=0,atol=2e-7)]
        assert len(bench)==1;bench=bench[0];assert np.allclose(bench['half_extents_m'],[.08,.08,.035],rtol=0,atol=2e-7)
        q=bench['shape_world_pose']['quaternion_xyzw'];benchq=[q[3],*q[:3]]
        shape_offset=shape_center-runtime.movers[0][1];assert np.max(abs(shape_offset))<2e-7
        geometry=dict(oracle_geometry=dict(tool_half_extents_m=tool_half.tolist(),bench_position_m=bench['shape_world_pose']['position'],bench_quaternion_wxyz=benchq,bench_half_extents_m=bench['half_extents_m']),
            geometry_sources=dict(tool='USD collider world bounds and Tensor Tool pose; validated zero local offset',bench='actual PhysX SDK shape pose and box half-extents'),sdk_initial=geo,tool_initial_world_center_m=shape_center.tolist(),tool_local_offset_m=shape_offset.tolist(),
            camera_usd_row_matrix=actual.tolist(),camera_lens_readback=lens_readback,camera_contract=expected.tolist(),condition=condition,adapter_columns=['step']+['cp'+str(i)for i in range(7)]+['surface'+str(i)for i in range(16)])
        sys.path.insert(0,str(root/'work/paper-method-generalization-v1/reference/friction-sdk-v1'))
        import physx_friction_readback_v1 as friction_sdk
        from side_push_friction_readback import validate as validate_friction
        friction=validate_friction(friction_sdk.read(interface,b['path']),shape_center)
        dump(out/'friction-native.json',friction);rec['native_friction_readback']=friction
        dump(out/'geometry.json',geometry)
        rec['gauge_native_geometry']=validate_geometry(friction_sdk.read(interface,b['path']),gauge_spec)
        gauge=NativeGauge(gauge_spec,config['solver']['hz'],gauge_views)
        from newton_client import Client,publish,GaugeRecorder
        fixture=dict(rest=rest.tolist(),elements=elements.tolist(),mass=mass.tolist(),tool_initial=runtime.movers[0][1].tolist(),gauge=gauge_spec,geometry=geometry['oracle_geometry'],material=config['material'],hz=config['solver']['hz'])
        external=Client(root,out,fixture,case['external_settings'])
        rec['physics_backend']='Newton VBD 1.5.2; native scene is a frozen renderer template'
        rec['newton_worker_pid']=external.ready['pid'];rec['external_settings']=case['external_settings']
        gauge=GaugeRecorder(gauge_spec,config['solver']['hz'])
        rec['display_max_node_error_m']=publish(runtime,gauge_views,external.initial)
        origin=SimulationManager.get_simulation_time();hz=config['solver']['hz'];n=round(config['duration_s']*hz);assert hz in [960,1920] and hz%10==0
        rec.update(native_clock_origin_s=origin,requested_steps=n,actual_native_settings=handle['settings'],initial_node_error_m=float(abs(b['last']-rest).max()))
        assert rec['initial_node_error_m']<1e-6
        warm=SimulationManager.get_simulation_time()
        checkpoint('camera_warmup')
        for _ in range(30):RenderingManager.render()
        checkpoint('feedback_ready')
        faulthandler.cancel_dump_traceback_later()
        if case['observer'] in ['depth','color'] and case['learning']['mode']=='demo':
            observer_worker=ObserverWorker(source_files={
                'observers':root/'work/paper-generalization-v1/vision/v5/observers.py',
                'side_push_config':Path(__file__).with_name('side_push_config.py'),
                'side_push_observers':Path(__file__).with_name('side_push_observers.py'),
                'component_flat':Path(__file__).with_name('component_flat.py'),
            },numpy_module=np,first_sequence=0,timeout_s=20.,startup_timeout_s=30.)
            rec['observer_worker_handshake']=observer_worker.handshake
        assert SimulationManager.get_simulation_time()==warm
        (out/'rgbd').mkdir()
        def capture_batch(data):
            for i,step in enumerate(data['step']):
                positions.append(data['positions'][i]);velocities.append(data['velocities'][i]);times.append(float(step)/hz);native_times.append(float(step)/hz)
                pos=data['tool'][i,:3];q=data['tool'][i,3:];tool_positions.append(pos);body_positions.append(pos);tool_quats.append(np.r_[q[3],q[:3]]);commands.append(data['command'][i])
            gauge.append(data)
            runtime.steps=int(data['step'][-1])
        from feedback_profile import FeedbackProfile
        feedback_profile=FeedbackProfile()
        from learning_hook import EpisodeHook
        learning=EpisodeHook(case,out)
        rec['learning_contract']=learning.contract
        def feedback():
            from newton_client import publish as common_publish
            display_data=dict(positions=np.asarray([positions[-1]],np.float32),tool=np.asarray([np.r_[tool_positions[-1],tool_quats[-1][1:],tool_quats[-1][0]]]),gauge=np.asarray([gauge.actual[-1]]))
            rec['display_max_node_error_m']=max(rec.get('display_max_node_error_m',0.),common_publish(runtime,gauge_views,display_data))
            feedback_profile.begin(runtime.steps)
            before=SimulationManager.get_simulation_time()
            for render_index in range(4):
                with feedback_profile.stage('render_'+str(render_index)): RenderingManager.render()
            with feedback_profile.stage('annotator_fetch'):
                rgb,ri=sensor.get_data('rgb');depth,di=sensor.get_data('distance_to_image_plane')
            with feedback_profile.stage('array_readback_copy'):
                rgb=rgb.numpy().copy();depth=depth.numpy().reshape(480,640).copy()
            after=SimulationManager.get_simulation_time();assert abs(after-before)<1e-10
            t=runtime.steps/hz;seq=len(packets);kwargs=dict(sample_time_s=t,sequence=seq)
            s=schedule(t,case['initial_x_offset_m'])
            with feedback_profile.stage('learning_observation'):
                learning.observe(rgb,depth,t,tool_positions[-1],gauge.actual[-1][2],policy.correction,s)
            if learning.mode=='demo':
                with feedback_profile.stage('observer'):
                    if case['observer'] in ['color','depth']:
                        ob=observer_worker.observe(mode=case['observer'],depth=depth,calibration=CALIBRATION,roi=ROI,**kwargs,**({'rgb':rgb} if case['observer']=='color' else {}))
                    else:raise ValueError('Learning demonstrations require image-based expert')
                with feedback_profile.stage('calibration_controller'):
                    raw_ob=ob;ob=scale.apply(raw_ob)
                    action=policy.update(t,s['target_position_m'],ob,active=s['feedback_active'])
            else:
                with feedback_profile.stage('learning_inference'):
                    raw_ob=ob=dict(valid=True,sequence=seq,sample_time_s=t,input='RGB-D+proprio only')
                    action=learning.act(policy,t,s)
            with feedback_profile.stage('learning_record'):
                learning.record(action)
            with feedback_profile.stage('packet_metadata'):
                packet=dict(step=runtime.steps,time_s=t,raw_observation=raw_ob,observation=ob,calibration_state=scale.state(),rgb_metadata=common.annotator_frame_info(ri),depth_metadata=common.annotator_frame_info(di),physics_time_before=before,physics_time_after=after)
            feedback_profile.archive(out/'rgbd'/f'frame-{runtime.steps:05d}.npz',np,rgb=rgb,depth=depth)
            packets.append(packet);actions.append(action)
            feedback_profile.feedback_json(out/'feedback.json',dict(packets=packets,actions=actions))
            feedback_profile.end()
        capture_batch(external.initial);feedback();wp.synchronize_device('cuda:0');rec['loop_started_unix']=time.time();started=time.monotonic();rec['loop_started_monotonic']=started;rec['performance_mode']=dict(full_round_trace_enabled=False);compute_seconds=[];feedback_seconds=[];rec['status']='running';dump(out/'process-receipt.json',rec)
        for stop_step in range(hz//10,n+1,hz//10):
            tick=time.monotonic();data=external.advance(policy.correction,case['initial_x_offset_m'],hz//10);compute_seconds.append(time.monotonic()-tick)
            capture_batch(data)
            assert runtime.steps==stop_step
            assert SimulationManager.get_simulation_time()==warm, 'Native template clock advanced'
            tick=time.monotonic();feedback();feedback_seconds.append(time.monotonic()-tick)
            rec.update(completed_steps=runtime.steps,elapsed_s=time.monotonic()-started);dump(out/'process-receipt.json',rec);print('EXTERNAL_STEP',stop_step,flush=True)
        wp.synchronize_device('cuda:0');rec['loop_ended_monotonic']=time.monotonic();rec['loop_ended_unix']=time.time();rec['elapsed_s']=rec['loop_ended_monotonic']-started
        rec['timing']=dict(loop_elapsed_s=rec['elapsed_s'],physics_instrumented_s=float(sum(compute_seconds)),render_observer_io_s=float(sum(feedback_seconds)),physics_step_seconds=compute_seconds,feedback_seconds=feedback_seconds,scope='Batch wall includes VBD drive/collision/solve/GPU sync, worker state capture and IPC. Feedback includes identical four RTX renders, observer, controller and writes. Solver iterations are not equal cost across engines. Native clock is frozen; native_time_s stores the external discrete clock for common scoring.')
        rec['observer_component_calls']=sum(event['component_calls'] for event in observer_worker.events) if observer_worker else 0
        rec['learning_export']=learning.finish(rec['timing'],process_started,rec['loop_started_monotonic'])
        dump(out/'timing.json',rec['timing'])
        dump(out/'feedback-profile.json',feedback_profile.report())
        rec['gauge_final_sdk']=friction_sdk.read(interface,b['path']);rec['gauge_motion']=gauge.save(out/'gauge-trajectory.npz');rec['adapter']=dict(external_backend=True,no_native_contact_adapter_executed=True);external.close();external=None;rec.update(status='completed_pending_exit',completed_steps=runtime.steps,physics_time_s=native_times[-1])
    except BaseException as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        faulthandler.cancel_dump_traceback_later()
        if external:
            try:external.close()
            except BaseException as e:rec['errors'].append(dict(stage='external_close',error=repr(e)))
        if learning:
            try:learning.close()
            except BaseException as e:rec['errors'].append(dict(stage='learning_close',error=repr(e)))
        if observer_worker:
            try:
                observer_worker.close()
                rec['observer_worker_exit_code']=observer_worker.process.returncode
            except BaseException as e:rec['errors'].append(dict(stage='observer_worker_close',error=repr(e)))
            dump(out/'observer-worker.json',dict(handshake=observer_worker.handshake,events=observer_worker.events,slow_samples=observer_worker.slow_samples,exit_code=observer_worker.process.returncode,stderr_tail=bytes(observer_worker.stderr_tail).decode('utf-8',errors='replace')))
        try:
            imports=module_inventory(root)
            if observer_worker:imports.update(observer_worker.handshake['actual_project_imports'])
            dump(out/'actual-project-imports.json',imports)
        except BaseException as e:rec['errors'].append(dict(stage='module_inventory',error=repr(e)))
        if gauge:
            try:rec['gauge_motion']=gauge.save(out/'gauge-trajectory.npz')
            except BaseException as e:rec['errors'].append(dict(stage='gauge_partial_raw',error=repr(e)))
        if positions:
            try:
                np.savez_compressed(out/'trajectory.npz',time_s=times,native_time_s=native_times,step_index=np.arange(len(times)),positions=np.asarray(positions,dtype=np.float32),velocities=np.asarray(velocities,dtype=np.float32),rest=rest,elements=elements,mass=mass,tool_positions=tool_positions,tool_quaternions=tool_quats,tool_body_positions=body_positions,commanded_tool_positions=commands,adapter_counts=np.asarray(adapter.records if adapter else [],dtype=np.uint64).reshape(-1,24))
            except BaseException as e:rec['errors'].append(dict(stage='partial_raw',error=repr(e)))
        if adapter:
            try:adapter.stop();rec['adapter_cleanup']=adapter.evidence
            except BaseException as e:rec['errors'].append(dict(stage='adapter_stop',error=repr(e)))
        for name,fn in [('timeline',lambda:app_utils.stop()if app_utils else None),('runtime',lambda:runtime.release()if runtime else None)]:
            try:fn()
            except BaseException as e:rec['errors'].append(dict(stage=name,error=repr(e)))
        rec['exit_code']=0 if rec['status']=='completed_pending_exit'and not rec['errors']else 1
        rec['exit_code_scope']='Semantic completion persisted before SimulationApp.close; raw OS exit may be0 under Kit fast_shutdown'
        dump(out/'process-receipt.json',rec)
        if app:
            try:app.close()
            except BaseException as e:rec['errors'].append(dict(stage='app_close',error=repr(e)))
        rec['exit_code']=0 if rec['status']=='completed_pending_exit'and not rec['errors']else 1;dump(out/'process-receipt.json',rec)
    return rec['exit_code']

if __name__=='__main__':raise SystemExit(main())
