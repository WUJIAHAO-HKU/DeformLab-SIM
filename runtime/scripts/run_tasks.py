"""Version-pinned GPU regression, backend evidence and synchronized timing."""
import argparse,asyncio,importlib.util,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deformable_tasks.config import default_config,load_config,save_config,regression_configs,validate,TASKS
from deformable_tasks.backends import BACKENDS,CP_MODES,plan,CpAdapter,observe_volumes

def gpu_processes():
    try:return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=8).strip().splitlines()
    except Exception as exc:return ['query failed: '+repr(exc)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS,default='cloth_grip');p.add_argument('--config')
    p.add_argument('--suite',action='store_true');p.add_argument('--cases');p.add_argument('--output-dir',required=True)
    p.add_argument('--render',action='store_true');p.add_argument('--duration',type=float)
    p.add_argument('--profile',choices=('studio','pathtraced'));p.add_argument('--backend',choices=BACKENDS,default='native')
    p.add_argument('--iterations',type=int);p.add_argument('--timing-only',action='store_true')
    p.add_argument('--base-hz',type=int)
    p.add_argument('--volume-floor',type=float)
    p.add_argument('--volume-sweeps',type=int)
    p.add_argument('--material-damping-integrator',choices=('backward-euler','exponential'))
    p.add_argument('--audit-plan',help='Explicit material fixtures and strict task configurations')
    p.add_argument('--observe-failed-horizon',action='store_true',help='Volume benchmarks only: retain inversion failure but keep driving the finite trajectory to its requested horizon')
    args=p.parse_args();sys.argv=[sys.argv[0]]
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Outputs must be in project results')
    audits={}
    if args.audit_plan:
        from deformable_tasks.material_audit import MaterialAudit
        raw=Path(args.audit_plan).read_text();audit_plan=json.loads(raw)
        if set(audit_plan)!={'cases'} or not audit_plan['cases']:raise ValueError('Invalid audit plan')
        configs={}
        for name,entry in audit_plan['cases'].items():
            if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',name) or set(entry)!={'config','audit'}:raise ValueError('Invalid audit case')
            configs[name]=validate(entry['config']);audits[name]=entry['audit']
            if configs[name]['task']!='soft_squeeze' or audits[name].get('mode') not in MaterialAudit.MODES:raise ValueError('Invalid audit fixture')
            from deformable_tasks.kernel_stage_audit import validate_steps
            validate_steps(audits[name],args.backend,round(configs[name]['duration_s']*configs[name]['solver']['hz']))
        if args.backend not in ('native','volume-project','volume-project-control','cp-control',*CP_MODES):raise ValueError('Unsupported audit backend')
        (out/'audit-plan.json').write_text(raw)
    elif args.suite:configs=regression_configs()
    else:
        c=load_config(args.config) if args.config else default_config(args.task);configs={c['task']:c}
    if args.cases:
        wanted=args.cases.split(',')
        if set(wanted)-set(configs):raise ValueError('Unknown cases')
        configs={k:configs[k] for k in wanted}
    # Each case still has exactly one volume body. The version-checked CP
    # adapter now releases and resets its host state between successive scenes.
    if args.observe_failed_horizon and (args.backend not in ('native','volume-project','volume-project-control') or any(c['task'].startswith('cloth') for c in configs.values())):
        raise ValueError('Failed-horizon observation is restricted to native/volume-project volume benchmarks')
    from deformable_tasks.trajectory_recording import requested as recording_requested,stride as recording_stride,evidence as recording_evidence
    full_rate_recording=recording_requested(next(iter(configs.values()))['task'],args.backend,rendered=args.render,suite=args.suite,timing_only=args.timing_only)
    plans={}
    for name,c in configs.items():
        if args.duration is not None:c['duration_s']=args.duration
        if args.profile:c['rendering']['profile']=args.profile
        if args.base_hz is not None:c['solver']['hz']=args.base_hz
        plans[name]=plan(c,args.backend,args.iterations,args.volume_floor,args.volume_sweeps,args.material_damping_integrator)
    spec=importlib.util.spec_from_file_location('common',ROOT/'scripts/isaac_platform_smoke.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    record=dict(status='starting',cases={},errors=[],pid=os.getpid(),backend=args.backend,timing_only=args.timing_only)
    def save():(out/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    save();app=None;adapter=None;audit=None;runtime=None;app_utils=None;profile=None;failure_checkpoint=None
    try:
        binding=common.runtime_binding(ROOT/'upstream/IsaacSim',os.environ.get('ISAAC_PATH'),sys.executable,'')
        receipt=common.verify_build_receipt(ROOT/'results/isaac-initial-runtime-egl-v4/receipt.json',ROOT/'upstream/IsaacSim',binding)
        assert binding['passed'] and receipt['passed'];record['receipt']=receipt
        from isaacsim import SimulationApp
        first=next(iter(plans.values()))[0]
        app=SimulationApp(dict(headless=not bool(os.environ.get('DISPLAY')),fast_shutdown=True,multi_gpu=False,
            renderer='PathTracing' if first['rendering']['profile']=='pathtraced' else 'RaytracedLighting',
            width=first['rendering']['width'],height=first['rendering']['height'],
            extra_args=['--/app/useFabricSceneDelegate=0','--/app/extensions/pythonFullLeakCheck=false','--/crashreporter/enabled=false']),experience=binding['experience'])
        import numpy as np
        import warp as wp
        from deformable_tasks.runtime import TaskRuntime
        from deformable_tasks.volume_admissibility import VolumeAdmissibility
        from isaacsim.core.experimental.utils import stage as stage_utils,app as app_utils
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.rendering_manager import RenderingManager
        from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
        import omni.timeline
        for _ in range(60):app.update()
        runtime=None;viewport=get_active_viewport()
        for name,(c,backend) in plans.items():
            failure_checkpoint=None;step=0
            record.update(status='running',active_case=name);save();print('CASE '+name,flush=True)
            dest=out/name;dest.mkdir();save_config(dest/'task.json',backend['requested_config']);save_config(dest/'effective-task.json',c)
            (dest/'backend.json').write_text(json.dumps(backend,indent=2)+'\n')
            if runtime:runtime.release();runtime=None
            app_utils.stop()
            for _ in range(4):app.update()
            stage_utils.create_new_stage();stage=stage_utils.get_current_stage(backend='usd')
            runtime=TaskRuntime(c,backend['external_every_iteration'],backend.get('contact_schedule')).build(stage);runtime.update_visuals=args.render
            runtime.observe_failed_horizon=args.observe_failed_horizon
            audit=MaterialAudit(runtime,audits[name],dest) if name in audits else None
            if audit:audit.prepare()
            stage.Export(str(dest/'initial.usda'))
            for _ in range(20):app.update()
            viewport.camera_path='/World/Camera'
            if args.render:viewport.set_texture_resolution((c['rendering']['width'],c['rendering']['height']))
            app_utils.play();app.update();RenderingManager.render();SimulationManager.step(update_fabric=False)
            # Native initialization advances hidden steps. Reset ONCE before t=0.
            # No prescribed nodal motion or resets occur during the measured rollout.
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                for b in runtime.bodies:
                    b['view'].set_nodal_positions(np.asarray([b['rest']],dtype=np.float32))
                    b['view'].set_nodal_velocities(np.zeros((1,len(b['rest']),3),dtype=np.float32))
                runtime.initialize_fixture_drives()
            wp.synchronize_device('cuda:0')
            adapter=CpAdapter(ROOT,runtime,args.backend)
            runtime.after_step(advance=False)
            (dest/'native-volumes.json').write_text(json.dumps(observe_volumes(ROOT,runtime),indent=2)+'\n')
            admissibility=VolumeAdmissibility(ROOT,runtime,args.backend,backend.get('volume_policy'))
            from deformable_tasks.implicit_material_damping import ImplicitMaterialDamping
            material_damping=ImplicitMaterialDamping(ROOT,runtime,backend.get('material_damping_policy'))
            if audit:audit.bind(ROOT)
            from deformable_tasks.kernel_stage_audit import KernelStageAudit
            stages=KernelStageAudit(runtime,adapter,audits.get(name,{}),dest)
            initial_error=max(float(np.max(np.abs(b['last']-b['rest']))) for b in runtime.bodies)
            if initial_error>1e-6:raise RuntimeError('Initial state reset mismatch')
            origin=SimulationManager.get_simulation_time()
            initial=dict(reset_once_before_recording=True,node_error_m=initial_error,native_clock_origin_s=origin,
                         resets_during_trajectory=0,node_counts=[len(b['rest']) for b in runtime.bodies])
            (dest/'initial-state.json').write_text(json.dumps(initial,indent=2)+'\n')
            viewport.updates_enabled=args.render
            nsteps=round(c['duration_s']*c['solver']['hz']);stride=recording_stride(c['solver']['hz'],full_rate_recording)
            nodes=[[b['last'].copy() for b in runtime.bodies]];times=[0.];native_times=[0.];recorded_steps=[0]
            from deformable_tasks.failure_checkpoint import FailureCheckpoint
            failure_checkpoint=FailureCheckpoint(runtime.bodies)
            failure_checkpoint.capture(0,0.,0.,nodes[0])
            timings=[];diagnostics=[];first_failure=None
            activity=[dict(step=0,processes=gpu_processes())];start=time.monotonic()
            snapshots={int(x*nsteps) for x in (.25,.5,.75,1.)}
            from deformable_tasks.kernel_profile import KernelProfile
            profile=KernelProfile(ROOT,dest,c['task'],args.backend)
            from deformable_tasks.cloth_geometry_checkpoint import requested as geometry_requested,GeometryCheckpoint
            geometry_checkpoint=GeometryCheckpoint(runtime,profile,dest) if geometry_requested(c['task'],args.backend) else None
            for step in range(1,nsteps+1):
                wp.synchronize_device('cuda:0');begin=time.perf_counter()
                runtime.before_step()
                if audit:audit.before_step()
                stages.before(step)
                SimulationManager.step(update_fabric=False);admissibility.apply();wp.synchronize_device('cuda:0')
                stages.after(step)
                if audit:audit.capture_motion(step,'native')
                material_damping.apply()
                if audit:audit.capture_motion(step,'damped')
                timings.append(time.perf_counter()-begin)
                begin=time.perf_counter();runtime.after_step();diagnostics.append(time.perf_counter()-begin)
                if geometry_checkpoint:geometry_checkpoint.capture(step)
                if audit:audit.after_step(admissibility)
                if step==1:adapter.check_active()
                observed=SimulationManager.get_simulation_time()-origin
                failure_checkpoint.capture(step,runtime.latest['time_s'],observed,[b['last'] for b in runtime.bodies])
                if abs(observed-step/c['solver']['hz'])>1e-6:raise RuntimeError('Physics clock drift: '+str((step,observed)))
                if step%stride==0 or step==nsteps or (runtime.error and not args.observe_failed_horizon):
                    nodes.append([b['last'].copy() for b in runtime.bodies]);times.append(runtime.latest['time_s']);native_times.append(observed);recorded_steps.append(step)
                if first_failure is None:
                    failed=[k for k,v in runtime.report()['checks'].items() if not v]
                    if failed:
                        first_failure=dict(step=step,time_s=observed,checks=failed)
                        np.savez_compressed(dest/'first-failure.npz',time_s=observed,
                            **{f'nodes_{i}':b['last'] for i,b in enumerate(runtime.bodies)})
                        (dest/'first-failure.json').write_text(json.dumps(dict(summary=first_failure,measurement=runtime.latest),indent=2)+'\n')
                if step%c['solver']['hz']==0:activity.append(dict(step=step,processes=gpu_processes()))
                if args.render and (step in snapshots or runtime.error):
                    before=SimulationManager.get_simulation_time()
                    import omni.physx
                    omni.physx.get_physx_interface().update_transformations(False,True)
                    omni.timeline.get_timeline_interface().pause()
                    for _ in range(16):RenderingManager.render()
                    capture=capture_viewport_to_file(viewport,str(dest/f'frame-{step:05d}.png'))
                    future=asyncio.ensure_future(capture.wait_for_result(completion_frames=8));deadline=time.monotonic()+90
                    while time.monotonic()<deadline:
                        app.update();time.sleep(.02)
                        if future.done():break
                    if not future.done():raise RuntimeError('Viewport capture timeout')
                    future.result()
                    assert abs(SimulationManager.get_simulation_time()-before)<1e-8
                    omni.timeline.get_timeline_interface().play()
                if runtime.error and not args.observe_failed_horizon:break
            wall=time.monotonic()-start;activity.append(dict(step=runtime.steps,processes=gpu_processes()))
            geometry_evidence=geometry_checkpoint.save() if geometry_checkpoint else None
            profile_evidence=profile.finish(runtime.steps);profile=None
            stage_evidence=stages.finish()
            evidence=adapter.stop();adapter=None
            result=runtime.report();result.update(wall_time_s=wall,requested_steps=nsteps,
                observe_failed_horizon=args.observe_failed_horizon,
                completed_duration=runtime.steps==nsteps,observed_duration_s=SimulationManager.get_simulation_time()-origin,
                backend=backend,backend_evidence=evidence,initial_state=initial,first_failure=first_failure,timing_only=args.timing_only)
            if geometry_evidence:result['geometry_checkpoint']=geometry_evidence
            if profile_evidence:
                result['kernel_profile']=profile_evidence
                result['checks']['kernel_profile_valid']=profile_evidence['passed']
            if stage_evidence:result['kernel_stage_audit']=stage_evidence
            damping_evidence=material_damping.report()
            if damping_evidence:
                result['implicit_material_damping']=damping_evidence
                result['checks']['implicit_material_damping_accepted']=damping_evidence['accepted']
            if backend['kernel_intervention']:result['checks']['kernel_intervention_accepted']=evidence.get('accepted',False)
            projection=admissibility.report()
            if audit:result['material_audit']=audit.finish();audit=None
            if projection:
                result['volume_admissibility']=projection;result['checks'].update(projection['checks'])
            if full_rate_recording:
                result['trajectory_recording']=recording_evidence(full_rate_recording,stride,recorded_steps,runtime.steps,nsteps)
                result['checks']['full_rate_recording_complete']=result['trajectory_recording']['complete_full_rate_coverage']
            result['status']='passed' if result['completed_duration'] and all(result['checks'].values()) else 'failed'
            result['timing']=dict(control_physics_sync_s=sum(timings),diagnostics_s=sum(diagnostics),
                compute_s_per_sim_second=sum(timings)/(runtime.steps/c['solver']['hz']),
                boundary='CPU control + SimulationManager.step + CUDA device synchronization; excludes initialization, rendering, diagnostics and files',
                end_to_end_rollout_s=wall,step_count=len(timings),gpu_activity=activity)
            arrays=dict(time_s=times,native_time_s=native_times,control_physics_sync_s=timings,diagnostics_s=diagnostics)
            if full_rate_recording:arrays['step_index']=recorded_steps
            for i,b in enumerate(runtime.bodies):
                arrays[f'nodes_{i}']=np.asarray([x[i] for x in nodes]);arrays[f'rest_{i}']=b['rest'];arrays[f'faces_{i}']=b['faces'];arrays[f'elements_{i}']=b.get('sim_elements',b['elements'])
            np.savez_compressed(dest/'trajectory.npz',**arrays)
            (dest/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
            record['cases'][name]={k:v for k,v in result.items() if k!='samples'};save()
            print(json.dumps(dict(case=name,status=result['status'],steps=runtime.steps,timing=result['timing']['control_physics_sync_s'],checks=result['checks'])),flush=True)
            viewport.updates_enabled=True
            for _ in range(2):RenderingManager.render()
        app_utils.stop();runtime.release()
        record.update(status='completed_pending_process_exit',exit_code=0);save();app.close()
    except BaseException as exc:
        if failure_checkpoint is not None:
            try:
                record['exception_checkpoint']=failure_checkpoint.save(dest,error=repr(exc),attempted_step=step,
                    nodes=nodes,times=times,native_times=native_times,timings=timings,diagnostics=diagnostics)
            except Exception as checkpoint_error:
                record['errors'].append(dict(stage='exception_checkpoint',error=repr(checkpoint_error)))
        if profile:
            try:profile.finish(runtime.steps if runtime else 0)
            except Exception:pass
        if audit:
            try:audit.finish()
            except Exception:pass
        record.update(status='failed',exit_code=1);record['errors'].append(dict(error=repr(exc),traceback=traceback.format_exc()));save()
        if adapter:
            try:adapter.stop()
            except Exception:pass
        if app_utils:
            try:app_utils.stop()
            except Exception as cleanup_error:record['errors'].append(dict(cleanup='timeline_stop',error=repr(cleanup_error)))
        if runtime:
            try:runtime.release()
            except Exception as cleanup_error:record['errors'].append(dict(cleanup='runtime_release',error=repr(cleanup_error)))
        save()
        if app:app.close(exit_code=1)
        raise

if __name__=='__main__':main()
