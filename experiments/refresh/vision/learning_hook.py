"""Causal RGB-D episode capture and external-process residual-policy injection.

Task: 1D visual residual control of a driven side pusher, followed by withdrawal
and the original descending gauge endpoint. This is not learned arm grasping.
"""
from pathlib import Path
import hashlib,json,os,select,subprocess,time
import numpy as np

PROPRIO=['time_s','tool_x_m','tool_y_m','tool_z_m','gauge_z_m','previous_correction_m','scheduled_target_x_m','feedback_active']
CONTRACT=dict(schema_version=1,task='finite_support_side_push_precision_gauge',observation_shape=[96,96],
    proprio=PROPRIO,action=['absolute_pusher_x_residual_m'],action_bounds=[[-.004,.004]],
    max_action_change_m=.0005,feedback_period_s=.1,feedback_active_window_s=[.4,2.4],
    observation_order='physical state -> render -> RGB-D/proprio -> action -> next physical step',
    task_scope='1D learned residual on a fixed drive schedule; complete withdrawal and gauge endpoint',
    common_evaluation_backend='Newton VBD; independent implementation, not physical ground truth',
    depth_invalid_value_m=0.,resize='area RGB, nearest depth; aspect ratio preserved by letterboxing')

def validate_case(case,external):
    spec=case['learning'];assert spec['mode'] in ['demo','eval']
    for key in ['episode_id','split_group_id','split','source_backend']:assert spec.get(key) is not None,key
    assert spec['source_backend'] in ['B','D','dense']
    assert spec['split'] in ['train','val','test','preflight']
    assert not case.get('oracle_clearance_gate') and not case.get('approach_reserve_m',0.)
    assert not case.get('gate_mode') and case.get('gauge_collisions_enabled',True)
    assert abs(case['config']['duration_s']-4.4)<1e-8
    assert case['observer'] in ['depth','color']
    if external:
        assert case['backend']=='VBD'
        if spec['mode']=='demo':
            assert spec['split']=='preflight', 'Newton expert is reference-only, never B/D/dense training data'
        else:
            assert spec['split'] in ['test','preflight']
            assert spec.get('policy_command')
        assert case.get('external_settings'), 'Freeze common evaluation settings explicitly'
    else:
        assert spec['mode']=='demo' and case['arm']==spec['source_backend']
        assert spec['split'] in ['train','val','preflight'] or (spec['split']=='test' and case.get('independent_contact_validation') is True)

def image_observation(rgb,depth):
    from PIL import Image
    rgb=np.asarray(rgb);depth=np.asarray(depth)
    if rgb.ndim==4 and len(rgb)==1:rgb=rgb[0]
    assert rgb.ndim==3 and rgb.shape[-1] in [3,4] and depth.ndim==2
    assert rgb.shape[:2]==depth.shape
    h,w=depth.shape;scale=96/max(h,w);oh,ow=round(h*scale),round(w*scale)
    canvas=np.zeros((96,96,3),np.uint8);dcanvas=np.zeros((96,96),np.float32)
    y=(96-oh)//2;x=(96-ow)//2
    canvas[y:y+oh,x:x+ow]=np.asarray(Image.fromarray(rgb[:,:,:3].astype(np.uint8)).resize((ow,oh),Image.Resampling.BOX))
    clean=np.where(np.isfinite(depth)&(depth>0),depth,0).astype(np.float32)
    dcanvas[y:y+oh,x:x+ow]=np.asarray(Image.fromarray(clean).resize((ow,oh),Image.Resampling.NEAREST))
    return canvas,dcanvas

class PolicyWorker:
    def __init__(self,command,out):
        self.out=Path(out);self.stderr=(self.out/'policy-stderr.log').open('w')
        env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','LD_','ISAAC','CARB','EXP_','DEFORMABLE','CUDA_PATH'))}
        env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(self.out),TEMP=str(self.out),TMP=str(self.out))
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1,env=env)
        self.handshake=self.read(180);assert self.handshake.get('status')=='ready',self.handshake
    def read(self,timeout):
        if not select.select([self.process.stdout],[],[],timeout)[0]:raise TimeoutError('Policy worker timeout')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Policy worker exited: inspect policy-stderr.log')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result)
        return result
    def predict(self,obs):
        path=self.out/'policy-observation.npz';tick=time.monotonic();np.savez(path,**obs)
        write_s=time.monotonic()-tick;tick=time.monotonic()
        self.process.stdin.write(json.dumps(dict(command='predict',observation_path=str(path)))+'\n');self.process.stdin.flush()
        result=self.read(120);return result,dict(observation_ipc_write_s=write_s,inference_ipc_wall_s=time.monotonic()-tick)
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10);raise
        self.stderr.close()
        if self.process.returncode:raise RuntimeError('Policy worker nonzero exit '+str(self.process.returncode))

class EpisodeHook:
    def __init__(self,case,out):
        self.case=case;self.spec=case['learning'];self.mode=self.spec['mode'];self.out=Path(out)
        self.contract=dict(CONTRACT);self.rows=[];self.current=None;self.events=[];self.worker=None;self.closed=False
        tick=time.monotonic()
        if self.mode=='eval':self.worker=PolicyWorker(self.spec['policy_command'],self.out)
        self.startup_s=time.monotonic()-tick
    def observe(self,rgb,depth,t,tool,gauge_z,previous,s):
        rgb,depth=image_observation(rgb,depth)
        proprio=np.asarray([t,*np.asarray(tool).tolist(),gauge_z,previous,s['target_position_m'],float(s['feedback_active'])],np.float32)
        assert proprio.shape==(8,) and np.isfinite(proprio).all()
        self.current=dict(rgb=rgb,depth_m=depth,proprio=proprio,time_s=float(t),action_start_s=float(t)+1/self.case['config']['solver']['hz'])
    def act(self,policy,t,s):
        old=float(policy.correction);terminal=t>=self.case['config']['duration_s']-1e-8
        # Do not invoke a predictor without a following physical action; inactive
        # predictions are still consumed so history has the same 10Hz timeline.
        if not terminal:
            prediction,event=self.worker.predict({k:self.current[k] for k in ['rgb','depth_m','proprio']})
            raw=np.asarray(prediction['action'],float)
            assert raw.shape==(1,) and np.isfinite(raw).all()
            requested=float(raw[0]);event.update(time_s=t,requested_correction_m=requested);self.events.append(event)
        else:requested=old
        new=float(np.clip(old+np.clip(requested-old,-.0005,.0005),-.004,.004)) if s['feedback_active'] and not terminal else old
        policy.correction=new
        return dict(time_s=t,target_position_m=s['target_position_m'],observed_position_m=None,
            observation_valid=True,active=bool(s['feedback_active']),correction_m=new,correction_change_m=new-old,
            requested_correction_m=requested,command_effect='next_physics_step',controller='diffusion_residual',
            observation_age_s=0.,software_sequence=round(t*10))
    def record(self,action):
        assert self.current is not None
        if self.current['time_s']<self.case['config']['duration_s']-1e-8:
            row=dict(self.current);row['action']=np.asarray([action['correction_m']],np.float32);self.rows.append(row)
    def finish(self,timing,process_started,loop_started):
        tick=time.monotonic();assert len(self.rows)==44,len(self.rows)
        arrays={key:np.stack([row[key]for row in self.rows]) for key in ['rgb','depth_m','proprio','action','time_s','action_start_s']}
        assert np.all(np.diff(arrays['time_s'])>0) and np.all(arrays['action_start_s']>arrays['time_s'])
        path=self.out/'episode.npz';np.savez_compressed(path,**arrays)
        export_s=time.monotonic()-tick
        # All physics/feedback loop work and evidence writing stay in measured
        # wall time. Extra post-loop dataset serialization is an additive cost.
        costs=dict(cold_start_s=loop_started-process_started,loop_wall_s=timing['loop_elapsed_s'],
            dataset_export_s=export_s,policy_startup_s=self.startup_s,
            elapsed_through_dataset_s=time.monotonic()-process_started,
            scope='Raw render/evidence stages are in feedback-profile.json; no cost subtractions')
        metadata=dict(self.spec,backend=self.case['backend'],contract=self.contract,samples=44,terminal_observation_excluded=True,
            dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cost=costs,
            task_success=None,task_success_source='Filled only by independent full-horizon audit',
            policy_handshake=self.worker.handshake if self.worker else None,policy_events=self.events)
        metadata.pop('policy_command',None)
        (self.out/'episode.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        return dict(dataset_path=str(path),samples=44,cost=costs)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.worker:self.worker.close()
