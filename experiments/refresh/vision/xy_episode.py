"""Full XY-control recording. Separate contract from old1D policy experiments."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from learning_hook import image_observation


class XYEpisodeHook:
    def __init__(self,case,out):
        self.case=case;self.spec=case['learning'];self.mode='demo';self.out=Path(out);self.rows=[];self.current=None
        self.contract=dict(schema_version=2,action=['pusher_x_residual_m','pusher_y_residual_m'],
            proprio=['time_s','tool_x_m','tool_y_m','tool_z_m','gauge_z_m','previous_x_correction_m','previous_y_correction_m','scheduled_target_x_m','feedback_active'],
            action_bounds_m=[[-.004,.004],[-.008,.008]],max_action_change_m=.0005,
            feedback_period_s=.1,observation_route=case['xy_control']['observation'],
            task_scope='Robot XY side-push and full precision-gauge endpoint; truth route is development diagnostic only')
    def observe(self,rgb,depth,t,tool,gauge_z,previous,s):
        rgb,depth=image_observation(rgb,depth);previous=np.asarray(previous,float)
        assert previous.shape==(2,)
        proprio=np.asarray([t,*np.asarray(tool).tolist(),gauge_z,*previous.tolist(),s['target_position_m'],float(s['feedback_active'])],np.float32)
        assert proprio.shape==(9,)and np.isfinite(proprio).all()
        self.current=dict(rgb=rgb,depth_m=depth,proprio=proprio,time_s=float(t),action_start_s=float(t)+1/self.case['config']['solver']['hz'])
    def record(self,action):
        if self.current['time_s']<self.case['config']['duration_s']-1e-8:
            self.rows.append(dict(self.current,action=np.asarray([action['correction_m'],action['correction_y_m']],np.float32)))
    def finish(self,timing,process_started,loop_started):
        tick=time.monotonic();assert len(self.rows)==44
        arrays={key:np.stack([r[key]for r in self.rows])for key in ['rgb','depth_m','proprio','action','time_s','action_start_s']}
        path=self.out/'episode.npz';np.savez_compressed(path,**arrays)
        cost=dict(cold_start_s=loop_started-process_started,loop_wall_s=timing['loop_elapsed_s'],dataset_export_s=time.monotonic()-tick,
            elapsed_through_dataset_s=time.monotonic()-process_started,scope='Complete new XY contract; do not pool costs with old1D acquisition')
        meta=dict(self.spec,mode='demo',backend=self.case['backend'],contract=self.contract,samples=44,cost=cost,
            dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),task_success=None,
            observation_route=self.case['xy_control']['observation'],development_only=bool(self.case.get('development_smoke',False)))
        (self.out/'episode.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
        return dict(dataset_path=str(path),samples=44,cost=cost)
    def close(self):pass
