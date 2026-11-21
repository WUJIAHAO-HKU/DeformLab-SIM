"""CPU-only preservation of complete readbacks when a GPU task raises.

Complete readback is not physical validity. No failed GPU state is fetched and
no last-known state is mislabeled as the attempted failing step.
"""
import hashlib,json
from pathlib import Path
import numpy as np

class FailureCheckpoint:
    def __init__(self,bodies):
        self.geometry=[]
        for b in bodies:
            self.geometry.append({k:np.array(v,copy=True) for k,v in dict(rest=b['rest'],faces=b['faces'],elements=b.get('sim_elements',b['elements'])).items()})
        self.last=None

    def capture(self,step,time_s,native_time_s,nodes):
        if len(nodes)!=len(self.geometry):raise ValueError('Incomplete body readback')
        copied=[np.array(n,copy=True) for n in nodes]
        if any(n.shape!=g['rest'].shape for n,g in zip(copied,self.geometry)):raise ValueError('Readback topology changed')
        # Publish the CPU checkpoint only after every body has been copied.
        self.last=dict(step=int(step),time_s=float(time_s),native_time_s=float(native_time_s),nodes=copied)

    def save(self,dest,*,error,attempted_step,nodes,times,native_times,timings,diagnostics):
        if self.last is None:raise ValueError('No complete CPU readback')
        dest=Path(dest);data=dest/'exception-trajectory.npz';receipt=dest/'exception-checkpoint.json'
        if data.exists() or receipt.exists():raise FileExistsError('Exception evidence already exists')
        if len(nodes)!=len(times) or len(nodes)!=len(native_times):raise ValueError('Misaligned buffered samples')
        frames=list(nodes);t=list(times);nt=list(native_times)
        if not t or t[-1]!=self.last['time_s']:
            frames.append(self.last['nodes']);t.append(self.last['time_s']);nt.append(self.last['native_time_s'])
        elif any(not np.array_equal(frames[-1][i],v,equal_nan=True) for i,v in enumerate(self.last['nodes'])):
            raise ValueError('Buffered and checkpoint states disagree at the same time')
        arrays=dict(time_s=np.asarray(t),native_time_s=np.asarray(nt),control_physics_sync_s=np.asarray(timings),diagnostics_s=np.asarray(diagnostics))
        for i,g in enumerate(self.geometry):
            arrays[f'nodes_{i}']=np.asarray([frame[i] for frame in frames])
            for k,v in g.items():arrays[f'{k}_{i}']=v
        with data.open('xb') as f:np.savez_compressed(f,**arrays)
        result=dict(status='failed',error=str(error),attempted_step=int(attempted_step),
            last_complete_readback_step=self.last['step'],last_complete_readback_time_s=self.last['time_s'],
            sampled_frames=len(frames),body_count=len(self.geometry),
            control_timing_attempts=len(timings),completed_diagnostic_timings=len(diagnostics),
            failed_step_state_captured=False,physical_validity_accepted=False,complete_horizon=False,
            source='Previously copied CPU readbacks only; no GPU calls during exception handling',
            trajectory=str(data),trajectory_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
        with receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        return result
