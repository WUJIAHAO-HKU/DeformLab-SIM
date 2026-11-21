"""Same pinned VBD operations via two parity CUDA graphs; no model changes."""
import time
import warp as wp
from newton_backend import Backend as PlainBackend

class Backend(PlainBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.graphs=[]
        for a,b in [(self.state,self.next),(self.next,self.state)]:
            with wp.ScopedCapture(device=self.model.device) as capture:
                a.clear_forces();self.pipeline.collide(a,self.contacts)
                self.solver.step(a,b,self.control,self.contacts,self.dt)
            self.graphs.append(capture.graph)
        self.receipt['execution_mode']='Alternating CUDA graphs, one step per graph, identical target update and state readback'

    def advance(self,correction,offset,count):
        from side_push_config_v3 import target_pose
        from gauge_articulation import target
        samples=[]
        for _ in range(count):
            step=self.steps+1;t=step/self.hz
            command=target_pose(t,correction,offset);gz,gv=target(t,self.gauge_spec)
            self.targets[:3]=command-self.initial;self.targets[3]=gz;self.target_vel[3]=gv
            tick=time.monotonic();self.control.joint_target_q.assign(self.targets);self.control.joint_target_qd.assign(self.target_vel)
            wp.capture_launch(self.graphs[self.steps%2]);self.state,self.next=self.next,self.state;wp.synchronize_device()
            self.compute_s.append(time.monotonic()-tick);self.steps=step;samples.append(self.sample(command))
        return samples
