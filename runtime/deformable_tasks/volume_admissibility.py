"""Experimental native-step / volume-inequality operator splitting adapter."""
import json,sys
import numpy as np
from .volume_constraints import color_tetrahedra
from .backends import VOLUME_BACKENDS

class VolumeAdmissibility:
    def __init__(self,root,runtime,backend,policy=None):
        self.runtime=runtime;self.bodies=[];self.steps=0
        self.policy=policy
        self.enabled=backend in VOLUME_BACKENDS
        self.projecting=VOLUME_BACKENDS.get(backend,False)
        if not self.enabled:return
        import warp as wp
        from . import volume_constraints_gpu as gpu
        from omni.physx import get_physx_interface
        from .backends import observe_volumes
        observe_volumes(root,runtime)  # Pin and shape validation before pointer use.
        sys.path.insert(0,str(root/'work/native-readback'))
        import physx_deformable_readback as readback
        wp.load_module(module=gpu,device='cuda:0')
        for body in runtime.bodies:
            if body['kind']!='volume':raise ValueError('Volume constraint applied to cloth')
            handle=readback.simulation_buffer(get_physx_interface(),body['path'])
            native=wp.array(ptr=handle['device_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0').numpy()
            n=len(native);elements=body['sim_elements'];rest6=body['sim_restv']*6
            if not np.all(rest6>0):raise ValueError('Positive rest tetrahedra required')
            groups=color_tetrahedra(elements,n)
            data=dict(body=body,n=n,p=wp.empty(n,dtype=wp.vec3,device='cuda:0'),v=wp.empty(n,dtype=wp.vec3,device='cuda:0'),
                x=wp.empty(n,dtype=wp.vec3d,device='cuda:0'),tets=wp.array(elements,dtype=wp.vec4i,device='cuda:0'),
                invmass=wp.array(native[:,3].astype(float),dtype=wp.float64,device='cuda:0'),
                rest6=wp.array(rest6.astype(float),dtype=wp.float64,device='cuda:0'),
                groups=[wp.array(g,dtype=wp.int32,device='cuda:0') for g in groups],
                stats=wp.array(np.array([0,1,0,0,0,0,0,1],float),dtype=wp.float64,device='cuda:0'),
                impulse=wp.zeros(3,dtype=wp.float64,device='cuda:0'))
            with wp.ScopedCapture(device='cuda:0') as capture:
                data['impulse'].zero_()
                wp.launch(gpu.measure,len(elements),[data['p'],data['tets'],data['rest6'],data['stats'],1],device='cuda:0')
                wp.launch(gpu.begin,n,[data['p'],data['x']],device='cuda:0')
                if self.projecting:
                    for _ in range(policy['sweeps']):
                        for group in data['groups']:
                            wp.launch(gpu.project,len(group),[data['x'],data['tets'],group,data['invmass'],data['rest6'],policy['floor'],data['stats']],device='cuda:0')
                wp.launch(gpu.publish,n,[data['p'],data['v'],data['x'],data['invmass'],float(runtime.config['solver']['hz']),data['stats'],data['impulse']],device='cuda:0')
                wp.launch(gpu.impulse_check,1,[data['impulse'],data['stats']],device='cuda:0')
                wp.launch(gpu.measure,len(elements),[data['p'],data['tets'],data['rest6'],data['stats'],7],device='cuda:0')
            data['graph']=capture.graph;self.bodies.append(data)

    def apply(self):
        if not self.enabled:return
        import warp as wp
        from isaacsim.core.experimental.utils.backend import use_backend
        # PhysX and Warp may own different CUDA streams. Do not rely on legacy
        # default-stream behavior at the public tensor / captured-graph boundary.
        # These fences are part of the measured solver cost.
        wp.synchronize_device('cuda:0')
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            for b in self.bodies:
                view=b['body']['view'];p=view.get_nodal_positions()[0];v=view.get_nodal_velocities();n=b['n']
                wp.synchronize_device('cuda:0')
                pview=wp.array(ptr=p.ptr,shape=(n,),dtype=wp.vec3,device='cuda:0')
                vview=wp.array(ptr=v.ptr,shape=(n,),dtype=wp.vec3,device='cuda:0')
                wp.copy(b['p'],pview);wp.copy(b['v'],vview);wp.capture_launch(b['graph'])
                wp.synchronize_device('cuda:0')
                outp=wp.array(ptr=b['p'].ptr,shape=(1,n,3),dtype=wp.float32,device='cuda:0')
                outv=wp.array(ptr=b['v'].ptr,shape=(1,n,3),dtype=wp.float32,device='cuda:0')
                view.set_nodal_positions(outp);view.set_nodal_velocities(outv)
        self.steps+=1

    def report(self):
        if not self.enabled:return None
        bodies=[]
        for b in self.bodies:
            s=b['stats'].numpy()
            bodies.append(dict(path=b['body']['path'],colors=len(b['groups']),constraint_updates=int(s[0]),
                native_endpoint_min_volume_ratio=float(s[1]),native_endpoint_inverted_tet_observations=int(s[2]),
                published_min_volume_ratio=float(s[7]),maximum_constraint_violation=max(0.,self.policy['floor']-float(s[7])),
                cumulative_projection_kinetic_work_j=float(s[3]),maximum_nodal_correction_m=float(s[4]),
                maximum_position_publication_roundoff_m=float(s[5]),maximum_projection_net_impulse_ns=float(s[6])))
        return dict(active=self.projecting,steps=self.steps,volume_ratio_floor=self.policy['floor'],sweeps=self.policy['sweeps'],bodies=bodies,
            checks=dict(no_native_endpoint_inversion=all(b['native_endpoint_inverted_tet_observations']==0 for b in bodies),
                        volume_constraint_converged=(not self.projecting) or all(b['maximum_constraint_violation']<1e-3 for b in bodies),
                        projection_linear_impulse=all(b['maximum_projection_net_impulse_ns']<1e-6 for b in bodies),
                        projection_position_publication=all(b['maximum_position_publication_roundoff_m']<5e-8 for b in bodies)),
            scope='Experimental split inequality solver. Real nodal positions and velocities are updated by local mass-weighted constraint impulses. No table/press clamping, trajectory prescription or global momentum reset. Reaction-force accuracy, internal-stage inversion freedom and energy fidelity remain unverified.')
