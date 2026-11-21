"""Explicit backend capabilities and reproducible schedule transformations.

Substep is a PhysX scheduling candidate, not the CP kernel intervention.
No unsupported CP task silently falls back to native execution.
"""
import copy,ctypes,hashlib,sys,os
from pathlib import Path
from .config import validate,default_config
from .repair_contract import selection
from .weight_backend import WEIGHT_BACKENDS

BACKENDS=('native','substep','cp-control','cp-sequential','cp-polar','cp-residual','cp-coupled','cp-split-damping','cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one','contact-refresh','contact-adaptive','contact-discrete','contact-local','volume-project','volume-project-control','cp-volume-project','cp-volume-project-control')
from .ss_common_backend import COMMON_BACKENDS
from .batched_cp_backend import BACKENDS as BATCHED_CP_BACKENDS
from .cloth_split_backend import BACKENDS as CLOTH_SPLIT_BACKENDS,REFRESH_BACKENDS,ENVELOPE_BACKENDS,REFRESH16_BACKENDS
from .robot_surface_backend import BACKENDS as ROBOT_SURFACE_BACKENDS
BACKENDS=BACKENDS+WEIGHT_BACKENDS+tuple(COMMON_BACKENDS)+tuple(CLOTH_SPLIT_BACKENDS)+tuple(BATCHED_CP_BACKENDS)+ROBOT_SURFACE_BACKENDS
CP_MODES={'cp-sequential':4,'cp-polar':6,'cp-residual':8,'cp-coupled':6,'cp-split-damping':6,'cp-contact-clone':6,'cp-contact-velocity':6,'cp-contact-stabilized':6,'cp-contact-joint':6,'cp-contact-joint-terminal':6,'cp-contact-joint-terminal-coupled':6,'cp-contact-joint-terminal-pose':6,'cp-contact-joint-terminal-rank-one':6,'cp-volume-project':4,'cp-volume-project-control':4}
VOLUME_BACKENDS={'volume-project':True,'volume-project-control':False,'cp-volume-project':True,'cp-volume-project-control':False}
PIN='05220192126a9bacedeb432eca6070be2faddc5879475b90be00bfeb815c1cc7'

def plan(config,backend,iterations=None,volume_floor=None,volume_sweeps=None,damping_integrator=None):
    if backend not in BACKENDS:raise ValueError('Unknown backend: '+backend)
    if backend in ROBOT_SURFACE_BACKENDS:raise ValueError('Finite-table robot surface backend requires robot_volume; no cloth/multibody fallback')
    if backend in WEIGHT_BACKENDS:raise ValueError('Weight contact experiment requires the robot_volume route; no cloth/multibody fallback')
    source=validate(config);effective=copy.deepcopy(source)
    if backend in BATCHED_CP_BACKENDS:
        from .batched_cp_backend import requested as batch_requested
        batch_requested(source['task'],backend)
        if source['geometry']!=default_config('multi_soft')['geometry'] or not source['solver']['contacts']:raise ValueError('Batched CP live route requires default two-volume geometry and contacts')
    if backend in CLOTH_SPLIT_BACKENDS:
        from .cloth_split_backend import requested
        requested(source['task'],backend)
        if not source['solver']['contacts'] or not source['solver']['self_collision']:raise ValueError('Cloth split requires contacts and self-collision; no inactive repair claim')
    if backend in COMMON_BACKENDS and source['task']!='multi_soft':raise ValueError('Common SS binding supports exactly two volume bodies in multi_soft; no task fallback')
    if backend in ('ss-cone-control','ss-cone-plane','ss-cp-observe-native','ss-cp-map-native','ss-cp-map-common','volume-mapping') and (source['geometry']!=default_config('multi_soft')['geometry'] or not source['solver']['contacts']):
        raise ValueError('Cone experiment requires the default two-volume geometry with contacts enabled; auxiliary z=0.12 plane only')
    split_backends=('cp-split-damping','cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one')
    if damping_integrator is not None and (damping_integrator not in ('backward-euler','exponential') or backend not in split_backends):
        raise ValueError('Damping integrator requires a split-damping backend and a supported method')
    damping_policy={'gamma':source['material']['elasticity_damping']} if backend in split_backends else None
    if damping_integrator is not None:damping_policy['integrator']=damping_integrator
    if backend in ('cp-split-damping','cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):effective['material']['elasticity_damping']=0.
    if backend in ('cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one') and source['material']['friction']!=0:raise ValueError('Contact velocity ablation requires zero friction')
    if backend in VOLUME_BACKENDS and source['task'] not in ('multi_soft','soft_squeeze'):
        raise ValueError('Volume admissibility requires tetrahedral bodies')
    if volume_sweeps is not None and (backend not in VOLUME_BACKENDS or type(volume_sweeps) is not int or not 1<=volume_sweeps<=64):
        raise ValueError('Volume sweeps require a volume-project backend and an integer in [1, 64]')
    if volume_floor is not None and (backend not in VOLUME_BACKENDS or not .001<=volume_floor<=.5):
        raise ValueError('Volume floor must be in [0.001, 0.5] and requires a volume-project backend')
    if backend in ('contact-refresh','contact-adaptive','contact-local') and not source['task'].startswith('cloth'):
        raise ValueError('Surface contact scheduling supports cloth only; no volume fallback')
    if backend.startswith('cp-') and source['task']!='soft_squeeze':
        raise ValueError('CP adapters support exactly one volumetric body; cloth/multi-body are unsupported')
    if backend=='substep':
        effective['solver']['hz']*=2
        effective['solver']['position_iterations']=max(1,source['solver']['position_iterations']//2)
    if iterations is not None:effective['solver']['position_iterations']=iterations
    effective=validate(effective)
    if abs(effective['duration_s']*effective['solver']['hz']-round(effective['duration_s']*effective['solver']['hz']))>1e-7:
        raise ValueError('Duration must contain an integral number of physics steps')
    hz=effective['solver']['hz'];n=effective['solver']['position_iterations']
    contact_schedule=({'rigid_contact_offset_m':.02,'pair_updates':16,'collision_multiplier':1} if backend in REFRESH16_BACKENDS else
                      {'rigid_contact_offset_m':.02,'pair_updates':4,'collision_multiplier':1} if backend in ENVELOPE_BACKENDS and backend in REFRESH_BACKENDS else
                      {'rigid_contact_offset_m':.02} if backend in ENVELOPE_BACKENDS else
                      {'pair_updates':4,'collision_multiplier':1} if backend in REFRESH_BACKENDS else
                      {'speculative_ccd':False} if backend=='contact-discrete' else
                      {'pair_updates':0,'collision_multiplier':2,'filter_distance':.0016} if backend=='contact-local' else
                      {'pair_updates':4,'collision_multiplier':2} if backend=='contact-refresh' else
                      {'pair_updates':0,'collision_multiplier':2} if backend=='contact-adaptive' else None)
    if backend in REFRESH16_BACKENDS and n<16:raise ValueError('Sixteen pair refreshes require at least sixteen iterations')
    if contact_schedule and 'pair_updates' in contact_schedule and n<4:raise ValueError('Surface contact schedule requires at least 4 iterations')
    return effective,dict(name=backend,repair_selection=selection(backend),external_every_iteration=backend.startswith('cp-'),contact_schedule=contact_schedule,
        material_damping_policy=damping_policy,
        coupled_material_linearization=backend in ('cp-coupled','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'),
        volume_policy=({'floor':.15 if volume_floor is None else volume_floor,'sweeps':8 if volume_sweeps is None else volume_sweeps} if backend in VOLUME_BACKENDS else None),
        kernel_intervention=backend in CP_MODES,requested_config=source,
        physics_hz=hz,position_iterations=n,nominal_iteration_work_per_sim_second=hz*n,
        scope={**{name:'Restricted two-volume CP mapping with frame validation and GPU execution counters; no material or cost acceptance' for name in BATCHED_CP_BACKENDS},**{name:'Experimental single-cloth VT/EE split contact with execution counters; physical and cost acceptance require complete comparisons' for name in CLOTH_SPLIT_BACKENDS},**{name:'Experimental two-volume SS common relaxation; synchronized diagnostic binding, no trajectory or performance acceptance' for name in COMMON_BACKENDS},'volume-mapping':'Shared volume CP mass/velocity mapping; native contacts; explicit task-specific bridges, no auxiliary projection or platform acceptance','ss-cp-observe-native':'Two-volume native CP and native contacts with matched synchronized observation; no CP repair or auxiliary projection','ss-cp-map-native':'Two-volume CP mass/velocity mapping with native soft-soft contacts; no auxiliary plane; instrumented diagnostic only','ss-cp-map-common':'Two-volume CP mapping with common SS weight/history relaxation; no auxiliary plane; instrumented diagnostic only','ss-cone-control':'SS common history plus native velocity-stage split control; auxiliary-plane diagnostic only','ss-cone-plane':'SS common history plus pre-drift velocity cone and explicit auxiliary fixed-plane impulse ledger; no general contact or performance acceptance','cp-contact-joint-terminal-rank-one':'Experimental device-only coupled unilateral contact with terminal pose and coupled material; audited fixed Cartesian coupon only',
               'cp-contact-joint-terminal-pose':'Experimental terminal deferred-velocity pose/history update; fixed Cartesian coupon only, no convergence or work acceptance',
               'native':'Pinned native PhysX baseline','substep':'Native PhysX temporal refinement; no custom CP or contact kernel',
               'contact-refresh':'Native surface contacts refreshed four times per step, two collision subiterations; no custom kernel',
               'contact-adaptive':'Native adaptive surface contact refresh, two collision subiterations; no custom kernel',
               'contact-discrete':'Native contacts with speculative CCD disabled; diagnostic schedule, no state projection',
               'contact-local':'Adaptive surface contact refresh, two collision subiterations, explicit 1.6 mm self-contact filter',
               'cp-control':'Native constitutive solver with CP-compatible initialization controls',
               'volume-project':'Experimental native step plus colored GPU positive-volume inequality solve; changes the discrete admissibility domain',
               'volume-project-control':'Native step plus identical GPU nodal read/write adapter without volume projection',
               'cp-volume-project':'Single-volume CP mass/velocity and sequential damping update plus end-step positive-volume inequality projection',
               'cp-volume-project-control':'Single-volume CP mass/velocity and sequential damping update plus nodal read/write control without projection',
               'cp-polar':'Single-volume CP mass/velocity, sequential damping and current-strain double polar rotation',
               'cp-coupled':'Experimental single-volume polar update with a coupled ARAP/volume damped constraint linearization',
               'cp-contact-clone':'Matched native contact PTX clone with split material damping; diagnostic control',
               'cp-contact-joint-terminal-coupled':'Experimental shared-input ARAP/volume linearization with terminal identity, joint contact and split material damping; audited Cartesian coupon only',
               'cp-contact-joint-terminal':'Experimental terminal material identity transport; ordinary material and all contact updates retained, causal ablation only',
               'cp-contact-joint':'Experimental joint gap/normal-velocity inequality Jacobi sweep, fixed linearization, split damping',
               'cp-contact-stabilized':'Experimental relative-normal-velocity contact with explicit 0.1 penetration bias and split damping',
               'cp-contact-velocity':'Experimental relative-normal-velocity contact projection, zero position bias, split material damping',
               'cp-split-damping':'Diagnostic polar elastic step with zero native damping plus whole-body implicit material velocity damping using the requested gamma',
               'cp-residual':'Single-volume CP polar update with cross-substep position residual accumulation',
               'cp-sequential':'Version-locked single-volume CP mass/velocity + sequential damping intervention'}[backend])

def hook_library(root,backend):
    if backend not in CP_MODES:raise ValueError('Hook library requires a CP intervention')
    coupled=os.environ.get('DEFORMABLE_DRIVE_CONTACT','0')
    if coupled not in ('0','1'):raise ValueError('Drive contact selection must be 0 or 1')
    residual=os.environ.get('DEFORMABLE_ROOT_RESIDUAL','0')
    if residual not in ('0','1') or (residual=='1' and coupled!='1'):
        raise ValueError('Root residual requires explicit 0/1 selection and contact-drive coupling')
    if coupled=='1':
        if backend!='cp-contact-joint-terminal-rank-one' or os.environ.get('DEFORMABLE_CAPTURE_DRIVE_LAUNCH','0')!='0':
            raise ValueError('Drive contact requires its isolated rank-one hook without launch-capture mode')
        name='libphysx_cp_root_residual_hook.so' if residual=='1' else 'libphysx_cp_drive_contact_hook.so'
        return Path(root)/'work/kernel-observer'/name
    capture=os.environ.get('DEFORMABLE_CAPTURE_DRIVE_LAUNCH','0')
    if capture not in ('0','1'):raise ValueError('Drive capture selection must be 0 or 1')
    if capture=='1':
        if backend!='cp-contact-joint-terminal-rank-one':raise ValueError('Drive launch capture requires the verified rank-one coupon backend')
        return Path(root)/'work/kernel-observer/libphysx_cp_drive_capture_hook.so'
    if backend=='cp-contact-joint-terminal-rank-one':return Path(root)/'work/kernel-observer/libphysx_cp_rank_one_hook.so'
    if backend=='cp-contact-joint-terminal-pose':return Path(root)/'work/kernel-observer/libphysx_cp_terminal_pose_hook.so'
    filename='libphysx_cp_terminal_hook.so' if backend in ('cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one') else 'libphysx_cp_trial_hook.so'
    return Path(root)/'work/kernel-observer'/filename

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def observe_volumes(root,runtime):
    """Pinned SDK read-only evidence, outside the measured simulation interval."""
    bodies=[b for b in runtime.bodies if b['kind']=='volume']
    if not bodies:return []
    paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libPhysXGpu' in line})
    if len(paths)!=1 or sha(paths[0])!=PIN:raise RuntimeError('Volume diagnostic binary mismatch')
    sys.path.insert(0,str(Path(root)/'work/native-readback'))
    import physx_deformable_readback as readback
    from omni.physx import get_physx_interface
    import warp as wp
    import numpy as np
    result=[]
    for body in bodies:
        handle=readback.simulation_buffer(get_physx_interface(),body['path'])
        state=wp.array(ptr=handle['device_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0').numpy().copy()
        if not np.all(state[:,3]>0):raise RuntimeError('Unexpected nonpositive nodal inverse mass')
        if np.max(np.abs(state[:,:3]-body['last']))>1e-6:raise RuntimeError('Native diagnostic node mismatch')
        result.append(dict(path=body['path'],settings=handle['settings'],node_count=len(state),
            actual_mass_kg=float((1/state[:,3].astype(float)).sum()),
            authored_mass_kg=float(np.abs(body['restv']).sum()*runtime.config['material']['density']),
            native_gpu_sha256=PIN,readback_sha256=sha(Path(root)/'work/native-readback/physx_deformable_readback.so')))
    return result

class CpAdapter:
    def __init__(self,root,runtime,backend):
        self.observer=None;self.root=Path(root);self.backend=backend;self.evidence={};self.runtime=runtime
        if backend!='cp-control' and backend not in CP_MODES:return
        assert len(runtime.bodies)==1 and runtime.bodies[0]['kind']=='volume'
        paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libPhysXGpu' in line})
        if len(paths)!=1 or sha(paths[0])!=PIN:raise RuntimeError('CP pinned binary mismatch')
        sys.path.insert(0,str(self.root/'work/native-readback'))
        import physx_deformable_readback as readback
        import physx_deformable_probe_controls as controls
        from omni.physx import get_physx_interface
        from isaacsim.core.experimental.utils.backend import use_backend
        import warp as wp
        import numpy as np
        body=runtime.bodies[0];path=body['path'];interface=get_physx_interface()
        controls.configure(interface,path,0,False)
        handle=readback.simulation_buffer(interface,path)
        assert handle['settings']['material_model']==0
        state=wp.array(ptr=handle['device_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0').numpy().copy()
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):nodes=body['view'].get_nodal_positions()[0].numpy()[0]
        assert np.all(state[:,3]>0) and np.max(np.abs(state[:,:3]-nodes))<1e-6
        self.evidence=dict(native_gpu_sha256=PIN,node_count=len(state),mass_kg=float((1/state[:,3].astype(float)).sum()),
            material_model=0,private_abi=True,hook_active=False,controls_sha256=sha(self.root/'work/native-readback/physx_deformable_probe_controls.so'))
        if backend=='cp-control':return
        lib=hook_library(self.root,backend);ptx=self.root/'work/kernel-observer/physx_cp_conservative_trial.ptx'
        self.observer=ctypes.CDLL(str(lib));o=self.observer
        if not hasattr(o,'observer_lifecycle_version'):raise RuntimeError('CP hook must be rebuilt with scene lifecycle reset support')
        o.observer_lifecycle_version.restype=ctypes.c_int
        if o.observer_lifecycle_version()!=1:raise RuntimeError('Unsupported CP lifecycle ABI')
        self.evidence['lifecycle_version']=1
        o.observer_start.argtypes=[ctypes.c_char_p,ctypes.c_ulonglong,ctypes.c_ulonglong,ctypes.c_size_t]
        o.observer_enable_trial.argtypes=[ctypes.c_char_p,ctypes.c_int]
        o.observer_trial_count.argtypes=[ctypes.c_int];o.observer_trial_count.restype=ctypes.c_ulonglong
        o.observer_stop.restype=ctypes.c_uint
        code=o.observer_start(b'',handle['device_pointer'],handle['velocity_pointer'],handle['node_count'])
        if code:raise RuntimeError('CP initialization error '+str(code))
        code=o.observer_enable_trial(str(ptx).encode(),CP_MODES[backend])
        if code:raise RuntimeError('CP activation error '+str(code))
        self.evidence.update(hook_active=True,mode=CP_MODES[backend],hook_sha256=sha(lib),ptx_sha256=sha(ptx))
        if os.environ.get('DEFORMABLE_CAPTURE_DRIVE_LAUNCH','0')=='1':
            if not hasattr(o,'observer_drive_launch_start'):raise RuntimeError('Selected library lacks drive launch observation')
            self.evidence['drive_launch_capture']=True
        if backend in ('cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
            variant={'cp-contact-clone':'clone','cp-contact-velocity':'velocity','cp-contact-stabilized':'stabilized','cp-contact-joint':'joint','cp-contact-joint-terminal':'joint','cp-contact-joint-terminal-coupled':'joint','cp-contact-joint-terminal-pose':'joint','cp-contact-joint-terminal-rank-one':'joint'}[backend]
            contact_ptx=self.root/('work/kernel-observer/contact-normal-'+variant+'.ptx')
            o.observer_normal_contact_start.argtypes=[ctypes.c_char_p,ctypes.c_int]
            o.observer_normal_contact_start.restype=ctypes.c_int
            o.observer_normal_contact_count.restype=ctypes.c_ulonglong
            if o.observer_normal_contact_start(str(contact_ptx).encode(),1 if variant=='clone' else 2):raise RuntimeError('Normal contact activation failed')
            self.evidence['normal_contact']=dict(variant=variant,ptx_sha256=sha(contact_ptx),position_bias={'clone':'native','velocity':0.,'stabilized':.1,'joint':'min(relative_velocity, unclamped_gap/substep)'}[variant])
        if backend in ('cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
            settings=handle['settings'];n=runtime.config['solver']['position_iterations']
            if settings['position_iterations']!=n or settings['velocity_iterations']!=1:
                raise RuntimeError('Terminal experiment requires matched N position and one velocity iteration')
            terminal_ptx=self.root/'work/kernel-observer/physx_terminal_material.ptx'
            o.observer_terminal_material_start.argtypes=[ctypes.c_char_p,ctypes.c_uint]
            o.observer_terminal_material_start.restype=ctypes.c_int
            o.observer_terminal_material_count.argtypes=[ctypes.c_int]
            o.observer_terminal_material_count.restype=ctypes.c_ulonglong
            if o.observer_terminal_material_start(str(terminal_ptx).encode(),n):
                raise RuntimeError('Terminal material schedule activation failed')
            self.evidence['terminal_material']=dict(policy='identity-copy-transport',ptx_sha256=sha(terminal_ptx),
                ordinary_material_updates=n,terminal_transports=1,contact_updates=n+1,
                scope='Causal ablation; no claim of fully coupled feasibility, passivity or convergence')
        if backend in ('cp-coupled','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
            o.observer_coupled_damping.argtypes=[ctypes.c_int];o.observer_coupled_damping.restype=ctypes.c_int
            if o.observer_coupled_damping(1):raise RuntimeError('Coupled material update activation failed')
            self.evidence['coupled_damping']=True
        if backend in ('cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
            pose_ptx=self.root/'work/kernel-observer/terminal_prismatic_correction.ptx'
            h=1/(runtime.config['solver']['hz']*runtime.config['solver']['position_iterations'])
            o.observer_terminal_pose_start.argtypes=[ctypes.c_char_p,ctypes.c_float]
            o.observer_terminal_pose_start.restype=ctypes.c_int
            o.observer_terminal_pose_count.restype=ctypes.c_ulonglong
            if o.observer_terminal_pose_start(str(pose_ptx).encode(),h):raise RuntimeError('Terminal pose activation failed')
            self.evidence['terminal_pose']=dict(policy='deferred-velocity-position-history',ptx_sha256=sha(pose_ptx),
                inner_dt_s=float(ctypes.c_float(h).value),scope='Explicit Cartesian causal experiment; no precision-residual, feasibility, work or convergence acceptance')

        if backend=='cp-contact-joint-terminal-rank-one':
            if len(runtime.drives)!=1 or runtime.drives[0].dof_names!=['DriveX','DriveY','DriveZ']:
                raise RuntimeError('Rank-one contact requires the fixed Cartesian coupon')
            with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
                mass=runtime.drives[0].get_mass_matrices().numpy()[0].astype(float)
            expected=np.diag([2.2,1.2,.2])
            if mass.shape!=(3,3) or np.linalg.norm(mass-expected)>2e-6*np.linalg.norm(expected):
                raise RuntimeError('Rank-one contact generalized mass mismatch')
            adapter_ptx=self.root/'work/kernel-observer/native_rank_one_adapter.ptx'
            transport_ptx=self.root/'work/kernel-observer/contact-normal-rank-one.ptx'
            o.observer_rank_one_contact_start.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
            o.observer_rank_one_contact_start.restype=ctypes.c_int
            o.observer_rank_one_contact_count.restype=ctypes.c_ulonglong
            if o.observer_rank_one_contact_start(str(adapter_ptx).encode(),str(transport_ptx).encode()):
                raise RuntimeError('Rank-one contact activation failed')
            self.evidence['normal_contact'].update(variant='rank-one',position_bias='coupled min(relative_velocity, unclamped_gap/substep)',
                ptx_sha256=sha(transport_ptx),bootstrap_ptx_sha256=sha(contact_ptx))
            self.evidence['rank_one_contact']=dict(adapter_ptx_sha256=sha(adapter_ptx),transport_ptx_sha256=sha(transport_ptx),
                initial_generalized_mass_kg=mass.tolist(),authored_generalized_mass_kg=expected.tolist(),
                scope='Device-only grouping and solve; <=1024 rows and <=128 nodes per world-z face; native trajectory and work acceptance remain separate')

        if os.environ.get('DEFORMABLE_DRIVE_CONTACT','0')=='1':
            stage=self.root/'work/kernel-observer/native_drive_contact.ptx'
            transfer=self.root/'work/kernel-observer/native_drive_pending.ptx'
            o.observer_drive_contact_start.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
            o.observer_drive_contact_start.restype=ctypes.c_int
            o.observer_drive_contact_count.argtypes=[ctypes.c_uint]
            o.observer_drive_contact_count.restype=ctypes.c_ulonglong
            if o.observer_drive_contact_start(str(stage).encode(),str(transfer).encode()):
                raise RuntimeError('Contact-drive candidate activation failed')
            self.evidence['drive_contact']=dict(stage_ptx_sha256=sha(stage),transport_ptx_sha256=sha(transfer),
                policy='position-pass implicit response with paired native drive increment after articulation push',
                terminal_velocity_policy='original physical response and zero added drive impulse',
                platform_acceptance=False)

        if os.environ.get('DEFORMABLE_ROOT_RESIDUAL','0')=='1':
            o.observer_root_residual_start.argtypes=[];o.observer_root_residual_start.restype=ctypes.c_int
            o.observer_root_residual_enabled.argtypes=[];o.observer_root_residual_enabled.restype=ctypes.c_int
            if o.observer_root_residual_start() or o.observer_root_residual_enabled()!=1:
                raise RuntimeError('Root position residual activation failed')
            self.evidence['root_residual']=dict(policy='compensated root position and frame delta through integration, material merge and contact publication',
                velocity_policy='native contact/integration velocity retained; material mean-copy merge retained',platform_acceptance=False)

    def check_active(self):
        if self.observer:
            counts=[int(self.observer.observer_trial_count(i)) for i in (0,1,2,6)]
            if not (counts[0]>0 and counts[0]==counts[1]==counts[2] and counts[3]>=counts[0]):raise RuntimeError('CP intervention not active on all observed rounds: '+str(counts))

    def stop(self):
        if self.observer:
            self.check_active()
            self.evidence['launch_counts']=[int(self.observer.observer_trial_count(i)) for i in (0,1,2,6)]
            rounds=self.runtime.steps*(self.runtime.config['solver']['position_iterations']+1)
            self.evidence['coverage_passed']=self.evidence['launch_counts']==[rounds,rounds,rounds,8*rounds]
            if self.backend in ('cp-contact-clone','cp-contact-velocity','cp-contact-stabilized','cp-contact-joint','cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
                calls=int(self.observer.observer_normal_contact_count())
                self.evidence['normal_contact']['calls']=calls
                self.evidence['coverage_passed'] &= calls==rounds
            if self.backend in ('cp-contact-joint-terminal','cp-contact-joint-terminal-coupled','cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
                terminal=[int(self.observer.observer_terminal_material_count(i)) for i in (0,1)]
                expected=[8*self.runtime.steps,self.runtime.steps]
                self.evidence['terminal_material'].update(counts=terminal,expected_counts=expected)
                self.evidence['coverage_passed'] &= terminal==expected
            if self.backend=='cp-residual' or 'root_residual' in self.evidence:
                residual=[int(self.observer.observer_trial_count(i)) for i in (7,8,9)]
                self.evidence['residual_stage_counts']=residual
                expected=[self.runtime.steps,self.runtime.steps*self.runtime.config['solver']['position_iterations'],4*rounds]
                self.evidence['residual_expected_counts']=expected
                self.evidence['coverage_passed'] &= residual==expected
            if self.backend in ('cp-contact-joint-terminal-pose','cp-contact-joint-terminal-rank-one'):
                frames=int(self.observer.observer_terminal_pose_count())
                self.evidence['terminal_pose'].update(frames=frames,expected_frames=self.runtime.steps)
                self.evidence['coverage_passed'] &= frames==self.runtime.steps
            if self.backend=='cp-contact-joint-terminal-rank-one':
                calls=int(self.observer.observer_rank_one_contact_count())
                self.evidence['rank_one_contact'].update(calls=calls,expected_calls=rounds)
                self.evidence['coverage_passed'] &= calls==rounds
            if 'drive_contact' in self.evidence:
                counters=[int(self.observer.observer_drive_contact_count(i)) for i in range(3)]
                self.evidence['drive_contact'].update(counts=counters,expected_counts=[rounds]*3)
                self.evidence['coverage_passed'] &= counters==[rounds]*3
            errors=int(self.observer.observer_stop());self.observer=None
            self.evidence['hook_errors']=errors
            # Preserve the failed trajectory and counters instead of losing them
            # to a finalization exception. The runner requires this acceptance bit.
            self.evidence['accepted']=errors==0 and self.evidence['coverage_passed']
        return self.evidence
