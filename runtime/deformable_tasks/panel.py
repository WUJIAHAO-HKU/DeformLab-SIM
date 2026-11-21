"""Native task workbench. Config reload is deterministic reset, not checkpoint restore."""
import asyncio
import gc
import json
from datetime import datetime,timezone
from pathlib import Path
import uuid
import omni.kit.app
import omni.kit.window.file
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.core.simulation_manager import IsaacEvents,SimulationManager
from .config import TASKS,LABELS,default_config,validate,save_config,load_config,digest
from .runtime import TaskRuntime,ROOT
from .backends import plan
UI_BACKENDS=('native','substep','contact-adaptive','contact-local')

class TaskPanel:
    def __init__(self,owner):
        self.owner=owner;self.runtime=None;self.config=default_config();self.busy=False;self.run_dir=None;self.generation=0
        self.backend_name='native';self.backend_metadata=None;self.iteration_override=None
        self.timeline=omni.timeline.get_timeline_interface()
        self.window=ui.Window('Interaction Tasks',width=460,height=680)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=7):
                    ui.Label('INTERACTION TASKS',height=30,style={'font_size':20})
                    ui.Label('Shared configuration | Native GPU physics | Studio renderer',height=32,word_wrap=True)
                    self.task=ui.ComboBox(0,*LABELS).model
                    self.backend_selector=ui.ComboBox(0,'Native PhysX','Substep candidate (2x Hz / half iterations)',
                        'Adaptive contact refresh (cloth)','Adaptive contact + local filter (cloth)').model
                    ui.Label('Fields below describe the base configuration. CP single-volume tests use the batch entry.',height=36,word_wrap=True)
                    self.fields={}
                    for label,key,value in [('Young modulus / Pa','young_pa',20000.),('Density / kg m-3','density',500.),('Friction','friction',.8),('Solid size / m','size_m',.08),('Physics frequency / Hz','hz',480),('Position iterations','position_iterations',64),('Compression fraction','compression',.55)]:
                        with ui.HStack(height=24):
                            ui.Label(label,width=245)
                            model=ui.IntField().model if key in ('hz','position_iterations') else ui.FloatField().model
                            model.set_value(value);self.fields[key]=model
                    with ui.HStack(height=24):
                        self.self_contact=ui.SimpleBoolModel(True);ui.CheckBox(model=self.self_contact,width=24);ui.Label('Enable deformable self contact')
                    with ui.HStack(height=24):
                        ui.Label('Rendering',width=245);self.render=ui.ComboBox(0,'RTX studio','Path tracing').model
                    ui.Button('Build / apply configuration',height=32,clicked_fn=self.request_build)
                    with ui.HStack(height=32):
                        ui.Button('Play',clicked_fn=self.play);ui.Button('Pause',clicked_fn=self.timeline.pause)
                        ui.Button('Reset task',clicked_fn=lambda:self.owner._schedule(self.load(self.config)))
                    ui.Separator(height=4)
                    ui.Label('Save bundle / reopen task.json (initial state)',height=24)
                    self.path=ui.StringField(height=26).model
                    self.path.set_value(str(ROOT/'configs/tasks/cloth_grip.json'))
                    with ui.HStack(height=30):
                        ui.Button('Save bundle',clicked_fn=self.save)
                        ui.Button('Reopen config',clicked_fn=self.request_reopen)
                    ui.Button('Export measured state',height=28,clicked_fn=self.export)
                    ui.Button('Open architectural render reference',height=30,clicked_fn=lambda:omni.kit.window.file.prompt_if_unsaved_stage(lambda:self.owner._schedule(self.open_architecture())))
                    self.telemetry=ui.Label('No task loaded.',height=110,word_wrap=True)
                    self.status=ui.Label('Cartesian fixtures use joint drives. Friction grip has no cloth attachments.',height=72,word_wrap=True)
                    ui.Spacer()
        self.stop_sub=self.timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(int(omni.timeline.TimelineEventType.STOP),self.on_stop)
        self.stage_sub=omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self.on_stage_event)
        owner._schedule(self.dock())

    async def dock(self):
        await omni.kit.app.get_app().next_update_async()
        target=ui.Workspace.get_window('Property')
        if target:self.window.dock_in(target,ui.DockPosition.SAME)

    def read_fields(self):
        c=default_config(TASKS[self.task.get_item_value_model().as_int])
        # Preserve non-visible fields when editing the currently selected task.
        if c['task']==self.config['task']:c=json.loads(json.dumps(self.config))
        for k,model in self.fields.items():
            group='material' if k in ('young_pa','density','friction') else 'solver' if k in ('hz','position_iterations') else 'geometry' if k=='size_m' else 'action'
            c[group][k]=model.as_int if k in ('hz','position_iterations') else model.as_float
        c['solver']['self_collision']=self.self_contact.as_bool
        c['rendering']['profile']='pathtraced' if self.render.get_item_value_model().as_int else 'studio'
        return validate(c)

    def request_build(self):
        try:c=self.read_fields()
        except Exception as e:self.status.text=str(e);return
        backend=UI_BACKENDS[self.backend_selector.get_item_value_model().as_int]
        omni.kit.window.file.prompt_if_unsaved_stage(lambda:self.owner._schedule(self.load(c,backend)))

    def request_reopen(self):
        try:
            c=load_config(self.path.as_string)
            selected=Path(self.path.as_string);sidecar=selected.with_name('backend.json')
            bundle=json.loads(sidecar.read_text()) if selected.name=='task.json' and sidecar.exists() else None
            backend=bundle['name'] if bundle else 'native';iterations=None
            if backend not in UI_BACKENDS:raise ValueError('This experimental backend requires a fresh batch process; reopening as another backend is refused.')
            if bundle:
                requested=validate(bundle['requested_config']);iterations=bundle['position_iterations']
                effective,_=plan(requested,backend,iterations)
                if digest(c) not in (digest(requested),digest(effective)):raise ValueError('Configuration and backend bundle disagree')
                c=requested
        except Exception as e:self.status.text=str(e);return
        omni.kit.window.file.prompt_if_unsaved_stage(lambda:self.owner._schedule(self.load(c,backend,iterations)))

    def detach(self):
        if self.runtime:self.runtime.release();self.runtime=None

    async def load(self,config,backend=None,iterations=None):
        if self.busy:raise RuntimeError('Task switch already in progress')
        if backend is None and iterations is None:iterations=self.iteration_override
        backend=backend or self.backend_name
        if backend not in UI_BACKENDS:raise ValueError('Use a fresh batch process for this experimental backend.')
        effective,metadata=plan(config,backend,iterations)
        config=validate(config);self.busy=True;self.status.text='Building task…'
        try:
            self.timeline.stop();self.detach();self.owner._detach_callback()
            self.owner.body=self.owner.visual=self.owner.initial_points=None
            self.owner.case=None;self.owner.samples=[]
            app=omni.kit.app.get_app();context=omni.usd.get_context()
            for _ in range(3):await app.next_update_async()
            gc.collect()
            if context.get_stage() is not None:await context.close_stage_async()
            for _ in range(5):await app.next_update_async()
            await context.new_stage_async()
            self.runtime=TaskRuntime(effective,contact_schedule=metadata.get('contact_schedule')).build(context.get_stage());self.config=config;self.generation+=1
            self.backend_name=backend;self.backend_metadata=metadata
            self.iteration_override=iterations
            self.run_dir=ROOT/'results/task-workbench'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])
            self.run_dir.mkdir(parents=True);save_config(self.run_dir/'task.json',config)
            save_config(self.run_dir/'effective-task.json',effective)
            (self.run_dir/'backend.json').write_text(json.dumps(metadata,indent=2)+'\n')
            context.get_stage().Export(str(self.run_dir/'initial.usda'))
            self.runtime.callback_ids=[SimulationManager.register_callback(self.before_step,event=IsaacEvents.PRE_PHYSICS_STEP),SimulationManager.register_callback(self.after_step,event=IsaacEvents.POST_PHYSICS_STEP)]
            from omni.kit.viewport.utility import get_active_viewport
            vp=get_active_viewport()
            if vp:vp.camera_path='/World/Camera'
            import carb
            carb.settings.get_settings().set('/rtx/rendermode','PathTracing' if config['rendering']['profile']=='pathtraced' else 'RaytracedLighting')
            self.task.get_item_value_model().set_value(TASKS.index(config['task']))
            self.backend_selector.get_item_value_model().set_value(UI_BACKENDS.index(backend))
            for k,model in self.fields.items():
                group='material' if k in ('young_pa','density','friction') else 'solver' if k in ('hz','position_iterations') else 'geometry' if k=='size_m' else 'action'
                model.set_value(config[group][k])
            self.self_contact.set_value(config['solver']['self_collision']);self.render.get_item_value_model().set_value(int(config['rendering']['profile']=='pathtraced'))
            self.path.set_value(str(self.run_dir/'task.json'))
            self.status.text=f"Ready: {config['task']} | {backend}\nEffective: {effective['solver']['hz']} Hz / {effective['solver']['position_iterations']} iterations. Reset/reopen reconstructs initial state."
            self.telemetry.text='Awaiting playback.'
            return True
        except Exception as e:
            self.detach();self.status.text=type(e).__name__+': '+str(e);raise
        finally:self.busy=False

    async def open_architecture(self):
        path=ROOT/'assets/architecture/brownstone-interior.usda'
        if not path.exists():
            self.status.text='Architectural asset not installed; see docs/task-platform.md.';return False
        if self.busy:return False
        self.busy=True
        try:
            self.timeline.stop();self.detach();self.owner._detach_callback()
            self.owner.body=self.owner.visual=self.owner.initial_points=None
            context=omni.usd.get_context();app=omni.kit.app.get_app()
            if context.get_stage():await context.close_stage_async()
            self.status.text='Loading architectural reference assets...'
            await context.open_stage_async(str(path))
            for _ in range(20):await app.next_update_async()
            from omni.kit.viewport.utility import get_active_viewport
            vp=get_active_viewport()
            if vp:vp.camera_path='/World/Brownstone01/Cameras/Cam_BS1_L1_Kitchen001'
            import carb
            settings=carb.settings.get_settings();settings.set('/app/viewport/grid/enabled',False)
            settings.set('/rtx/rendermode','PathTracing');settings.set('/rtx/pathtracing/totalSpp',256)
            self.status.text='NVIDIA Brownstone interior. Rendering reference; interaction tasks are validated separately.'
            return True
        except Exception as e:self.status.text=str(e);raise
        finally:self.busy=False

    def before_step(self,*args):
        if not self.runtime or not self.timeline.is_playing():return
        try:self.runtime.before_step(*args)
        except Exception as e:self.fail(e)

    def after_step(self,*args):
        if not self.runtime or not self.timeline.is_playing():return
        try:
            self.runtime.after_step(*args);r=self.runtime
            if r.error:raise RuntimeError(r.error)
            if r.steps%12==0:
                self.telemetry.text=f"{self.config['task']}  |  t = {r.latest['time_s']:.3f} s\nGPU nodes: {sum(len(b['rest']) for b in r.bodies)}\nMin volume ratio: {r.minimum_volume_ratio:.4f}\nMaximum bench penetration: {r.maximum_penetration*1000:.2f} mm\nRender / physics discrepancy: {r.latest['display_error_m']:.2e} m"
            if r.latest['time_s']>=self.config['duration_s']:
                self.timeline.pause();self.export();self.status.text='Task complete. Measurements exported; inspect individual checks.'
        except Exception as e:self.fail(e)

    def fail(self,e):
        self.timeline.pause()
        if self.runtime:self.runtime.error=str(e)
        self.status.text='Paused: '+str(e);export=self.export()
        if export and not (export/'first-failure.json').exists():
            import shutil
            shutil.copy2(export/'measurements.json',export/'first-failure.json')
            shutil.copy2(export/'measured-state.npz',export/'first-failure.npz')

    def play(self):
        if self.runtime and not self.busy and not self.runtime.error:self.timeline.play()

    def on_stop(self,event):
        if self.runtime and not self.busy:
            # Native timeline stop resets physics; mirror the task controller clock.
            r=self.runtime;r.steps=0;r.samples=[];r.latest={};r.error=None
            r.minimum_volume_ratio=1.;r.maximum_penetration=0.;r.maximum_speed=0.
            r.maximum_actuator_error=0.;r.maximum_press_penetration=0.
            r.maximum_attachment_error=0.;r.state_finite=True
            for b in r.bodies:b['visual'].GetPointsAttr().Set(r.Vt.Vec3fArray.FromNumpy(b['rest']))
            for prim,(_,position,_) in zip(r.mover_visuals,r.movers):prim.GetAttribute('xformOp:translate').Set(r.Gf.Vec3d(*[float(v) for v in position]))

    def on_stage_event(self,event):
        if event.type==int(omni.usd.StageEventType.CLOSING) and not self.busy:
            self.detach();self.status.text='Stage closed; task callbacks released.'

    def save(self):
        if not self.runtime:return None
        save_config(self.run_dir/'task.json',self.config)
        # initial.usda was captured before physics; avoid mislabelling a deformed export as a checkpoint.
        self.export();self.path.set_value(str(self.run_dir/'task.json'));self.status.text='Saved task bundle: '+str(self.run_dir)
        return self.run_dir

    def export(self):
        if not self.runtime or not self.run_dir:return None
        (self.run_dir/'measurements.json').write_text(json.dumps(self.runtime.report(),indent=2)+'\n')
        import numpy as np
        states={f'nodes_{i}':b.get('last',b['rest']) for i,b in enumerate(self.runtime.bodies)}
        np.savez_compressed(self.run_dir/'measured-state.npz',**states)
        return self.run_dir

    def shutdown(self):
        self.detach();self.stop_sub=None;self.stage_sub=None
        if self.window:self.window.destroy();self.window=None
