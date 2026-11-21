"""Native controls inside the full Isaac application; no external simulation loop."""
import asyncio
import gc
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import uuid
from .volume_quality import assess_volume

import carb
import numpy as np
import omni.ext
import omni.kit.app
import omni.kit.window.file
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Usd, UsdGeom, Vt
from isaacsim.core.experimental.prims import DeformablePrim
from isaacsim.core.experimental.utils.backend import use_backend
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from omni.kit.viewport.utility import get_active_viewport

_instance = None


def get_instance():
    return _instance


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        global _instance
        self.ext_id = ext_id
        self.root = Path(__file__).resolve().parents[4]
        self.catalog = json.loads((self.root / 'configs/deformable-lab-scenes.json').read_text())
        self.body = None
        self.volume_reference = None
        self.stability_violation = None
        self.failure_state = None
        self.case = None
        self.loaded_layer = None
        self.samples = []
        self.latest = {}
        self.ticks = 0
        self.errors = []
        self.tasks = set()
        self.timeline = omni.timeline.get_timeline_interface()
        self.callback = None
        self.visual = None
        self.initial_points = None
        self.stop_subscription = self.timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._on_stop)
        self.window = ui.Window('Deformable Lab', width=430, height=520)
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8):
                    ui.Label('DEFORMABLE LAB', height=26, style={'font_size': 20})
                    ui.Label('Cloth and soft solids in the native physics timeline', height=26, word_wrap=True)
                    with ui.HStack(height=32, spacing=6):
                        ui.Button('Cloth drape', clicked_fn=lambda: self.request_load('surface'))
                        ui.Button('Soft block', clicked_fn=lambda: self.request_load('volume'))
                    self.baseline_label = ui.Label('Choose a scene to begin.', height=56, word_wrap=True)
                    with ui.HStack(height=24):
                        ui.Label('Young modulus (Pa)', width=210)
                        self.young = ui.FloatField().model
                    with ui.HStack(height=24):
                        ui.Label('Density (kg / m3)', width=210)
                        self.density = ui.FloatField().model
                    with ui.HStack(height=24):
                        ui.Label('Poisson ratio', width=210)
                        self.poisson = ui.FloatField().model
                    with ui.HStack(height=24):
                        self.guard_enabled = ui.SimpleBoolModel(True)
                        ui.CheckBox(model=self.guard_enabled, width=24)
                        ui.Label('Pause on invalid volume mesh')
                    ui.Button('Apply material and reset', height=28, clicked_fn=self.apply_material)
                    with ui.HStack(height=32, spacing=6):
                        ui.Button('Play', clicked_fn=self.play)
                        ui.Button('Pause', clicked_fn=self.timeline.pause)
                        ui.Button('Reset', clicked_fn=self.reset)
                    ui.Separator(height=4)
                    self.telemetry = ui.Label('Measured node state appears while playing.', height=98, word_wrap=True)
                    with ui.HStack(height=30, spacing=6):
                        ui.Button('Save scene copy', clicked_fn=self.save_scene)
                        ui.Button('Export measurements', clicked_fn=self.export_measurements)
                    self.status = ui.Label('Ready', height=54, word_wrap=True)
                    ui.Spacer()
        import sys
        if str(self.root) not in sys.path: sys.path.insert(0, str(self.root))
        from deformable_tasks.panel import TaskPanel
        self.workbench = TaskPanel(self)
        _instance = self
        self._schedule(self._dock())

    def _schedule(self, coroutine):
        task = asyncio.ensure_future(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def _dock(self):
        await omni.kit.app.get_app().next_update_async()
        target = ui.Workspace.get_window('Property')
        if target:
            self.window.dock_in(target, ui.DockPosition.SAME)

    def request_load(self, case):
        # Use the application's save/discard prompt for any existing user edits.
        omni.kit.window.file.prompt_if_unsaved_stage(lambda: self._schedule(self.load_case(case)))

    async def load_case(self, case):
        try:
            assert case in ('surface', 'volume')
            self.workbench.detach()
            self.timeline.stop()
            self._detach_callback()
            for _ in range(3):
                await omni.kit.app.get_app().next_update_async()
            self.body = None
            self.visual = None
            gc.collect()
            context = omni.usd.get_context()
            if context.get_stage() is not None:
                await context.close_stage_async()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            entry = self.catalog['scenes'][case]
            path = (self.root / entry['path']).resolve()
            assert path.is_relative_to(self.root / 'results') and path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'Scene changed since registration'
            context = omni.usd.get_context()
            # The catalog is frozen evidence; edit a private working copy in the UI.
            working_scene = self._output_dir() / "scene.usda"
            shutil.copy2(path, working_scene)
            self._prepare_visual(working_scene, case)
            ok, error = await context.open_stage_async(str(working_scene))
            if not ok:
                raise RuntimeError(error)
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()
            assert SimulationManager.get_active_physics_engine() == 'physx'
            SimulationManager.set_device('cuda:0')
            # Host-readback display bridge: physics remains CUDA; visualization uses USD.
            # Do not advertise this compatibility path as GPU zero-copy rendering.
            SimulationManager.enable_fabric(False)
            self.case = case
            self.loaded_layer = context.get_stage().GetRootLayer().identifier
            self.body = DeformablePrim('/World/Deformable', deformable_type=case, reset_xform_op_properties=False)
            self.visual = UsdGeom.Mesh(context.get_stage().GetPrimAtPath('/World/DeformableDisplay'))
            self.initial_points = np.array(self.visual.GetPointsAttr().Get(), dtype=np.float32)
            # Opening a USD stage clears SimulationManager callbacks. Register only now.
            self.callback = SimulationManager.register_callback(self._on_physics, event=IsaacEvents.POST_PHYSICS_STEP)
            self.samples, self.latest, self.ticks, self.errors = [], {}, 0, []
            self.volume_reference, self.stability_violation, self.failure_state = None, None, None
            material = context.get_stage().GetPrimAtPath('/World/DeformableMaterial')
            for model, attr in [(self.young, 'omniphysics:youngsModulus'), (self.density, 'omniphysics:density'),
                                (self.poisson, 'omniphysics:poissonsRatio')]:
                model.set_value(float(material.GetAttribute(attr).Get()))
            self.baseline_label.text = entry['description']
            self.telemetry.text = 'Ready to measure GPU node positions and velocity.'
            self.status.text = 'Scene loaded. Use Play or adjust the material.'
            viewport = get_active_viewport()
            if viewport:
                ViewportManager.set_camera_view(str(viewport.camera_path), eye=[2.3, -2.3, 1.7], target=[0, 0, .45])
            context.get_selection().set_selected_prim_paths(['/World/DeformableDisplay'], True)
            self.window.focus()
            return True
        except Exception as error:
            self._error(error)
            return False

    @staticmethod
    def _prepare_visual(path, case):
        # Only project-generated, identity-transform scenes are accepted here.
        # The visual mesh has no physics schema and exactly the simulation's node ordering.
        stage = Usd.Stage.Open(str(path))
        source = stage.GetPrimAtPath('/World/Deformable')
        assert not UsdGeom.Xformable(source).GetOrderedXformOps()
        points = source.GetAttribute('points').Get()
        if case == 'volume':
            indices = np.asarray(source.GetAttribute('surfaceFaceVertexIndices').Get(), dtype=np.int32).reshape(-1)
            counts = np.full(len(indices) // 3, 3, dtype=np.int32)
        else:
            indices = np.asarray(source.GetAttribute('faceVertexIndices').Get(), dtype=np.int32)
            counts = np.asarray(source.GetAttribute('faceVertexCounts').Get(), dtype=np.int32)
        visual = UsdGeom.Mesh.Define(stage, '/World/DeformableDisplay')
        visual.CreatePointsAttr(points)
        visual.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        visual.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
        visual.CreateSubdivisionSchemeAttr('none')
        visual.CreateDoubleSidedAttr(True)
        visual.CreateOrientationAttr(source.GetAttribute('orientation').Get())
        visual.CreateDisplayColorAttr(source.GetAttribute('primvars:displayColor').Get())
        visual.CreateExtentAttr(source.GetAttribute('extent').Get())
        UsdGeom.Imageable(source).MakeInvisible()
        stage.GetRootLayer().Save()

    def _sync_visual(self, points):
        if self.visual and self.visual.GetPrim().IsValid():
            self.visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32, copy=False)))
            bounds = np.asarray([points.min(axis=0), points.max(axis=0)], dtype=np.float32)
            self.visual.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(bounds))

    def _on_stop(self, event):
        self.volume_reference, self.stability_violation, self.failure_state = None, None, None
        if self.initial_points is not None:
            self._sync_visual(self.initial_points)
        self.samples, self.latest, self.ticks = [], {}, 0

    def _error(self, error):
        message = f'{type(error).__name__}: {error}'
        self.errors.append(message)
        self.status.text = message
        carb.log_error('Deformable Lab: ' + message)

    def _check_stage(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None or stage.GetRootLayer().identifier != self.loaded_layer or self.body is None:
            raise RuntimeError('Load a Deformable Lab scene before using these controls.')
        return stage

    def apply_material(self):
        try:
            stage = self._check_stage()
            young, density, poisson = self.young.as_float, self.density.as_float, self.poisson.as_float
            if not (np.isfinite([young, density, poisson]).all() and young > 0 and density > 0 and 0 <= poisson < .49):
                raise ValueError('Use positive stiffness and density; Poisson ratio must be in [0, 0.49).')
            self.reset()
            material = stage.GetPrimAtPath('/World/DeformableMaterial')
            for key, value in [('omniphysics:youngsModulus', young), ('omniphysics:density', density), ('omniphysics:poissonsRatio', poisson)]:
                assert material.GetAttribute(key).Set(value)
            self.status.text = 'Material updated. Baseline results apply only to the original parameters.'
        except Exception as error:
            self._error(error)

    def play(self):
        try:
            self._check_stage()
            if self.guard_enabled.as_bool and self.stability_violation is not None:
                raise RuntimeError('Reset the invalid volume scene before replaying.')
            self.timeline.play()
        except Exception as error:
            self._error(error)

    def reset(self):
        self.volume_reference, self.stability_violation, self.failure_state = None, None, None
        self.timeline.stop()
        self.samples, self.latest, self.ticks = [], {}, 0
        self.telemetry.text = 'Reset to the authored initial state.'

    def _on_physics(self, *args):
        if self.body is None or not self.timeline.is_playing():
            return
        self.ticks += 1
        if not self.body.is_physics_tensor_entity_valid():
            return
        try:
            self._check_stage()
            with use_backend('tensor', raise_on_fallback=True, raise_on_unsupported=True):
                node_arrays = self.body.get_nodal_positions()
                positions = node_arrays[0]
                points = positions.numpy()[0]
                assert str(positions.device) == 'cuda:0' and np.isfinite(points).all()
                self._sync_visual(points)
                if self.case == 'volume':
                    if self.volume_reference is None:
                        self.volume_reference = (node_arrays[2].numpy()[0].copy(),
                                                 self.body.get_element_indices()[0].numpy()[0].copy())
                    quality = assess_volume(points, *self.volume_reference)
                    if not quality['valid']:
                        if self.stability_violation is None:
                            self.failure_state = dict(points=points.copy(), rest_points=self.volume_reference[0].copy(), elements=self.volume_reference[1].copy())
                            self.stability_violation = dict(**quality, time_s=float(SimulationManager.get_simulation_time()),
                                physics_step=int(SimulationManager.get_num_physics_steps()))
                        if self.guard_enabled.as_bool:
                            self.timeline.pause()
                            self.telemetry.text = f"Invalid volume mesh at {self.stability_violation['time_s']:.3f} s\nMinimum volume ratio: {quality['minimum_signed_volume_ratio']:.6f}"
                            self.status.text = 'Paused by quality monitor. Failure preserved; reset before replay.'
                            return
                if self.ticks % 12:
                    return
                velocity = self.body.get_nodal_velocities()
            assert str(positions.device) == 'cuda:0' and str(velocity.device) == 'cuda:0'
            points, velocities = positions.numpy()[0], velocity.numpy()[0]
            assert np.isfinite(points).all() and np.isfinite(velocities).all()
            self.latest = {'time_s': float(SimulationManager.get_simulation_time()), 'physics_step': int(SimulationManager.get_num_physics_steps()),
                           'nodes': len(points), 'device': str(positions.device), 'minimum_z_m': float(points[:, 2].min()),
                           'node_centroid_z_m': float(points[:, 2].mean()), 'maximum_speed_m_s': float(np.linalg.norm(velocities, axis=1).max())}
            self.samples.append(self.latest.copy())
            # Bounded UI telemetry, never advertise this host-readback path as a solver benchmark.
            if len(self.samples) > 10000:
                del self.samples[:1000]
            q = self.latest
            self.telemetry.text = (f"Time: {q['time_s']:.2f} s    Nodes: {q['nodes']}\n"
                                   f"Centroid height: {q['node_centroid_z_m']:.4f} m\n"
                                   f"Lowest node: {q['minimum_z_m']:.4f} m\n"
                                   f"Maximum speed: {q['maximum_speed_m_s']:.3f} m/s")
        except Exception as error:
            self.timeline.pause()
            self._error(error)

    def _output_dir(self):
        path = self.root / 'results/deformable-lab-exports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        path.mkdir(parents=True, exist_ok=False)
        return path

    def save_scene(self):
        try:
            stage = self._check_stage()
            if not self.timeline.is_stopped():
                raise RuntimeError('Reset before saving an initial scene; live solver state is not a resumable USD snapshot.')
            out = self._output_dir()
            assert stage.Export(str(out / 'scene.usda'))
            self.status.text = 'Saved scene copy: ' + out.name
            return out
        except Exception as error:
            self._error(error)

    def export_measurements(self):
        try:
            self._check_stage()
            if not self.samples and self.stability_violation is None:
                raise RuntimeError('Play the scene to collect measurements first.')
            out = self._output_dir()
            (out / 'measurements.json').write_text(json.dumps({'case': self.case, 'samples': self.samples, 'stability_violation': self.stability_violation,
                'errors': self.errors,
                'volume_validity_observed': self.case == 'volume' and self.volume_reference is not None and self.stability_violation is None and not self.errors,
                'scope': 'Sparse live GPU node diagnostics with host readback; not a solver-speed or material-accuracy benchmark.'}, indent=2))
            if self.failure_state is not None:
                np.savez_compressed(out / 'invalid-volume-state.npz', **self.failure_state)
            self.status.text = 'Exported measurements: ' + out.name
            return out
        except Exception as error:
            self._error(error)

    def _detach_callback(self):
        if self.callback is not None:
            SimulationManager.deregister_callback(self.callback)
            self.callback = None

    def on_shutdown(self):
        global _instance
        if hasattr(self, 'workbench'): self.workbench.shutdown()
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()
        self._detach_callback()
        self.body = None
        self.visual = None
        self.stop_subscription = None
        if self.window:
            self.window.destroy()
            self.window = None
        _instance = None
