# SPDX-License-Identifier: Apache-2.0
"""Local desktop RPC and a single-thread-owned Newton CUDA workbench.

The module imports without scientific packages so validation and HTTP boundary
tests can run independently. Newton and Warp are imported only by the simulation
worker. This is an interactive baseline, not a calibrated material model.
"""

from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


SERVICE = "deformable-sim-workbench"
FRAME_DT = 1.0 / 60.0
MAX_BODY_BYTES = 16_384
CLOTH_SIDE = 1.2
PRESETS = {
    "hanging_cloth": {
        "preset": "hanging_cloth",
        "cloth": {"stiffness": 1000.0, "bending": 0.2, "mass": 0.25, "resolution": 20, "height": 1.15},
        "sphere": {"radius": 0.24, "mass": 0.1, "height": 0.3},
        "solver": {"substeps": 10, "iterations": 10},
    },
    "cloth_drop": {
        "preset": "cloth_drop",
        "cloth": {"stiffness": 1000.0, "bending": 0.2, "mass": 0.25, "resolution": 20, "height": 1.3},
        "sphere": {"radius": 0.3, "mass": 0.1, "height": 0.4},
        "solver": {"substeps": 10, "iterations": 10},
    },
    "sphere_impact": {
        "preset": "sphere_impact",
        "cloth": {"stiffness": 1000.0, "bending": 0.2, "mass": 0.25, "resolution": 20, "height": 0.75},
        "sphere": {"radius": 0.16, "mass": 0.1, "height": 1.35},
        "solver": {"substeps": 10, "iterations": 10},
    },
}
LIMITS = {
    "cloth": {"stiffness": (100, 20_000), "bending": (0, 50), "mass": (0.05, 2),
              "resolution": (8, 40), "height": (0.3, 2.5)},
    "sphere": {"radius": (0.08, 0.4), "mass": (0.02, 2), "height": (0.1, 3)},
    "solver": {"substeps": (2, 30), "iterations": (2, 40)},
}
INTEGER_FIELDS = {("cloth", "resolution"), ("solver", "substeps"), ("solver", "iterations")}


class CommandError(ValueError):
    """A user-facing rejected command that leaves the simulation unchanged."""


def validate_scene(patch: dict, current: dict | None = None) -> dict:
    """Merge a bounded partial scene. Changing presets starts from its defaults."""
    if not isinstance(patch, dict):
        raise CommandError("scene must be a JSON object")
    if set(patch) - {"preset", *LIMITS}:
        raise CommandError("unknown scene field")
    preset = patch.get("preset", (current or PRESETS["hanging_cloth"])["preset"])
    if not isinstance(preset, str) or preset not in PRESETS:
        raise CommandError("unknown preset")
    scene = copy.deepcopy(current if current and current["preset"] == preset else PRESETS[preset])
    scene["preset"] = preset
    for group, limits in LIMITS.items():
        values = patch.get(group, {})
        if not isinstance(values, dict) or set(values) - set(limits):
            raise CommandError(f"invalid {group} fields")
        for key, value in values.items():
            low, high = limits[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CommandError(f"{group}.{key} must be numeric")
            if (group, key) in INTEGER_FIELDS and not isinstance(value, int):
                raise CommandError(f"{group}.{key} must be an integer")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or not low <= value <= high:
                raise CommandError(f"{group}.{key} must be between {low} and {high}")
            scene[group][key] = value
    cloth, sphere = scene["cloth"], scene["sphere"]
    if sphere["height"] - sphere["radius"] < 0.015:
        raise CommandError("sphere must start at least 0.015 m above the ground")
    if preset == "sphere_impact":
        gap = sphere["height"] - sphere["radius"] - cloth["height"]
    else:
        gap = cloth["height"] - sphere["height"] - sphere["radius"]
    if gap < 0.04:
        raise CommandError("cloth and sphere must start separated by at least 0.04 m")
    return scene


def validate_command(payload: dict) -> tuple[str, dict | None]:
    if not isinstance(payload, dict) or set(payload) - {"action", "scene"}:
        raise CommandError("command must contain only action and optional scene")
    action = payload.get("action")
    if not isinstance(action, str) or action not in {"play", "pause", "step", "reset", "configure"}:
        raise CommandError("unknown action")
    scene = payload.get("scene")
    if action == "configure" and not isinstance(scene, dict):
        raise CommandError("configure requires a scene object")
    if "scene" in payload and (action not in {"reset", "configure"} or not isinstance(scene, dict)):
        raise CommandError("scene is allowed only as an object for reset or configure")
    return action, scene


def parse_json(data: bytes) -> dict:
    def reject_constant(value):
        raise CommandError(f"non-finite JSON number: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CommandError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CommandError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CommandError("JSON body must be an object")
    return value


class NewtonSimulation:
    """All methods must be called on the same worker thread, including construction."""

    def __init__(self, scene: dict, device_name: str = "cuda:0", cache_dir: Path | None = None):
        import numpy as np
        import warp as wp

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            wp.config.kernel_cache_dir = str(cache_dir.resolve())
        import newton
        from newton.solvers import SolverVBD

        if newton.__version__ != "1.5.2" or wp.__version__ != "1.16.0":
            raise RuntimeError("workbench requires newton==1.5.2 and warp-lang==1.16.0")
        self.wp, self.np = wp, np
        self.device = wp.get_device(device_name)
        if not self.device.is_cuda:
            raise RuntimeError("workbench requires a CUDA device")
        self.device_label = f"{self.device} ({self.device.name})"
        self.scene = copy.deepcopy(scene)
        self.frame = 0
        self.compute_ms = 0.0
        self.sphere_body = None
        cloth, sphere, preset = scene["cloth"], scene["sphere"], scene["preset"]
        n = cloth["resolution"]
        count = (n + 1) ** 2
        if preset == "hanging_cloth":
            pinned = {0, n * (n + 1)}
        elif preset == "sphere_impact":
            pinned = {j * (n + 1) + i for j in range(n + 1) for i in (0, n)}
        else:
            pinned = set()
        # mass means total dynamic cloth mass. Pinning is an explicit boundary
        # condition; these scenes must not be evaluated as isolated systems.
        particle_mass = cloth["mass"] / (count - len(pinned))
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=wp.vec3(0, 0, -9.81))
            contact_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=10.0, mu=0.35,
                                                         mu_torsional=0.0, mu_rolling=0.0)
            builder.add_ground_plane(height=0.0, cfg=contact_cfg)
            builder.add_cloth_grid(
                pos=wp.vec3(-CLOTH_SIDE / 2, -CLOTH_SIDE / 2, cloth["height"]),
                rot=wp.quat_identity(), vel=wp.vec3(0), dim_x=n, dim_y=n,
                cell_x=CLOTH_SIDE / n, cell_y=CLOTH_SIDE / n, mass=particle_mass,
                tri_ke=cloth["stiffness"], tri_ka=cloth["stiffness"], tri_kd=0.1,
                edge_ke=cloth["bending"], edge_kd=0.0, tri_drag=0.0, tri_lift=0.0,
                particle_radius=0.008,
            )
            for index in pinned:
                builder.particle_mass[index] = 0.0
                builder.particle_flags[index] &= ~int(newton.ParticleFlags.ACTIVE)
            pose = wp.transform(wp.vec3(0, 0, sphere["height"]), wp.quat_identity())
            density = sphere["mass"] / ((4.0 / 3.0) * math.pi * sphere["radius"] ** 3)
            sphere_cfg = newton.ModelBuilder.ShapeConfig(density=density, ke=1.0e4, kd=10.0, mu=0.35,
                                                        mu_torsional=0.0, mu_rolling=0.0)
            if preset == "sphere_impact":
                self.sphere_body = builder.add_body(xform=pose, label="sphere")
                builder.add_shape_sphere(self.sphere_body, radius=sphere["radius"], cfg=sphere_cfg)
            else:
                builder.add_shape_sphere(-1, xform=pose, radius=sphere["radius"], cfg=sphere_cfg)
            builder.color(include_bending=True)
            self.model = builder.finalize(device=self.device)
            self.model.soft_contact_ke = 1.0e4
            self.model.soft_contact_kd = 10.0
            self.model.soft_contact_mu = 0.35
            self.state, self.next_state = self.model.state(), self.model.state()
            self.control = self.model.control()
            capacity = 2 * self.model.particle_count
            self.pipeline = newton.CollisionPipeline(self.model, soft_contact_max=capacity,
                                                     enable_rigid_soft_full_surface_contact=False,
                                                     verify_buffers=True)
            self.contacts = self.pipeline.contacts()
            self.solver = SolverVBD(self.model, iterations=scene["solver"]["iterations"],
                                    integrate_with_external_rigid_solver=False,
                                    particle_enable_self_contact=False, particle_enable_tile_solve=True,
                                    rigid_body_particle_contact_buffer_size=capacity)
            self.triangles = self.model.tri_indices.numpy().reshape(-1).tolist()
            if not self.state.particle_q.device.is_cuda:
                raise RuntimeError("particle state was not allocated on CUDA")
            wp.synchronize_device(self.device)

    def advance(self) -> None:
        wp = self.wp
        substeps = self.scene["solver"]["substeps"]
        start = time.perf_counter()
        with wp.ScopedDevice(self.device):
            for _ in range(substeps):
                self.state.clear_forces()
                self.pipeline.collide(self.state, self.contacts)
                self.solver.step(self.state, self.next_state, self.control, self.contacts, FRAME_DT / substeps)
                self.state, self.next_state = self.next_state, self.state
            wp.synchronize_device(self.device)
        self.compute_ms = (time.perf_counter() - start) * 1000
        self.frame += 1

    def snapshot(self) -> dict:
        np = self.np
        with self.wp.ScopedDevice(self.device):
            positions = self.state.particle_q.numpy()
            velocities = self.state.particle_qd.numpy()
            if not (np.isfinite(positions).all() and np.isfinite(velocities).all()):
                raise RuntimeError("non-finite cloth state; reset or change scene parameters")
            sphere_pos = [0.0, 0.0, self.scene["sphere"]["height"]]
            quaternion = [0.0, 0.0, 0.0, 1.0]
            if self.sphere_body is not None:
                poses, velocities = self.state.body_q.numpy(), self.state.body_qd.numpy()
                if not (np.isfinite(poses).all() and np.isfinite(velocities).all()):
                    raise RuntimeError("non-finite rigid state; reset or change scene parameters")
                sphere_pos = poses[self.sphere_body, :3].tolist()
                quaternion = poses[self.sphere_body, 3:].tolist()
        return {"time": self.frame * FRAME_DT, "frame": self.frame, "device": self.device_label,
                "compute_ms": self.compute_ms, "scene": copy.deepcopy(self.scene),
                "geometry": {"positions": positions.reshape(-1).tolist(), "triangles": self.triangles,
                             "sphere": {"position": sphere_pos, "quaternion": quaternion,
                                        "radius": self.scene["sphere"]["radius"]}},
                "stats": {"particles": self.model.particle_count, "triangles": len(self.triangles) // 3}}


@dataclass
class _Request:
    payload: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    status: int = 200


class SimulationWorker:
    """Serialize CUDA ownership, commands, and immutable published snapshots."""

    def __init__(self, device: str = "cuda:0", cache_dir: Path | None = None, *, factory=NewtonSimulation):
        self._device, self._cache_dir, self._factory = device, cache_dir, factory
        self._lock = threading.Lock()
        self._commands = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._simulation = None
        self._running = False
        self._revision = 0
        self._scene = validate_scene({})
        self._state = {"ok": False, "ready": False, "running": False, "time": 0.0, "frame": 0,
                       "revision": 0, "error": None, "device": "initializing", "compute_ms": 0.0,
                       "scene": copy.deepcopy(self._scene), "geometry": {"positions": [], "triangles": [],
                       "sphere": {"position": [0, 0, self._scene["sphere"]["height"]],
                                  "quaternion": [0, 0, 0, 1], "radius": self._scene["sphere"]["radius"]}},
                       "stats": {"particles": 0, "triangles": 0}}
        self._thread = threading.Thread(target=self._run, name="newton-cuda-owner", daemon=True)
        self._thread.start()

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def health(self) -> dict:
        state = self.snapshot()
        return {"service": SERVICE, "ok": state["ok"] and state["ready"], "ready": state["ready"],
                "running": state["running"], "error": state["error"], "device": state["device"],
                "revision": state["revision"]}

    def submit(self, payload: dict) -> tuple[int, dict]:
        try:
            validate_command(payload)
        except CommandError as exc:
            return 400, {"ok": False, "error": str(exc)}
        if self._stop.is_set():
            return 503, {"ok": False, "error": "workbench is shutting down"}
        request = _Request(copy.deepcopy(payload))
        try:
            self._commands.put_nowait(request)
        except queue.Full:
            return 429, {"ok": False, "error": "command queue is full; retry after polling state"}
        while not request.done.wait(0.25):
            if not self._thread.is_alive():
                return 503, {"ok": False, "error": "simulation worker stopped"}
        return request.status, request.result

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _publish(self, snapshot: dict) -> None:
        value = dict(snapshot, ok=True, ready=True, running=self._running,
                     revision=self._revision, error=None)
        with self._lock:
            self._state = value

    def _failure(self, exc: Exception) -> None:
        self._running = False
        with self._lock:
            # Retain the last finite geometry and its time/frame. A reset builds
            # a fresh solver and state rather than reusing a damaged trajectory.
            self._state = dict(self._state, ok=False, running=False, error=f"{type(exc).__name__}: {exc}")

    def _rebuild(self, scene: dict) -> None:
        # Construct and validate first; a rejected/rebuild-failed configuration
        # leaves the last published scene and geometry intact.
        replacement = self._factory(scene, self._device, self._cache_dir)
        snapshot = replacement.snapshot()
        self._simulation, self._scene = replacement, scene
        self._running = False
        self._revision += 1
        self._publish(snapshot)

    def _advance(self) -> None:
        self._simulation.advance()
        self._publish(self._simulation.snapshot())

    def _execute(self, request: _Request) -> None:
        try:
            action, patch = validate_command(request.payload)
            if action in {"reset", "configure"}:
                scene = validate_scene(patch or {}, self._scene)
                self._rebuild(scene)
            else:
                if self._simulation is None or self.snapshot()["error"]:
                    raise CommandError("simulation is not ready; reset or configure to recover")
                if action == "step":
                    self._running = False
                    self._advance()
                else:
                    self._running = action == "play"
                    self._publish(self._simulation.snapshot())
            request.result = self.snapshot()
        except CommandError as exc:
            request.status, request.result = 400, {"ok": False, "error": str(exc)}
        except Exception as exc:
            self._failure(exc)
            request.status, request.result = 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            request.done.set()

    def _run(self) -> None:
        try:
            try:
                self._rebuild(self._scene)
            except Exception as exc:
                self._failure(exc)
            next_frame = time.monotonic()
            while not self._stop.is_set():
                delay = max(0.0, next_frame - time.monotonic()) if self._running else 0.1
                try:
                    request = self._commands.get(timeout=delay)
                except queue.Empty:
                    if self._running:
                        try:
                            self._advance()
                        except Exception as exc:
                            self._failure(exc)
                        # Target 60 simulated fps, without catch-up bursts when
                        # computation is slower than real time.
                        next_frame = max(next_frame + FRAME_DT, time.monotonic())
                    continue
                self._execute(request)
                if not self._running:
                    next_frame = time.monotonic()
        finally:
            while True:
                try:
                    request = self._commands.get_nowait()
                except queue.Empty:
                    break
                request.status, request.result = 503, {"ok": False, "error": "workbench stopped"}
                request.done.set()


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], worker: SimulationWorker):
        if address[0] != "127.0.0.1":
            raise ValueError("workbench RPC must bind to 127.0.0.1")
        self.worker = worker
        super().__init__(address, WorkbenchHandler)


class WorkbenchHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DeformableWorkbench/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        # Native clients poll frequently. Startup/errors are reflected through
        # structured state; do not flood the server log with successful polls.
        pass

    def _json(self, status: int, value: dict) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _origin_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        try:
            authority = urlsplit("http://" + host)
            if authority.hostname not in {"localhost", "127.0.0.1"} or authority.username or authority.password:
                return False
            if authority.path or authority.query or authority.fragment or not authority.port:
                return False
        except ValueError:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + host:
            return False
        if self.headers.get("Sec-Fetch-Site", "none") not in {"none", "same-origin"}:
            return False
        return True

    def do_GET(self):
        if not self._origin_allowed():
            self._json(403, {"ok": False, "error": "only localhost requests are accepted"})
            return
        path = urlsplit(self.path).path
        if path == "/api/state":
            self._json(200, self.server.worker.snapshot())
        elif path == "/api/health":
            self._json(200, self.server.worker.health())
        else:
            self._json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        # Always close mutation connections, including rejected bodies. This
        # prevents unread oversized/malformed data becoming a second request.
        self.close_connection = True
        if not self._origin_allowed():
            self._json(403, {"ok": False, "error": "cross-origin mutations are forbidden"})
            return
        if urlsplit(self.path).path != "/api/command":
            self._json(404, {"ok": False, "error": "unknown endpoint"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._json(415, {"ok": False, "error": "Content-Type must be application/json"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
            self._json(400, {"ok": False, "error": "one Content-Length header is required"})
            return
        length = int(lengths[0])
        if not 0 < length <= MAX_BODY_BYTES:
            self._json(413, {"ok": False, "error": f"body must be between 1 and {MAX_BODY_BYTES} bytes"})
            return
        try:
            data = self.rfile.read(length)
            if len(data) != length:
                raise CommandError("incomplete JSON body")
            payload = parse_json(data)
        except (CommandError, TimeoutError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        status, result = self.server.worker.submit(payload)
        self._json(status, result)

    def do_OPTIONS(self):
        self._json(405, {"ok": False, "error": "cross-origin access is not enabled"})
