#!/usr/bin/env python3
"""First official Isaac Sim 6.1.0 smoke; this is not full platform acceptance.

Run with the BUILT ISAAC SIM python.sh, never the standalone Newton virtualenv::

    ./python.sh /project/scripts/isaac_platform_smoke.py --mode run \
        --build-receipt /project/results/isaac-build-receipt.json \
        --source-root /project/upstream/IsaacSim --output-dir /project/results/isaac/run
    ./python.sh /project/scripts/isaac_platform_smoke.py --mode reload \
        --build-receipt /project/results/isaac-build-receipt.json \
        --source-root /project/upstream/IsaacSim --scene /project/results/isaac/run/scene.usda \
        --output-dir /project/results/isaac/reload

The second command must be a new process. The build owner first uses the separate
runtime inventory collector to observe Kit's version and generate a successful
build receipt with a nonempty kit_version. No robot/cloud asset is required. CPU
PhysX is the default, with real RTX camera rendering; --physics-device cuda:0
explicitly requests GPU PhysX and checks the resulting device and scene flags.
No CPU or Newton fallback is accepted. --experience can select a built .kit app.

API references inspected at v6.1.0, commit below:
  standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py
  standalone_examples/api/isaacsim.core.experimental.api/simulation_callbacks.py
  standalone_examples/api/isaacsim.sensors.experimental.rtx/create_camera_basic.py
  extensions/isaacsim.core.simulation_manager/python/impl/{simulation_manager,physx_scene}.py
  extensions/isaacsim.core.rendering_manager/python/impl/rendering_manager.py
  extensions/isaacsim.core.experimental.prims/python/impl/rigid_prim.py

Thresholds detect gross smoke failures; they are not claims of physical accuracy.
Physics timing includes Python, contact queries and synchronous host readbacks.
Rendering timing includes rendering and array readback, not isolated GPU kernels.
Output JSON uses exit_code 0 only when every required check for the selected mode
passes. Missing modules/assets/annotators are failures. Reload is never credited
by the run mode. No GUI, robot, ROS, lidar, deformable or full-platform pass is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time
import traceback
import uuid
import zlib

EXPECTED_COMMIT = "7c206f75bdadd9e05fc457f19863ca4c3f0cb693"
PROTOCOL = "isaac-platform-smoke-v1"
SPHERE = "/World/Sphere"
FLOOR = "/World/Floor"
CAMERA = "/World/Camera"
PHYSICS = "/World/PhysicsScene"


def runtime_binding(source_root: Path, isaac_path: str | None, executable: str, experience: str) -> dict:
    """Bind the process to this checkout's Linux build without importing Kit."""
    source = source_root.resolve()
    build = source / "_build/linux-x86_64/release"
    actual = Path(isaac_path).absolute() if isaac_path else None
    requested = Path(experience).absolute() if experience else build / "apps/isaacsim.exp.full.kit"
    checks = {
        "isaac_path_matches_checkout_build": actual is not None and actual.resolve() == build.resolve(),
        "official_launcher_exists": (build / "python.sh").is_file(),
        "bundled_python_exists": (build / "kit/python/bin/python3").is_file(),
        "bundled_python_is_executing": Path(executable).resolve() == (build / "kit/python/bin/python3").resolve(),
        "experience_from_build_apps": requested.parent.resolve() == (build / "apps").resolve(),
        "experience_exists": requested.is_file(),
    }
    source_launcher = source / "source/scripts/python/linux-x86_64/python.sh"
    source_experience = source / "source/apps" / requested.name
    checks["launcher_matches_pinned_source"] = bool(source_launcher.is_file() and (build / "python.sh").is_file()
        and sha256(source_launcher) == sha256(build / "python.sh"))
    checks["experience_matches_pinned_source"] = bool(source_experience.is_file() and requested.is_file()
        and sha256(source_experience) == sha256(requested))
    return {"passed": all(checks.values()), "checks": checks, "build_root": str(build),
            "isaac_path": str(actual) if actual else None, "python_executable": str(Path(executable).resolve()),
            "experience": str(requested), "experience_sha256": sha256(requested) if requested.is_file() else None,
            "scope": "checkout/build path and launcher/experience identity; not a reproducible-binary-build attestation"}


def verify_build_receipt(path: Path | None, source_root: Path, binding: dict) -> dict:
    """Require the build supervisor's successful-build record and current hashes.

    This function never generates a receipt from the presence of a build folder.
    Receipt schema is documented in its error/schema result for the build owner.
    """
    if path is None:
        return {"passed": False, "reason": "--build-receipt is required; build-directory location is not compilation provenance"}
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("build receipt must be a JSON object")
    source = receipt.get("source", {})
    build = receipt.get("build", {})
    runtime = Path(binding["build_root"]).resolve()
    checks = {
        "schema": receipt.get("schema_version") == 1 and receipt.get("kind") == "isaac_build_receipt",
        "source_commit": source.get("commit") == EXPECTED_COMMIT,
        "source_clean": source.get("dirty") is False,
        "source_root": isinstance(source.get("root"), str) and Path(source["root"]).resolve() == source_root.resolve(),
        "runtime_root": isinstance(receipt.get("runtime_root"), str) and Path(receipt["runtime_root"]).resolve() == runtime,
        "successful_build_exit": type(build.get("exit_code")) is int and build["exit_code"] == 0,
        "command_recorded": isinstance(build.get("command"), list) and bool(build["command"]) and all(isinstance(v, str) for v in build["command"]),
        "flags_recorded": isinstance(build.get("flags"), list) and all(isinstance(v, str) for v in build["flags"]),
        "completion_recorded": isinstance(build.get("completed_at_utc"), str) and bool(build["completed_at_utc"]),
        "build_log_hash_recorded": isinstance(build.get("log_sha256"), str) and len(build["log_sha256"]) == 64 and all(c in "0123456789abcdef" for c in build["log_sha256"]),
        "kit_version_recorded": isinstance(receipt.get("kit_version"), str) and bool(receipt["kit_version"]),
    }
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("build receipt requires a nonempty artifacts list")
    observed = []
    for artifact in artifacts:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("receipt artifact paths must be relative to runtime_root without parent traversal")
        target = runtime / relative
        exists = target.is_file()
        actual_hash = sha256(target) if exists else None
        observed.append({"role": artifact["role"], "path": relative.as_posix(), "exists": exists,
                         "expected_sha256": artifact["sha256"], "actual_sha256": actual_hash,
                         "matches": exists and actual_hash == artifact["sha256"]})
    roles = {item["role"] for item in observed if item["matches"]}
    checks["required_artifact_roles"] = {"launcher", "experience", "python_executable", "kit_executable", "isaac_native_plugin"} <= roles
    checks["all_artifact_hashes_match"] = all(item["matches"] for item in observed)
    checks["selected_experience_covered"] = any(item["role"] == "experience" and item["matches"]
        and (runtime / item["path"]).resolve() == Path(binding["experience"]).resolve() for item in observed)
    checks["executing_python_covered"] = any(item["role"] == "python_executable" and item["matches"]
        and (runtime / item["path"]).resolve() == Path(binding["python_executable"]).resolve() for item in observed)
    for role, known_path in (("launcher", runtime / "python.sh"), ("kit_executable", runtime / "kit/kit")):
        checks[role + "_identity"] = any(item["role"] == role and item["matches"]
            and (runtime / item["path"]).resolve() == known_path.resolve() for item in observed)
    return {"passed": all(checks.values()), "checks": checks, "path": str(path.resolve()),
            "receipt_sha256": sha256(path), "kit_version_expected": receipt.get("kit_version"),
            "kit_version_status": "recorded_pending_runtime_comparison" if checks["kit_version_recorded"] else "not_verified",
            "build_record": build, "artifacts": observed,
            "trust_scope": "build supervisor's recorded source/build relationship plus freshly verified artifact hashes; not independent reproducible-build attestation"}


def compare_frozen(expected, actual, path="") -> list[str]:
    """Compare a JSON snapshot, with only small USD numeric serialization tolerance."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = [f"{path}/{key}: missing" for key in expected.keys() - actual.keys()]
        differences += [f"{path}/{key}: unexpected" for key in actual.keys() - expected.keys()]
        for key in expected.keys() & actual.keys():
            differences.extend(compare_frozen(expected[key], actual[key], f"{path}/{key}"))
        return sorted(differences)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length differs"]
        return [difference for index, (left, right) in enumerate(zip(expected, actual))
                for difference in compare_frozen(left, right, f"{path}/{index}")]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return [] if math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected, actual, abs_tol=1e-7, rel_tol=1e-7) else [f"{path}: numeric value differs"]
    return [] if type(expected) is type(actual) and expected == actual else [f"{path}: value/type differs"]


def authored_snapshot(stage, UsdGeom) -> dict:
    """Capture authored defaults, connections, bindings and schemas below /World.

    Called before playback/export, and immediately after opening USD, before
    wrappers/settings can author anything. This deliberately includes mass,
    gravity, solver/device flags, complete transform ops, camera optics, floor
    geometry, light and materials; it does not infer them from CLI defaults.
    """
    def value(item):
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        if hasattr(item, "GetReal") and hasattr(item, "GetImaginary"):
            return [float(item.GetReal()), *[float(v) for v in item.GetImaginary()]]
        try:
            return [value(v) for v in item]
        except TypeError:
            return str(item)
    prims = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path != "/World" and not path.startswith("/World/"):
            continue
        attributes = {}
        for attr in prim.GetAttributes():
            if attr.HasAuthoredValueOpinion() or attr.GetConnections():
                attributes[attr.GetName()] = {"type": str(attr.GetTypeName()), "value": value(attr.Get()),
                    "connections": [str(p) for p in attr.GetConnections()],
                    "time_samples": [[t, value(attr.Get(t))] for t in attr.GetTimeSamples()]}
        prims[path] = {"type": prim.GetTypeName(), "schemas": sorted(prim.GetAppliedSchemas()),
                      "attributes": attributes,
                      "relationships": {rel.GetName(): [str(p) for p in rel.GetTargets()] for rel in prim.GetRelationships() if rel.HasAuthoredTargets()}}
    return {"prims": prims, "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "time_codes_per_second": stage.GetTimeCodesPerSecond(),
            "default_prim": str(stage.GetDefaultPrim().GetPath())}


def assess_depth_alignment(rgb, depth, height: int, width: int, radius: float = 0.2,
                           focal_pixels: float | None = None) -> dict:
    """Check depth at the red sphere, rejecting previous-pose depth buffers.

    Frozen camera: (0,-4,1), optical axis world +Y; sphere center world Y=0.
    distance_to_image_plane therefore sees its visible front surface near
    4-radius metres. The final-pose rays hit farther floor in an initial-pose
    depth frame, so a stale initial depth cannot pass with updated final RGB.
    """
    import numpy as np
    if rgb is None or depth is None:
        return {"passed": False, "reason": "RGB/depth pair absent"}
    if rgb.ndim != 3 or rgb.shape[:2] != (height, width) or rgb.shape[2] not in (3, 4) or depth.shape not in ((height, width), (height, width, 1)):
        return {"passed": False, "reason": "RGB/depth shape mismatch"}
    rgb = rgb[:, :, :3].astype(np.float32)
    depth = depth.reshape(height, width)
    if np.isnan(depth).any() or np.isneginf(depth).any():
        return {"passed": False, "reason": "NaN or negative infinity in depth; only positive infinity is allowed for background"}
    red = (rgb[:, :, 0] > 30) & (rgb[:, :, 0] > 1.5 * rgb[:, :, 1]) & (rgb[:, :, 0] > 1.5 * rgb[:, :, 2])
    rows, columns = np.nonzero(red)
    if rows.size < 10:
        return {"passed": False, "reason": "red sphere not sufficiently visible", "red_pixels": int(rows.size)}
    center_y, center_x = float(rows.mean()), float(columns.mean())
    roi_radius = max(1.0, 0.5 * math.sqrt(rows.size / math.pi))
    interior = ((rows - center_y) ** 2 + (columns - center_x) ** 2) <= roi_radius ** 2
    values = depth[rows[interior], columns[interior]]
    finite = values[np.isfinite(values)]
    low, high = 4.0 - radius - 0.05, 4.0 - 0.1 * radius
    median = float(np.median(finite)) if finite.size else None
    ratio = float(np.mean(np.isfinite(values) & (values >= low) & (values <= high))) if values.size else 0.0
    # Independent static-floor geometry rejects a spatially constant depth
    # image even when its constant happens to equal the sphere's front surface.
    # World point (1,1,0) is visible beside the sphere at optical depth 5 m.
    focal = float(focal_pixels if focal_pixels is not None else width * 24.0 / 20.955)
    floor_x, floor_y = round(width / 2 + focal / 5), round(height / 2 + focal / 5)
    floor_ok = False
    floor_error = None
    if 2 <= floor_x < width - 2 and 2 <= floor_y < height - 2:
        floor = depth[floor_y - 2:floor_y + 3, floor_x - 2:floor_x + 3]
        # The ray from the fixed camera's height 1 m meets z=0 at this
        # image-plane distance. Half-pixel convention is covered by 0.15 m.
        expected = focal / (np.arange(floor_y - 2, floor_y + 3)[:, None] + 0.5 - height / 2)
        if np.isfinite(floor).all():
            floor_error = float(np.max(np.abs(floor - expected)))
            floor_ok = floor_error <= 0.15
    return {"passed": bool(values.size >= 5 and ratio >= 0.9 and median is not None and low <= median <= high and floor_ok),
            "red_pixels": int(rows.size), "interior_pixels": int(values.size), "in_expected_band_fraction": ratio,
            "interior_median_depth_m": median, "expected_depth_band_m": [low, high],
            "red_centroid_xy": [center_x, center_y],
            "floor_roi_center_xy": [floor_x, floor_y], "floor_maximum_depth_error_m": floor_error,
            "floor_depth_consistent": floor_ok, "focal_length_pixels": focal,
            "association": "same capture iteration plus scene geometry; annotator frame IDs may be unavailable"}


def annotator_frame_info(info: dict) -> dict:
    allowed = {"frame", "frameid", "framenumber", "renderingframe", "renderingframenumber", "timestamp", "time", "simulationtime", "simtime"}
    result = {}
    for key, value in info.items():
        normalized = str(key).lower().replace("_", "").replace("-", "")
        if normalized in allowed and isinstance(value, (str, int, float, bool)):
            if not isinstance(value, float) or math.isfinite(value):
                result[str(key)] = value
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rgb_png(path: Path, rgb) -> None:
    """Lossless RGB PNG without Pillow/matplotlib or a GUI dependency."""
    height, width, channels = rgb.shape
    if channels != 3 or str(rgb.dtype) != "uint8":
        raise ValueError("PNG input must be uint8 HxWx3")
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\0" + rgb[y].tobytes() for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def assess_physics(samples: list[dict], config: dict) -> dict:
    """Pure, conservative assessment; no caller-supplied 'finite' flag is trusted."""
    if len(samples) < 3:
        return {"passed": False, "reason": "insufficient trajectory samples"}
    fields = ("position", "orientation_wxyz", "linear_velocity", "angular_velocity", "contact_force")
    try:
        finite = all(math.isfinite(float(v)) for s in samples for key in fields for v in s[key])
        finite = finite and all(math.isfinite(float(s["time"])) for s in samples)
    except (KeyError, TypeError, ValueError):
        return {"passed": False, "reason": "invalid trajectory record"}
    if not finite:
        return {"passed": False, "reason": "nonfinite physics state or contact data"}
    tail = samples[-min(config["tail_steps"], len(samples) - 1):]
    radius = config["radius"]
    gap = config["height"] - radius
    norm = lambda values: math.sqrt(sum(float(v) ** 2 for v in values))
    metrics = {
        "sample_count": len(samples),
        "drop_m": samples[0]["position"][2] - min(s["position"][2] for s in samples),
        "minimum_vertical_velocity_m_s": min(s["linear_velocity"][2] for s in samples),
        "minimum_bottom_z_m": min(s["position"][2] - radius for s in samples),
        "tail_maximum_height_error_m": max(abs(s["position"][2] - radius) for s in tail),
        "tail_maximum_linear_speed_m_s": max(norm(s["linear_velocity"]) for s in tail),
        "tail_maximum_angular_speed_rad_s": max(norm(s["angular_velocity"]) for s in tail),
        "tail_position_span_m": max(max(s["position"][i] for s in tail) - min(s["position"][i] for s in tail) for i in range(3)),
        "peak_upward_contact_force_n": max(s["contact_force"][2] for s in samples),
        "time_advance_s": samples[-1]["time"] - samples[0]["time"],
        "step_advance": samples[-1]["step"] - samples[0]["step"],
    }
    checks = {
        "falls": metrics["drop_m"] >= gap * 0.5 and metrics["minimum_vertical_velocity_m_s"] < -0.1,
        "ground_not_grossly_penetrated": metrics["minimum_bottom_z_m"] >= -config["penetration_tolerance_m"],
        "settles_near_ground": metrics["tail_maximum_height_error_m"] <= config["settle_height_tolerance_m"],
        "settles_velocity": metrics["tail_maximum_linear_speed_m_s"] <= config["settle_speed_tolerance_m_s"],
        "settles_position": metrics["tail_position_span_m"] <= config["settle_position_span_m"],
        "measured_ground_contact": metrics["peak_upward_contact_force_n"] > 0.1 * config["mass"] * config["gravity"],
        "physics_steps_advance": metrics["step_advance"] >= config["steps"],
        "simulation_time_advances": metrics["time_advance_s"] >= 0.99 * config["steps"] / config["physics_hz"],
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics,
            "tolerance_scope": "gross numerical smoke only, not calibrated real-world accuracy"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("run", "reload"), default="run")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1] / "upstream/IsaacSim")
    parser.add_argument("--build-receipt", type=Path, help="Successful official build receipt; absence is a failed provenance check, not a fallback")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", type=Path, help="JSON report path; defaults to OUTPUT_DIR/report.json")
    parser.add_argument("--scene", type=Path, help="reload: previously exported scene.usda; run: optional destination")
    parser.add_argument("--experience", default="", help="Built .kit experience; default is this checkout's built apps/isaacsim.exp.full.kit")
    parser.add_argument("--physics-device", default="cpu", help="cpu or cuda:N; reload uses the frozen manifest value")
    parser.add_argument("--renderer", choices=("RaytracedLighting", "RealTimePathTracing", "PathTracing"), default="RaytracedLighting")
    parser.add_argument("--steps", type=int, default=240, help="Explicit physics steps; reload uses frozen manifest value")
    parser.add_argument("--physics-hz", type=int, default=60)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240, help="Image height, not sphere height")
    parser.add_argument("--render-warmup", type=int, default=20)
    parser.add_argument("--render-max-frames", type=int, default=120)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error("Unrecognized script arguments: " + " ".join(unknown))
    if args.mode == "reload" and args.scene is None:
        parser.error("--mode reload requires --scene from a completed run-mode export")
    if args.steps < 60 or not 20 <= args.physics_hz <= 240:
        parser.error("--steps must be >=60 and --physics-hz must be in [20,240]")
    if min(args.width, args.height) < 64 or max(args.width, args.height) > 2048:
        parser.error("image dimensions must be in [64,2048]")
    if not (args.physics_device == "cpu" or (args.physics_device.startswith("cuda:") and args.physics_device[5:].isdigit())):
        parser.error("--physics-device must be cpu or cuda:N")
    if args.render_warmup < 1 or args.render_max_frames < args.render_warmup:
        parser.error("render frame bounds are invalid")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    # SimulationApp forwards unknown process argv to Kit. Our script options
    # belong only to argparse, and must not become Kit settings/arguments.
    sys.argv = [sys.argv[0]]
    output = (args.output_dir or Path(__file__).resolve().parents[1] / "results/isaac-platform-smoke" / args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = (args.json or output / "report.json").resolve()
    scene_path = (args.scene or output / "scene.usda").resolve()
    manifest_path = scene_path.with_suffix(".manifest.json")
    run_id = str(uuid.uuid4())
    required = ["source_identity", "runtime_binding", "build_receipt", "official_runtime", "kit_version_binding"]
    required += ["scene_authoring" if args.mode == "run" else "clean_process_reload",
                 "scene_export" if args.mode == "run" else "scene_integrity", "physics", "rgb", "depth", "render_motion"]
    report = {
        "schema_version": 1, "protocol_id": PROTOCOL, "mode": args.mode, "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "status": "running", "exit_code": None, "full_platform_passed": False,
        "clean_process_reload_verified": False, "source_commit_expected": EXPECTED_COMMIT,
        "script_sha256": sha256(Path(__file__)), "python": sys.version, "platform": platform.platform(),
        "configuration_requested": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checks": {key: {"status": "not_run"} for key in required}, "artifacts": {},
        "limitations": ["No GUI/robot/ROS/lidar/deformable acceptance; no full-platform parity credit.",
                        "CPU PhysX is explicit by default; RTX rendering is a separate GPU workload.",
                        "Scene export is the authored initial scene, not a Fabric-state snapshot.",
                        "PNG is a preview; .npy files contain actual sensor arrays, including background depth infinities."],
    }
    config = {"radius": 0.2, "height": 1.3, "mass": 1.0, "gravity": 9.81,
              "physics_hz": args.physics_hz, "steps": args.steps, "tail_steps": 30,
              "physics_device": args.physics_device, "width": args.width, "image_height": args.height,
              "penetration_tolerance_m": 0.02, "settle_height_tolerance_m": 0.025,
              "settle_speed_tolerance_m_s": 0.10, "settle_position_span_m": 0.02}
    app = None
    app_utils = None
    started = time.perf_counter()
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True, exit=False)

    def check(name: str, passed: bool, **details):
        report["checks"][name] = {"status": "passed" if passed else "failed", **details}
        write_json(report_path, report)

    def fail(name: str, error: Exception):
        check(name, False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())

    try:
        try:
            git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
            commit = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True, timeout=10, env=git_env).strip()
            dirty = subprocess.check_output(["git", "-C", str(args.source_root), "status", "--porcelain", "--untracked-files=normal"], text=True, timeout=10, env=git_env).strip()
            check("source_identity", commit == EXPECTED_COMMIT and not dirty, actual_commit=commit,
                  source_dirty_including_untracked=bool(dirty), source_root=str(args.source_root.resolve()))
        except Exception as error:
            fail("source_identity", error)
        if report["checks"]["source_identity"]["status"] != "passed":
            raise RuntimeError("The source checkout is not the clean pinned commit; runtime launch refused")
        binding = runtime_binding(args.source_root, os.environ.get("ISAAC_PATH"), sys.executable, args.experience)
        check("runtime_binding", binding.pop("passed"), **binding)
        if report["checks"]["runtime_binding"]["status"] != "passed":
            raise RuntimeError("Use this checkout's built python.sh and experience, without PYTHONEXE override")
        receipt_result = verify_build_receipt(args.build_receipt, args.source_root, binding)
        check("build_receipt", receipt_result.pop("passed"), **receipt_result)
        if report["checks"]["build_receipt"]["status"] != "passed":
            raise RuntimeError("A matching successful-build receipt is required before launching Kit")

        manifest = None
        if args.mode == "reload":
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_id") != PROTOCOL or manifest.get("source_commit") != EXPECTED_COMMIT:
                raise ValueError("Scene manifest protocol/source mismatch")
            config = manifest["scene_config"]
            if manifest.get("creator_pid") == os.getpid():
                raise ValueError("Reload must be invoked in a separate process")
            if sha256(scene_path) != manifest["scene_sha256"]:
                raise ValueError("Saved scene SHA256 mismatch")
            if manifest.get("build_receipt_sha256") != receipt_result["receipt_sha256"]:
                raise ValueError("Reload must use the same verified build receipt as scene creation")
        report["configuration_effective"] = config
        launch = {"headless": True, "renderer": args.renderer, "multi_gpu": False,
                  "active_gpu": 0, "physics_gpu": 0, "width": config["width"], "height": config["image_height"],
                  "anti_aliasing": 0, "fast_shutdown": True, "sync_loads": True,
                  "extra_args": ["--/exts/isaacsim.core.simulation_manager/default_engine=physx",
                                 "--/app/extensions/pythonFullLeakCheck=false"]}
        report["launch_config"] = launch
        write_json(report_path, report)

        # Never import Omniverse modules before starting the official runtime.
        from isaacsim import SimulationApp
        app = SimulationApp(launch, experience=binding["experience"])
        import numpy as np
        import carb.settings
        import omni.kit.app
        import omni.usd
        import warp as wp
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
        from isaacsim.core.experimental.objects import Cube, Sphere, DistantLight
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
        from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.version import get_version
        from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
        from pxr import UsdGeom, UsdPhysics, PhysxSchema

        settings = carb.settings.get_settings()
        version = list(get_version())
        engine = SimulationManager.get_active_physics_engine()
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        versions = {}
        for name in ("omni.physx", "isaacsim.simulation_app", "isaacsim.core.simulation_manager", "isaacsim.sensors.experimental.rtx"):
            ext_id = extension_manager.get_enabled_extension_id(name)
            versions[name] = ext_id
        report["runtime"] = {"isaac_version_tuple": version, "extension_ids": versions,
                             "warp_version": wp.__version__, "numpy_version": np.__version__,
                             "isaac_path": os.environ.get("ISAAC_PATH"), "app_folder": settings.get("/app/folder"),
                             "active_physics_engine": engine, "renderer_requested": args.renderer,
                             "renderer_setting": settings.get("/rtx/rendermode"),
                             "fast_shutdown_observed": settings.get("/app/fastShutdown")}
        module_path = Path(sys.modules[SimulationApp.__module__].__file__).resolve()
        pinned_module = args.source_root.resolve() / "source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py"
        observed_experience = omni.kit.app.get_app().get_app_filename()
        observed_kit_version = omni.kit.app.get_app().get_kit_version()
        requested_experience = Path(binding["experience"])
        experience_matches = (Path(observed_experience).name == requested_experience.name
                              or observed_experience == requested_experience.stem)
        if Path(observed_experience).is_absolute():
            experience_matches = experience_matches and Path(observed_experience).resolve() == requested_experience.resolve()
        module_matches = module_path.is_file() and pinned_module.is_file() and sha256(module_path) == sha256(pinned_module)
        report["runtime"].update(simulation_app_module=str(module_path), simulation_app_matches_pinned_source=module_matches,
                                 observed_experience=observed_experience, experience_matches_requested=experience_matches,
                                 kit_version=observed_kit_version, kit_version_matches_build_receipt=observed_kit_version == receipt_result["kit_version_expected"])
        runtime_ok = (version[0] == "6.1.0" and engine == "physx" and bool(versions["omni.physx"])
                      and module_matches and experience_matches and settings.get("/app/fastShutdown") is True)
        check("official_runtime", runtime_ok, **report["runtime"])
        if not runtime_ok:
            raise RuntimeError("Observed runtime does not match the pinned build, experience, version or PhysX engine")
        version_matches = observed_kit_version == receipt_result["kit_version_expected"]
        check("kit_version_binding", version_matches, expected=receipt_result["kit_version_expected"], observed=observed_kit_version)
        if not version_matches:
            raise RuntimeError("Runtime Kit version differs from the successful-build receipt")
        if engine != "physx":
            raise RuntimeError(f"Expected PhysX, received {engine}; refusing another backend")

        if args.mode == "run":
            stage_utils.create_new_stage()
            stage = stage_utils.get_current_stage(backend="usd")
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            world = stage_utils.define_prim("/World", "Xform")
            stage.SetDefaultPrim(world)
            scene = PhysxScene(PHYSICS)
            scene.set_steps_per_second(config["physics_hz"])
            scene.set_gravity([0, 0, -config["gravity"]])
            scene.set_solver_type("TGS")
            floor = Cube(FLOOR, sizes=1.0, positions=[0, 0, -0.1], scales=[6.0, 6.0, 0.2])
            GeomPrim(FLOOR, apply_collision_apis=True)
            sphere = Sphere(SPHERE, radii=config["radius"], positions=[0, 0, config["height"]])
            GeomPrim(SPHERE, apply_collision_apis=True)
            body = RigidPrim(SPHERE, masses=config["mass"], contact_filter_paths=[FLOOR])
            # Tensor contact views require PhysX contact reporting on both shapes,
            # as in pinned test_rigid_prim.populate_contact_stage.
            for path in (SPHERE, FLOOR):
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
            for shape, name, color in ((sphere, "red", [0.85, 0.04, 0.02]), (floor, "floor", [0.18, 0.28, 0.38])):
                material = PreviewSurfaceMaterial("/World/Materials/" + name)
                material.set_input_values("diffuseColor", color)
                shape.apply_visual_materials(material)
            light = DistantLight("/World/Light")
            light.set_intensities(1500.0)
            camera = RtxCamera(CAMERA, translations=np.array([0.0, -4.0, 1.0]),
                               orientations=np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0))
            # This experimental wrapper multiplies focal lengths by ten.
            # Author USD optical values directly so the intended field of view
            # and our pixel/depth projection use the same explicit units.
            camera_schema = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA))
            camera_schema.GetFocalLengthAttr().Set(24.0)
            camera_schema.GetHorizontalApertureAttr().Set(20.955)
            camera_schema.GetVerticalApertureAttr().Set(20.955 * config["image_height"] / config["width"])
            camera.camera.set_clipping_ranges(0.01, 100.0)
            check("scene_authoring", True, prim_paths=[SPHERE, FLOOR, CAMERA, PHYSICS], cloud_assets_required=False)
        else:
            ok, stage = stage_utils.open_stage(str(scene_path))
            if not ok or stage is None:
                raise RuntimeError("USD context failed to open the saved scene")
            for _ in range(120):
                if not stage_utils.is_stage_loading():
                    break
                app.update()
            if stage_utils.is_stage_loading():
                raise RuntimeError("Saved scene did not finish loading within 120 updates")
            # Inspect before constructing ANY wrapper or setting the device/dt.
            # Otherwise constructors or setters could silently repair a bad save.
            reloaded_snapshot = authored_snapshot(stage, UsdGeom)
            differences = compare_frozen(manifest["authored_snapshot"], reloaded_snapshot)
            check("scene_integrity", not differences, scene_sha256=sha256(scene_path),
                  authored_snapshot_sha256=hashlib.sha256(json.dumps(reloaded_snapshot, sort_keys=True).encode()).hexdigest(),
                  comparison_scope="all authored /World prim defaults/time samples, schemas, bindings/connections and stage units/time rate",
                  differences=differences[:50], difference_count=len(differences))
            if differences:
                raise RuntimeError("Reloaded authored scene does not match its manifest")
            check("clean_process_reload", True, creator_pid=manifest["creator_pid"], loader_pid=os.getpid(),
                  creator_run_id=manifest["creator_run_id"], scene_path=str(scene_path))
            report["clean_process_reload_verified"] = True
            scene = PhysxScene(PHYSICS)
            body = RigidPrim(SPHERE, contact_filter_paths=[FLOOR])
            camera = RtxCamera(CAMERA, reset_xform_op_properties=False)

        # Author scene flags explicitly as well: set_device() iterates only
        # scenes already registered with SimulationManager's stage callbacks.
        gpu_requested = config["physics_device"].startswith("cuda:")
        if args.mode == "run":
            scene.set_enabled_gpu_dynamics(gpu_requested)
            scene.set_broadphase_type("GPU" if gpu_requested else "MBP")
        elif scene.get_enabled_gpu_dynamics() != gpu_requested or scene.get_broadphase_type() != ("GPU" if gpu_requested else "MBP"):
            raise RuntimeError("Frozen device configuration differs from reloaded scene; refusing to repair it")
        SimulationManager.set_device(config["physics_device"])
        RenderingManager.set_dt(1.0 / config["physics_hz"])
        if args.mode == "reload":
            differences = compare_frozen(reloaded_snapshot, authored_snapshot(stage, UsdGeom))
            if differences:
                check("scene_integrity", False, reason="Wrapping/configuring reloaded scene modified authored state", differences=differences[:50])
                report["clean_process_reload_verified"] = False
                raise RuntimeError("Reload setup changed frozen authored attributes")
        if args.mode == "run":
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            if not stage.Export(str(scene_path)):
                raise RuntimeError("USD Stage.Export returned false")
            write_json(manifest_path, {"protocol_id": PROTOCOL,
                       "source_commit": report["checks"]["source_identity"].get("actual_commit"),
                       "creator_pid": os.getpid(), "creator_run_id": run_id, "scene_config": config,
                       "build_receipt_sha256": receipt_result["receipt_sha256"],
                       "authored_snapshot": authored_snapshot(stage, UsdGeom),
                       "scene_sha256": sha256(scene_path), "renderer": args.renderer,
                       "saved_state": "authored initial scene, before physics playback"})
            report["artifacts"].update(scene=str(scene_path), scene_manifest=str(manifest_path))
            check("scene_export", True, path=str(scene_path), sha256=sha256(scene_path), bytes=scene_path.stat().st_size)

        sensor = CameraSensor(camera, resolution=(config["image_height"], config["width"]),
                              annotators=["rgb", "distance_to_image_plane"])
        camera_prim = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA))
        focal_pixels = config["width"] * float(camera_prim.GetFocalLengthAttr().Get()) / float(camera_prim.GetHorizontalApertureAttr().Get())
        report["camera_projection"] = {"focal_pixels": focal_pixels,
            "camera_position_world": [0, -4, 1], "optical_axis_world": "+Y", "image_up_world": "+Z",
            "depth_reference": "distance_to_image_plane, metres", "floor_roi_reference_point_world": [1, 1, 0]}
        app_utils.play()
        app.update()  # official example's initialization update; excluded from measured steps
        # Do not reset pose, mass or velocity: in reload mode that could conceal
        # a failed persistence round trip. Measure the actual initialized state.
        device = SimulationManager.get_device()
        device_ok = (str(device) == config["physics_device"] and scene.get_enabled_gpu_dynamics() == gpu_requested)
        report["runtime"].update(physics_device=str(device), gpu_dynamics=scene.get_enabled_gpu_dynamics(),
                                 broadphase=scene.get_broadphase_type(), physics_dt_s=scene.get_dt(),
                                 fabric_enabled=SimulationManager.is_fabric_enabled())
        if not device_ok:
            raise RuntimeError("Actual physics device/GPU dynamics differs from requested configuration")

        def capture(label: str):
            capture_start = time.perf_counter()
            rgb = depth = None
            devices = {}
            frame_info = {}
            alignment = {"passed": False, "reason": "no same-iteration RGB/depth pair"}
            for index in range(args.render_max_frames):
                RenderingManager.render()
                # Clear both candidates on EVERY iteration. A None payload must
                # never leave a previous frame's array eligible for acceptance.
                rgb = depth = None
                devices = {}
                rgb_data, rgb_info = sensor.get_data("rgb")
                depth_data, depth_info = sensor.get_data("distance_to_image_plane")
                frame_info = {"rgb": annotator_frame_info(rgb_info), "depth": annotator_frame_info(depth_info)}
                if rgb_data is None or depth_data is None:
                    alignment = {"passed": False, "reason": "RGB or depth missing in this capture iteration"}
                    continue
                rgb, depth = rgb_data.numpy().copy(), depth_data.numpy().copy()
                devices = {"rgb": str(rgb_data.device), "depth": str(depth_data.device)}
                alignment = assess_depth_alignment(rgb, depth, config["image_height"], config["width"], config["radius"], focal_pixels)
                common_frame_keys = [key for key in frame_info["rgb"] if key in frame_info["depth"]
                                     and "frame" in key.lower()]
                frame_identity_ok = all(frame_info["rgb"][key] == frame_info["depth"][key] for key in common_frame_keys)
                alignment["matching_available_frame_ids"] = frame_identity_ok
                alignment["frame_identity_status"] = "compared" if common_frame_keys else "unavailable_geometry_check_required"
                alignment["passed"] = alignment["passed"] and frame_identity_ok
                if index + 1 >= args.render_warmup and alignment["passed"]:
                    break
            report.setdefault("render_captures", {})[label] = {"render_updates": index + 1,
                "elapsed_seconds": time.perf_counter() - capture_start, "array_devices": devices,
                "annotator_frame_info": frame_info, "depth_alignment": alignment,
                "timing_scope": "app rendering updates plus synchronous sensor host readbacks"}
            for name, value in (("rgb", rgb), ("depth", depth)):
                if value is not None:
                    path = output / f"{label}_{name}.npy"
                    np.save(path, value, allow_pickle=False)
                    report["artifacts"][f"{label}_{name}"] = str(path)
            if rgb is not None and rgb.ndim == 3 and rgb.shape[2] >= 3 and rgb.dtype == np.uint8:
                path = output / f"{label}_rgb.png"
                write_rgb_png(path, np.ascontiguousarray(rgb[:, :, :3]))
                report["artifacts"][f"{label}_rgb_preview"] = str(path)
            return rgb, depth

        initial_rgb = initial_depth = final_rgb = final_depth = None
        try:
            initial_rgb, initial_depth = capture("initial")
        except Exception as error:
            report["initial_capture_error"] = f"{type(error).__name__}: {error}"

        trajectory = []
        try:
            def sample():
                with use_backend("tensor", raise_on_fallback=True, raise_on_unsupported=True):
                    positions, orientations = body.get_world_poses()
                    linear, angular = body.get_velocities()
                    forces = body.get_net_contact_forces(dt=scene.get_dt())
                report["runtime"].setdefault("physics_query_array_devices", {
                    "positions": str(positions.device), "velocities": str(linear.device),
                    "contact_forces": str(forces.device)})
                return {"step": int(SimulationManager.get_num_physics_steps()),
                        "time": float(SimulationManager.get_simulation_time()),
                        "position": positions.numpy()[0].tolist(), "orientation_wxyz": orientations.numpy()[0].tolist(),
                        "linear_velocity": linear.numpy()[0].tolist(), "angular_velocity": angular.numpy()[0].tolist(),
                        "contact_force": forces.numpy()[0].tolist()}
            trajectory.append(sample())
            physics_start = time.perf_counter()
            for _ in range(config["steps"]):
                SimulationManager.step(update_fabric=SimulationManager.is_fabric_enabled())
                trajectory.append(sample())
            assessment = assess_physics(trajectory, config)
            check("physics", assessment.pop("passed"), **assessment,
                  elapsed_seconds=time.perf_counter() - physics_start,
                  timing_scope="physics stepping, Python, contact queries, synchronous host readbacks; excludes startup and rendering")
        except Exception as error:
            fail("physics", error)
        # Preserve invalid trajectories as strict JSON with explicit null, not NaN JSON.
        sanitized = [{k: ([v if math.isfinite(float(v)) else None for v in value] if isinstance(value, list)
                         else value if not isinstance(value, float) or math.isfinite(value) else None)
                      for k, value in row.items()} for row in trajectory]
        trajectory_path = output / "trajectory.json"
        write_json(trajectory_path, {"samples": sanitized, "invalid_numeric_values_replaced_by_null": sanitized != trajectory})
        report["artifacts"]["trajectory"] = str(trajectory_path)
        try:
            final_rgb, final_depth = capture("final")
        except Exception as error:
            report["final_capture_error"] = f"{type(error).__name__}: {error}"
        expected_hw = (config["image_height"], config["width"])
        if final_rgb is None:
            check("rgb", False, reason="RGB annotator did not return an array")
        else:
            valid_shape = final_rgb.ndim == 3 and final_rgb.shape[:2] == expected_hw and final_rgb.shape[2] in (3, 4)
            rgb_values = final_rgb[:, :, :3] if valid_shape else final_rgb
            std = float(np.std(rgb_values.astype(np.float64)))
            check("rgb", bool(valid_shape and final_rgb.dtype == np.uint8 and np.isfinite(final_rgb).all() and std > 1.0),
                  shape=list(final_rgb.shape), dtype=str(final_rgb.dtype), standard_deviation=std,
                  requirement="expected shape, uint8, finite and nonuniform (std > 1 digital level)")
        if final_depth is None:
            check("depth", False, reason="distance_to_image_plane annotator did not return an array")
        else:
            finite_positive = np.isfinite(final_depth) & (final_depth > 0.01) & (final_depth < 100.0)
            ratio = float(np.mean(finite_positive))
            values = final_depth[finite_positive]
            depth_shape_ok = final_depth.shape in (expected_hw, (*expected_hw, 1))
            phase_alignment = {label: report.get("render_captures", {}).get(label, {}).get("depth_alignment", {"passed": False})
                               for label in ("initial", "final")}
            check("depth", bool(depth_shape_ok and np.issubdtype(final_depth.dtype, np.floating)
                                 and not np.isnan(final_depth).any() and not np.isneginf(final_depth).any() and ratio >= 0.01
                                 and initial_depth is not None and all(item["passed"] for item in phase_alignment.values())),
                  shape=list(final_depth.shape), dtype=str(final_depth.dtype), positive_finite_fraction=ratio,
                  phase_alignment=phase_alignment,
                  valid_min_m=float(values.min()) if values.size else None,
                  valid_max_m=float(values.max()) if values.size else None,
                  background_nonfinite_count=int((~np.isfinite(final_depth)).sum()),
                  requirement="correct shape; no NaN/-inf; >=1% positive finite depth; initial AND final sphere/floor depth match scene geometry; same-iteration paired arrays and available frame IDs agree; background +inf permitted")
        if initial_rgb is None or final_rgb is None or initial_rgb.shape != final_rgb.shape:
            check("render_motion", False, reason="initial/final RGB pair missing or shapes differ")
        else:
            delta = np.abs(initial_rgb[:, :, :3].astype(np.int16) - final_rgb[:, :, :3].astype(np.int16))
            changed = int(np.count_nonzero(np.max(delta, axis=2) > 8))
            def red_centroid(rgb):
                color = rgb[:, :, :3].astype(np.float32)
                mask = (color[:, :, 0] > 30) & (color[:, :, 0] > 1.5 * color[:, :, 1]) & (color[:, :, 0] > 1.5 * color[:, :, 2])
                rows, _ = np.nonzero(mask)
                return int(rows.size), float(rows.mean()) if rows.size else None
            initial_count, initial_y = red_centroid(initial_rgb)
            final_count, final_y = red_centroid(final_rgb)
            downward_pixels = final_y - initial_y if initial_y is not None and final_y is not None else None
            # The fixed +Y camera has +Z image-up. The only red object is the
            # falling sphere, so a downward red centroid rejects static-image
            # false passes caused merely by stochastic rendering noise.
            motion_ok = (changed >= 10 and min(initial_count, final_count) >= 10
                         and downward_pixels is not None and downward_pixels >= 5.0)
            check("render_motion", motion_ok, changed_pixels_gt_8_levels=changed,
                  initial_red_pixels=initial_count, final_red_pixels=final_count,
                  initial_red_centroid_y=initial_y, final_red_centroid_y=final_y,
                  red_centroid_downward_displacement_pixels=downward_pixels,
                  requirement=">=10 changed pixels; >=10 red sphere pixels in both captures; centroid moves down >=5 pixels. Gross scene-update smoke, not photometric accuracy.")
    except Exception as error:
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        for name in required:
            if report["checks"][name]["status"] == "not_run":
                report["checks"][name] = {"status": "failed", "execution_status": "not_run",
                                           "reason": "Required phase could not execute; see fatal_error. No skip/pass credit."}
        if app is not None:
            try:
                if app_utils is not None:
                    app_utils.stop()
            except Exception as error:
                report["shutdown_stop_error"] = str(error)
        passed = all(report["checks"][name]["status"] == "passed" for name in required) and "fatal_error" not in report
        final_status = "passed_for_selected_smoke_mode" if passed else "failed"
        final_exit_code = 0 if passed else 1
        report["status"] = final_status
        report["exit_code"] = final_exit_code
        report["elapsed_seconds_before_shutdown"] = time.perf_counter() - started
        # Persist before Kit teardown so a teardown crash cannot erase evidence.
        report["shutdown_status"] = "pending" if app is not None else "not_started"
        if app is not None:
            # A hung/crashed teardown must leave an incomplete, non-passing
            # report even when every individual smoke check succeeded.
            report.update(status="pending_official_fast_exit", exit_code=final_exit_code,
                          shutdown_status="official_fast_exit_requested", shutdown_full_cleanup_verified=False)
        write_json(report_path, report)
        if app is not None:
            try:
                # Capture slow teardown stacks without attaching a debugger or
                # changing server ptrace policy. This timer never exits Kit.
                import faulthandler
                faulthandler.dump_traceback_later(120, repeat=True, exit=False)
                app.close(exit_code=final_exit_code)
                report.update(shutdown_status="completed", exit_code=final_exit_code, status=final_status)
            except Exception as error:
                report.update(status="failed", exit_code=1, shutdown_status="failed", shutdown_error=str(error))
            finally:
                faulthandler.cancel_dump_traceback_later()
            report["elapsed_seconds_total"] = time.perf_counter() - started
            write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "status": report["status"], "exit_code": report["exit_code"],
                      "full_platform_passed": False}, ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
