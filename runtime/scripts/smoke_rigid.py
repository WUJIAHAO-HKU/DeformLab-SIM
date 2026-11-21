#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Newton 1.5.2 CUDA rigid sphere-drop/contact baseline; no original solver.

The scene and XPBD stepping follow the official basic_shapes.py example
(Apache-2.0, Copyright (c) 2025 The Newton Developers). Rendering/assets are
omitted. Requires newton==1.5.2 and warp-lang==1.16.0, tested locally only for
CLI and assessment semantics until the target GPU run is performed.

    python scripts/smoke_rigid.py --json results/rigid-smoke.json

Warmup advances the same trajectory. Timings are eager Python dispatch plus
collision/solver GPU work and synchronization, excluding state downloads,
diagnostics, and JSON I/O. They are not peak-throughput measurements.

Default numerical smoke criteria: finite sampled states; >=1 cm downward
movement and negative vertical velocity; generated rigid contacts; <=5 mm
sampled sphere/plane penetration; final 20 frames within 5 mm of rest height
and <=0.02 m/s linear speed. These intentionally loose geometric/execution
checks are not real-world accuracy tolerances. Samples are at frame boundaries
and contact counts refer to each frame's LAST substep collision pass. There is
no claim that unsampled substeps are penetration-free. No GUI/torch/Isaac/Kit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


PINNED = {"newton": "1.5.2", "warp-lang": "1.16.0"}
SOURCE = "https://github.com/newton-physics/newton/blob/v1.5.2/newton/examples/basic/example_basic_shapes.py"


def positive_int(text: str) -> int:
    result = int(text)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(text: str) -> float:
    result = float(text)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=Path("results/rigid-smoke.json"))
    parser.add_argument("--device", default="cuda:0", help="Explicit CUDA device; CPU is rejected.")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("WARP_CACHE_PATH") or Path(__file__).resolve().parents[1] / ".cache/warp"))
    parser.add_argument("--frames", type=positive_int, default=240, help="Measured frames after warmup.")
    parser.add_argument("--warmup-frames", type=positive_int, default=10)
    parser.add_argument("--fps", type=positive_int, default=100)
    parser.add_argument("--substeps", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--radius", type=positive_float, default=0.5, help="Sphere radius, meters.")
    parser.add_argument("--initial-height", type=positive_float, default=2.0, help="Sphere-center z, meters.")
    parser.add_argument("--density", type=positive_float, default=1000.0, help="Sphere density, kg/m^3.")
    parser.add_argument("--penetration-tolerance", type=positive_float, default=0.005, help="Sampled penetration, meters.")
    parser.add_argument("--rest-height-tolerance", type=positive_float, default=0.005, help="Settled center-height error, meters.")
    parser.add_argument("--settle-speed-tolerance", type=positive_float, default=0.02, help="Settled linear speed, m/s.")
    parser.add_argument("--settle-frames", type=positive_int, default=20, help="Consecutive trailing frames checked for rest.")
    parser.add_argument("--fall-distance", type=positive_float, default=0.01, help="Minimum observed downward displacement, meters.")
    result = parser.parse_args()
    if result.initial_height - result.radius <= max(result.fall_distance, result.rest_height_tolerance):
        parser.error("initial-height must place the sphere clearly above the plane, with room to fall")
    if result.frames + result.warmup_frames < result.settle_frames:
        parser.error("frames + warmup-frames must be >= settle-frames")
    return result


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def assess_trajectory(samples: list[dict], config: dict) -> dict:
    """Pure assessment function, allowing CPU-only negative-control checks."""
    noninitial = [sample for sample in samples if sample["phase"] != "initial"]
    if (not samples or any(not sample["finite"] for sample in samples) or not noninitial
            or any(not math.isfinite(value)
                   for sample in samples
                   for field in ("position_m", "linear_velocity_m_s", "orientation_xyzw", "angular_velocity_rad_s")
                   for value in sample.get(field, []))):
        return {"passed": False, "checks": {"finite_nonempty_trajectory": False}}
    tail = noninitial[-config["settle_frames"]:]
    rest_z = config["ground_z_m"] + config["radius_m"]
    maximum_penetration = max(max(0.0, rest_z - sample["position_m"][2]) for sample in samples)
    downward_displacement = config["initial_height_m"] - min(sample["position_m"][2] for sample in samples)
    minimum_vertical_velocity = min(sample["linear_velocity_m_s"][2] for sample in noninitial)
    tail_height_error = max(abs(sample["position_m"][2] - rest_z) for sample in tail)
    tail_speed = max(math.sqrt(sum(v * v for v in sample["linear_velocity_m_s"])) for sample in tail)
    checks = {
        "finite_nonempty_trajectory": True,
        "actually_fell": downward_displacement >= config["fall_distance_m"] and minimum_vertical_velocity < 0.0,
        "rigid_contacts_observed": any(sample["last_substep_rigid_contact_count"] > 0 for sample in noninitial),
        "sampled_penetration_within_tolerance": maximum_penetration <= config["penetration_tolerance_m"],
        "complete_settling_window": len(tail) == config["settle_frames"],
        "settled_at_ground_height": tail_height_error <= config["rest_height_tolerance_m"],
        "settled_linear_velocity": tail_speed <= config["settle_speed_tolerance_m_s"],
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "maximum_sampled_penetration_m": maximum_penetration,
        "maximum_downward_displacement_m": downward_displacement,
        "minimum_sampled_vertical_velocity_m_s": minimum_vertical_velocity,
        "settling_window_maximum_height_error_m": tail_height_error,
        "settling_window_maximum_linear_speed_m_s": tail_speed,
        "sampled_frames_with_generated_rigid_contacts": sum(s["last_substep_rigid_contact_count"] > 0 for s in noninitial),
    }


def execute(args: argparse.Namespace, report: dict) -> None:
    for package, expected in PINNED.items():
        actual = report["environment"]["packages"][package]
        if actual != expected:
            raise RuntimeError(f"Expected {package}=={expected}, found {actual!r}; activate the pinned project venv.")

    import numpy as np
    import warp as wp

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.cache_dir.resolve())

    import newton
    from newton.solvers import SolverXPBD

    wp.init()
    if not args.device.startswith("cuda"):
        raise RuntimeError("An explicit CUDA device is required; CPU execution is not accepted.")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError(f"Requested CUDA but resolved to {device}.")
    if newton.__version__ != PINNED["newton"] or wp.__version__ != PINNED["warp-lang"]:
        raise RuntimeError("Imported modules do not match pinned package versions; check Python path shadowing.")
    report["environment"].update({
        "device": {"name": device.name, "alias": str(device), "is_cuda": bool(device.is_cuda)},
        "module_paths": {"newton": newton.__file__, "warp": wp.__file__},
        "module_versions": {"newton": newton.__version__, "warp": wp.__version__},
        "warp_kernel_cache_dir": wp.config.kernel_cache_dir,
    })
    config = report["config"]
    with wp.ScopedDevice(device):
        newton.use_coord_layout_targets = True
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.default_shape_cfg.density = args.density
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.mu_torsional = 0.01
        builder.default_shape_cfg.mu_rolling = 0.003
        builder.default_shape_cfg.restitution = 0.0
        builder.default_shape_cfg.margin = 0.0
        # add_ground_plane creates a static world shape; there is no ground body.
        builder.add_ground_plane(height=0.0)
        sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, args.initial_height),
                                                    q=wp.quat_identity()), label="sphere")
        builder.add_shape_sphere(sphere, radius=args.radius)
        model = builder.finalize(device=device)
        solver = SolverXPBD(model, iterations=args.iterations, enable_restitution=False)
        state, next_state = model.state(), model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        arrays = {"model.body_q": model.body_q, "model.body_inv_mass": model.body_inv_mass,
                  "state.body_q": state.body_q, "state.body_qd": state.body_qd,
                  "next_state.body_q": next_state.body_q,
                  "contacts.rigid_contact_count": contacts.rigid_contact_count}
        for name, array in arrays.items():
            if array is None or not array.device.is_cuda or str(array.device) != str(device):
                raise RuntimeError(f"{name} is not allocated on requested CUDA device {device}.")
        inverse_mass = float(model.body_inv_mass.numpy()[sphere])
        if model.body_count != 1 or model.shape_count != 2 or not math.isfinite(inverse_mass) or inverse_mass <= 0.0:
            raise RuntimeError("Expected exactly one dynamic sphere, two collision shapes, and positive inverse mass.")
        report["model"] = {"body_count": model.body_count, "shape_count": model.shape_count,
                           "sphere_body_index": sphere, "sphere_mass_kg": 1.0 / inverse_mass,
                           "array_devices": {name: str(array.device) for name, array in arrays.items()}}

        def record(frame: int, phase: str) -> None:
            pose = state.body_q.numpy()[sphere].astype(np.float64)
            velocity = state.body_qd.numpy()[sphere].astype(np.float64)
            finite = bool(np.isfinite(pose).all() and np.isfinite(velocity).all())
            sample = {"frame_index": frame, "phase": phase, "simulation_time_s": frame / args.fps,
                      "finite": finite, "last_substep_rigid_contact_count": int(contacts.rigid_contact_count.numpy()[0])}
            if finite:
                # Newton 1.5.2 uses [linear xyz, angular xyz], unlike older Warp conventions.
                sample.update({"position_m": pose[:3].tolist(), "orientation_xyzw": pose[3:].tolist(),
                               "linear_velocity_m_s": velocity[:3].tolist(), "angular_velocity_rad_s": velocity[3:].tolist(),
                               "geometric_penetration_m": max(0.0, args.radius - float(pose[2]))})
            report["trajectory"].append(sample)
            if not finite:
                raise RuntimeError(f"Non-finite rigid state at frame {frame}.")

        def step_frame() -> None:
            nonlocal state, next_state
            for _ in range(args.substeps):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, next_state, control, contacts, config["substep_dt_s"])
                state, next_state = next_state, state

        wp.synchronize_device(device)
        record(0, "initial")
        start = time.perf_counter()
        for frame in range(1, args.warmup_frames + 1):
            step_frame()
            wp.synchronize_device(device)
            record(frame, "warmup")
        report["warmup_wall_time_s_including_diagnostics"] = time.perf_counter() - start
        durations = []
        for measured_frame in range(1, args.frames + 1):
            wp.synchronize_device(device)
            start = time.perf_counter()
            step_frame()
            wp.synchronize_device(device)
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            record(args.warmup_frames + measured_frame, "measured")
            report["trajectory"][-1]["synchronized_frame_wall_time_s"] = elapsed
        total = sum(durations)
        report["timing"] = {
            "sum_synchronized_frame_times_s": total,
            "mean_frame_ms": statistics.mean(durations) * 1000.0,
            "median_frame_ms": statistics.median(durations) * 1000.0,
            "p95_frame_ms": float(np.percentile(durations, 95)) * 1000.0,
            "measured_simulation_start_s": args.warmup_frames / args.fps,
            "measured_simulation_duration_s": args.frames / args.fps,
            "simulated_frames_per_wall_second": args.frames / total,
            "real_time_factor": (args.frames / args.fps) / total,
        }
        report["assessment"] = assess_trajectory(report["trajectory"], config)
        if not report["assessment"]["passed"]:
            failed = [name for name, passed in report["assessment"]["checks"].items() if not passed]
            raise RuntimeError("Numerical smoke checks failed: " + ", ".join(failed))
        report["status"] = "passed_cuda_rigid_drop_contact_smoke"


def main() -> int:
    args = arguments()
    report = {
        "schema_version": 1, "status": "started", "baseline_only": True, "upstream_source": SOURCE,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "packages": {name: version(name) for name in (*PINNED, "numpy")},
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "config": {
            "expected_versions": PINNED, "requested_device": args.device, "solver": "SolverXPBD",
            "up_axis": "Z", "gravity_m_s2": [0.0, 0.0, -9.81], "ground_z_m": 0.0,
            "initial_position_m": [0.0, 0.0, args.initial_height], "initial_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "initial_height_m": args.initial_height, "radius_m": args.radius, "density_kg_m3": args.density,
            "shape_mu": 0.5, "shape_mu_torsional": 0.01, "shape_mu_rolling": 0.003,
            "shape_margin_m": 0.0, "shape_restitution": 0.0, "solver_enable_restitution": False,
            "fps": args.fps, "substeps": args.substeps, "substep_dt_s": 1.0 / args.fps / args.substeps,
            "solver_iterations": args.iterations, "measured_frames": args.frames, "warmup_frames": args.warmup_frames,
            "warmup_advances_trajectory": True, "cuda_graph": False, "settle_frames": args.settle_frames,
            "fall_distance_m": args.fall_distance, "penetration_tolerance_m": args.penetration_tolerance,
            "rest_height_tolerance_m": args.rest_height_tolerance, "settle_speed_tolerance_m_s": args.settle_speed_tolerance,
            "ballistic_time_to_ground_s": math.sqrt(2.0 * (args.initial_height - args.radius) / 9.81),
            "warp_cache_base": str(args.cache_dir.resolve()), "json_path": str(args.json.resolve()), "argv": sys.argv[1:],
        },
        "notes": [
            "Official Newton baseline; no original solver or real-world material calibration is claimed.",
            "Unspecified builder/collision/XPBD defaults are those of pinned Newton 1.5.2.",
            "Eager per-frame wall-clock timing is synchronized on the selected GPU; downloads/diagnostics/JSON are excluded.",
            "Warmup may not cover data-dependent contact compilation; numbers are not peak throughput.",
            "Finite-state/penetration checks sample frame boundaries; contact counts come from the last substep collision pass.",
            "Passing requires actual fall, generated contacts and a complete trailing rest window; it is numerical smoke validation only.",
            "Thresholds are explicit geometric/execution checks, not evidence of real-world contact-force accuracy.",
            "Shortened trajectories can correctly fail because the sphere has not contacted/settled yet.",
        ],
        "trajectory": [],
    }
    save(args.json, report)
    exit_code = 0
    try:
        execute(args, report)
    except Exception as exc:
        exit_code = 1
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.json, report)
    print(json.dumps({"status": report["status"], "json": str(args.json.resolve()),
                      "assessment": report.get("assessment"), "timing": report.get("timing")}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
