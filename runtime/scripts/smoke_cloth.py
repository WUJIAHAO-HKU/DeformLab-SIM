#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Headless CUDA cloth smoke test / instrumented baseline, not a new solver.

API reference: Newton v1.5.2 examples/cloth/example_cloth_hanging.py.
The model/stepping sequence follows that Apache-2.0 Newton example (Copyright
(c) 2025 The Newton Developers), with a smaller scene and explicit measurements.

Run in the project venv, from the project root:
    python scripts/smoke_cloth.py --json runs/cloth_smoke.json
    python scripts/smoke_cloth.py --json runs/cloth_usd.json --usd runs/cloth.usda
    python scripts/smoke_cloth.py --json runs/cloth_preview.json --npz runs/cloth.npz

Required: newton==1.5.2, warp-lang==1.16.0 (and its numpy dependency).
Optional USD export requires usd-core, e.g. the upstream-supported 25.5 release.
No torch, Isaac Sim, Kit, display server, GUI, or network access is used.

Timing is eager Python dispatch plus GPU execution, synchronized per frame.
Build/JIT warmup, state downloads, diagnostics, and USD writing are excluded
from measured frame times. Warmup ADVANCES the same trajectory. Every frame
boundary is checked, not every substep. This is a single-world baseline, not
Newton's maximum throughput; there is no CUDA graph or physical-accuracy claim.
Exit 0 means CUDA execution, finite sampled states, fixed vertices staying put,
and observable free-vertex motion under explicit numerical tolerances. Ground
penetration is reported without a fabricated pass/fail quality threshold.
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


EXPECTED_VERSIONS = {"newton": "1.5.2", "warp-lang": "1.16.0"}
UPSTREAM = "https://github.com/newton-physics/newton/blob/v1.5.2/"


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0", help="Explicit CUDA device; CPU is rejected.")
    parser.add_argument("--width", type=positive_int, default=32, help="Number of cloth cells along x.")
    parser.add_argument("--height", type=positive_int, default=32, help="Number of cloth cells along y.")
    parser.add_argument("--cell-size", type=positive_float, default=0.025, help="Cell size in meters.")
    parser.add_argument("--initial-height", type=positive_float, default=0.5, help="Initial z in meters.")
    parser.add_argument("--mass-per-particle", type=positive_float, default=0.01, help="Dynamic particle mass in kg.")
    parser.add_argument("--particle-radius", type=positive_float, default=0.01, help="Contact particle radius in meters.")
    parser.add_argument("--fps", type=positive_int, default=60)
    parser.add_argument("--substeps", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--warmup-frames", type=positive_int, default=30, help="At least one warmup frame is required.")
    parser.add_argument("--frames", type=positive_int, default=120, help="Measured frames after advancing warmup.")
    parser.add_argument("--json", type=Path, default=Path("runs/cloth_smoke.json"), help="Machine-readable report/config.")
    parser.add_argument("--usd", type=Path, default=None, help="Optional USD animation; requires usd-core.")
    parser.add_argument("--npz", type=Path, default=None, help="Optional initial/final geometry archive; numpy only.")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("WARP_CACHE_PATH") or Path(__file__).resolve().parents[1] / ".cache/warp"),
                        help="Warp cache base; default WARP_CACHE_PATH or project .cache/warp.")
    parser.add_argument("--fixed-tolerance", type=positive_float, default=1.0e-6,
                        help="Maximum sampled fixed-vertex drift, meters (numerical smoke check only).")
    parser.add_argument("--motion-tolerance", type=positive_float, default=1.0e-6,
                        help="Required free-vertex displacement from initial, meters (detects a no-op run).")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_report(args: argparse.Namespace) -> dict:
    scene = {
        "up_axis": "Z", "gravity_m_s2": [0.0, 0.0, -9.81], "ground_z_m": 0.0,
        "cloth_cells": [args.width, args.height], "cell_size_m": args.cell_size,
        "initial_position_m": [0.0, -0.5 * args.height * args.cell_size, args.initial_height],
        "initial_rotation_xyzw": [0.0, 0.0, 0.0, 1.0], "initial_velocity_m_s": [0.0, 0.0, 0.0],
        "mass_per_dynamic_particle_kg": args.mass_per_particle,
        "particle_radius_m": args.particle_radius, "fix_left": True,
        "tri_ke": 1000.0, "tri_ka": 1000.0, "tri_kd": 100.0,
        "edge_ke": 10.0, "edge_kd": 0.0, "tri_drag": 0.0, "tri_lift": 0.0,
        "soft_contact_ke": 100.0, "soft_contact_kd": 100.0, "soft_contact_mu": 1.0,
        "self_contact_enabled": False,
    }
    return {
        "schema_version": 1, "status": "started", "baseline_only": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [UPSTREAM + "newton/examples/cloth/example_cloth_hanging.py",
                    UPSTREAM + "newton/_src/sim/builder.py",
                    UPSTREAM + "newton/_src/solvers/vbd/solver_vbd.py"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": {name: package_version(name) for name in (*EXPECTED_VERSIONS, "numpy", "usd-core")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "config": {
            "expected_versions": EXPECTED_VERSIONS, "requested_device": args.device,
            "solver": "SolverVBD", "solver_iterations": args.iterations,
            "particle_enable_tile_solve": True, "friction_epsilon": 0.01,
            "scene": scene, "fps": args.fps, "substeps": args.substeps,
            "substep_dt_s": 1.0 / args.fps / args.substeps,
            "warmup_frames": args.warmup_frames, "measured_frames": args.frames,
            "warmup_advances_trajectory": True, "cuda_graph": False,
            "randomness": "No random sampling; bitwise GPU determinism is not claimed.",
            "json_path": str(args.json.resolve()),
            "usd_path": str(args.usd.resolve()) if args.usd else None,
            "npz_path": str(args.npz.resolve()) if args.npz else None,
            "warp_cache_base": str(args.cache_dir.resolve()),
            "numerical_fixed_vertex_drift_tolerance_m": args.fixed_tolerance,
            "numerical_free_vertex_motion_tolerance_m": args.motion_tolerance,
            "argv": sys.argv[1:],
        },
        "measurement_notes": [
            "A single fixed-edge cloth and a static ground; no self-collision or robot model.",
            "All non-overridden builder/collision/solver defaults are from pinned Newton v1.5.2.",
            "Timed intervals include eager dispatch, collision detection, solver steps and final device synchronization.",
            "Per-frame state download, diagnostics and optional USD logging are outside timed intervals.",
            "Warmup may not cover all data-dependent compilation; reported times are not a peak-throughput claim.",
            "Finite-state and penetration metrics sample frame boundaries, not the full substep trajectory.",
            "Center penetration=max(0, ground_z-z); radius-envelope penetration=max(0, ground_z+radius-z).",
            "Penetration counts use exact positive depth, with no tolerance or physical acceptance threshold.",
            "Fixed/moving vertex tolerances are numerical execution checks, not physical fidelity criteria.",
            "Passing this smoke test does not establish accuracy, realism, self-contact correctness or sim-to-real fidelity.",
        ],
        "frame_samples": [],
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace, report: dict) -> None:
    for name, expected in EXPECTED_VERSIONS.items():
        actual = report["environment"]["packages"][name]
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual!r}; activate the pinned project venv.")

    import numpy as np
    import warp as wp

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.cache_dir.resolve())

    import newton
    from newton.solvers import SolverVBD

    wp.init()
    report["environment"]["warp_kernel_cache_dir"] = wp.config.kernel_cache_dir
    if not args.device.startswith("cuda"):
        raise RuntimeError(f"CUDA is required; CPU/implicit device request {args.device!r} is rejected.")
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError(f"Requested {args.device!r} resolved to non-CUDA device {device}.")
    report["environment"]["device"] = {"alias": str(device), "name": device.name, "is_cuda": bool(device.is_cuda)}
    report["environment"]["module_paths"] = {"newton": newton.__file__, "warp": wp.__file__}
    report["environment"]["module_versions"] = {"newton": newton.__version__, "warp": wp.__version__}
    if newton.__version__ != EXPECTED_VERSIONS["newton"] or wp.__version__ != EXPECTED_VERSIONS["warp-lang"]:
        raise RuntimeError("Imported module versions differ from the pinned versions; check Python path shadowing.")

    viewer = None
    with wp.ScopedDevice(device):
        config = report["config"]["scene"]
        builder = newton.ModelBuilder(gravity=wp.vec3(*config["gravity_m_s2"]))
        builder.add_ground_plane(height=config["ground_z_m"])
        builder.add_cloth_grid(
            pos=wp.vec3(*config["initial_position_m"]), rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0), dim_x=args.width, dim_y=args.height,
            cell_x=args.cell_size, cell_y=args.cell_size, mass=args.mass_per_particle,
            fix_left=True, tri_ke=config["tri_ke"], tri_ka=config["tri_ka"], tri_kd=config["tri_kd"],
            tri_drag=config["tri_drag"], tri_lift=config["tri_lift"],
            edge_ke=config["edge_ke"], edge_kd=config["edge_kd"], particle_radius=args.particle_radius,
        )
        builder.color(include_bending=True)
        model = builder.finalize(device=device)
        model.soft_contact_ke = config["soft_contact_ke"]
        model.soft_contact_kd = config["soft_contact_kd"]
        model.soft_contact_mu = config["soft_contact_mu"]
        solver = SolverVBD(model, iterations=args.iterations, friction_epsilon=0.01,
                           particle_enable_self_contact=False, particle_enable_tile_solve=True)
        state, next_state = model.state(), model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        # Verify the actual simulation state location, not just the requested device.
        for label, array in (("model.particle_q", model.particle_q), ("state.particle_q", state.particle_q),
                             ("state.particle_qd", state.particle_qd), ("next_state.particle_q", next_state.particle_q)):
            if not array.device.is_cuda or str(array.device) != str(device):
                raise RuntimeError(f"{label} unexpectedly allocated on {array.device} instead of {device}.")

        masses = model.particle_mass.numpy().astype(np.float64)
        radii = model.particle_radius.numpy().astype(np.float64)
        initial_positions = state.particle_q.numpy().copy()
        fixed = masses == 0.0
        free = masses > 0.0
        report["model"] = {
            "particle_count": model.particle_count, "triangle_count": len(model.tri_indices),
            "dynamic_particle_count": int(np.count_nonzero(masses > 0)),
            "fixed_particle_count": int(np.count_nonzero(masses == 0)),
            "total_dynamic_mass_kg": float(masses.sum()), "state_device": str(state.particle_q.device),
        }
        if model.particle_count == 0:
            raise RuntimeError("Expected a non-empty cloth.")
        if not np.any(fixed) or not np.any(free):
            raise RuntimeError("Expected both fixed and dynamic cloth particles.")
        dt = report["config"]["substep_dt_s"]

        def inspect(phase: str, index: int, sim_time: float) -> None:
            positions = state.particle_q.numpy().astype(np.float64)
            velocities = state.particle_qd.numpy().astype(np.float64)
            finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
            sample = {"phase": phase, "frame_index": index, "simulation_time_s": sim_time, "finite": finite}
            if finite:
                z = positions[:, 2]
                center_depth = np.maximum(config["ground_z_m"] - z, 0.0)
                envelope_depth = np.maximum(config["ground_z_m"] + radii - z, 0.0)
                displacement = np.linalg.norm(positions - initial_positions, axis=1)
                sample.update({
                    "maximum_fixed_vertex_drift_m": float(displacement[fixed].max()),
                    "maximum_free_vertex_displacement_m": float(displacement[free].max()),
                    "minimum_center_z_m": float(z.min()),
                    "center_penetrating_particles": int(np.count_nonzero(center_depth > 0.0)),
                    "maximum_center_penetration_m": float(center_depth.max()),
                    "mean_center_penetration_all_particles_m": float(center_depth.mean()),
                    "radius_envelope_penetrating_particles": int(np.count_nonzero(envelope_depth > 0.0)),
                    "maximum_radius_envelope_penetration_m": float(envelope_depth.max()),
                    "mean_radius_envelope_penetration_all_particles_m": float(envelope_depth.mean()),
                    "maximum_speed_m_s": float(np.linalg.norm(velocities, axis=1).max()),
                })
            else:
                sample["nonfinite_position_components"] = int(np.count_nonzero(~np.isfinite(positions)))
                sample["nonfinite_velocity_components"] = int(np.count_nonzero(~np.isfinite(velocities)))
            report["frame_samples"].append(sample)
            if not finite:
                raise RuntimeError(f"Non-finite particle state at {phase} frame {index}.")
            if sample["maximum_fixed_vertex_drift_m"] > args.fixed_tolerance:
                raise RuntimeError(f"Fixed-vertex drift exceeds {args.fixed_tolerance} m at {phase} frame {index}.")

        def simulate_frame() -> None:
            nonlocal state, next_state
            for _ in range(args.substeps):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, next_state, control, contacts, dt)
                state, next_state = next_state, state

        def export_frame(sim_time: float) -> None:
            if viewer is not None:
                viewer.begin_frame(sim_time)
                viewer.log_state(state)
                viewer.end_frame()

        try:
            if args.usd:
                from newton.viewer import ViewerUSD
                args.usd.parent.mkdir(parents=True, exist_ok=True)
                viewer = ViewerUSD(str(args.usd), fps=args.fps, up_axis="Z",
                                   num_frames=args.warmup_frames + args.frames + 1)
                viewer.set_model(model)
            wp.synchronize_device(device)
            inspect("initial", 0, 0.0)
            export_frame(0.0)
            warmup_start = time.perf_counter()
            for frame in range(1, args.warmup_frames + 1):
                simulate_frame()
                wp.synchronize_device(device)
                sim_time = frame / args.fps
                inspect("warmup", frame, sim_time)
                export_frame(sim_time)
            report["warmup_wall_time_s_including_diagnostics_and_usd"] = time.perf_counter() - warmup_start

            durations = []
            for frame in range(1, args.frames + 1):
                wp.synchronize_device(device)
                start = time.perf_counter()
                simulate_frame()
                wp.synchronize_device(device)
                elapsed = time.perf_counter() - start
                durations.append(elapsed)
                sim_time = (args.warmup_frames + frame) / args.fps
                inspect("measured", frame, sim_time)
                report["frame_samples"][-1]["synchronized_frame_wall_time_s"] = elapsed
                export_frame(sim_time)
            total = sum(durations)
            measured = [sample for sample in report["frame_samples"] if sample["phase"] == "measured"]
            maximum_free_motion = max(s["maximum_free_vertex_displacement_m"] for s in report["frame_samples"])
            if maximum_free_motion <= args.motion_tolerance:
                raise RuntimeError(f"Free-vertex displacement never exceeded {args.motion_tolerance} m; possible no-op run.")
            report["timing"] = {
                "method": "perf_counter; synchronize selected CUDA device before/after each eager frame",
                "measured_simulation_start_s": args.warmup_frames / args.fps,
                "measured_simulation_duration_s": args.frames / args.fps,
                "sum_synchronized_frame_times_s": total,
                "mean_frame_ms": statistics.mean(durations) * 1000.0,
                "median_frame_ms": statistics.median(durations) * 1000.0,
                "p95_frame_ms": float(np.percentile(durations, 95)) * 1000.0,
                "min_frame_ms": min(durations) * 1000.0, "max_frame_ms": max(durations) * 1000.0,
                "simulated_frames_per_wall_second": args.frames / total,
                "simulated_substeps_per_wall_second": args.frames * args.substeps / total,
                "real_time_factor": (args.frames / args.fps) / total,
            }
            report["observations"] = {
                "all_sampled_states_finite": True, "sample_count": len(report["frame_samples"]),
                "fixed_vertices_within_numerical_tolerance": True,
                "free_vertex_motion_exceeds_numerical_tolerance": True,
                "maximum_fixed_vertex_drift_m": max(s["maximum_fixed_vertex_drift_m"] for s in report["frame_samples"]),
                "maximum_free_vertex_displacement_m": maximum_free_motion,
                "measured_minimum_center_z_m": min(s["minimum_center_z_m"] for s in measured),
                "measured_maximum_center_penetration_m": max(s["maximum_center_penetration_m"] for s in measured),
                "measured_maximum_radius_envelope_penetration_m": max(s["maximum_radius_envelope_penetration_m"] for s in measured),
                "measured_frames_with_center_penetration": sum(s["center_penetrating_particles"] > 0 for s in measured),
                "penetration_quality_verdict": "not_assessed; no acceptance threshold is defined",
            }
            if args.npz:
                args.npz.parent.mkdir(parents=True, exist_ok=True)
                # An open file avoids numpy silently appending a different suffix.
                with args.npz.open("wb") as archive:
                    np.savez_compressed(
                        archive, initial_positions=initial_positions, final_positions=state.particle_q.numpy(),
                        final_velocities=state.particle_qd.numpy(), triangle_indices=model.tri_indices.numpy(),
                        particle_mass=masses, particle_radius=radii, fixed_vertex_mask=fixed,
                        config_json=np.array(json.dumps(report["config"], allow_nan=False)),
                    )
            report["status"] = "passed_cuda_finite_state_smoke"
        finally:
            if viewer is not None:
                viewer.close()


def main() -> int:
    args = parse_args()
    output_paths = [path.resolve() for path in (args.json, args.usd, args.npz) if path is not None]
    if len(set(output_paths)) != len(output_paths):
        raise SystemExit("--json, --usd and --npz must use different paths")
    report = make_report(args)
    # Persist the exact requested configuration even if CUDA initialization fails.
    write_report(args.json, report)
    exit_code = 0
    try:
        run(args, report)
    except Exception as exc:
        exit_code = 1
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], file=sys.stderr)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(args.json, report)
    print(json.dumps({"status": report["status"], "json": str(args.json.resolve()),
                      "timing": report.get("timing"), "observations": report.get("observations")}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
