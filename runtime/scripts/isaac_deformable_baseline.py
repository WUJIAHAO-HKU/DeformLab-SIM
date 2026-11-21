#!/usr/bin/env python3
"""Two official Isaac Sim 6.1 / PhysX deformable baselines, one case per process.

Run with the pinned build's python.sh (not the Newton environment)::

    ./python.sh /project/scripts/isaac_deformable_baseline.py --case surface \
        --source-root /project/upstream/IsaacSim \
        --build-receipt /project/results/isaac-build-receipt.json
    ./python.sh /project/scripts/isaac_deformable_baseline.py --case volume \
        --source-root /project/upstream/IsaacSim \
        --build-receipt /project/results/isaac-build-receipt.json

Surface: a free triangular sheet falls over a FIXED rigid sphere and floor.
Volume: a directly authored, conforming TetMesh block falls onto a fixed floor.
No cloud asset, external tetrahedralizer, Newton solver or replacement GUI is
used. These are existing PhysX capabilities, not original solver contributions.
This is partial DEF-01 smoke coverage, not full-platform/material validation.

APIs verified in the pinned source's experimental.prims DeformablePrim tests
(direct Mesh / TetMesh), DeformablePrim implementation, and experimental.materials
SurfaceDeformableMaterial / VolumeDeformableMaterial. Surface queries deliberately
avoid the volume-only stress, gradient and kinematic-target methods.

initial.usda is the authored physics scene before playback; final-static.usda
is a VISUAL SNAPSHOT from measured simulation nodes, with no physics schema.
Neither export receives clean-process reload credit from this script. The NPZ
contains every sampled simulation/collision node position and nodal velocity.
Contact diagnostics are geometric distances, NOT measured contact forces.
Numerical gates detect gross no-op, rigid-only, inversion and penetration errors;
the illustrative material parameters have not been calibrated to a real object.
Use an external process timeout: pending_shutdown is not a passing result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = "isaac-physx-deformable-baseline-v1"
BODY = "/World/Deformable"


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_points(points):
    import numpy as np
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Expected nonempty finite N x 3 points")
    return value


def checked_elements(elements, node_count: int, width: int):
    import numpy as np
    value = np.asarray(elements)
    if (value.ndim != 2 or value.shape[1] != width or not len(value)
            or not np.issubdtype(value.dtype, np.integer)
            or value.min() < 0 or value.max() >= node_count):
        raise ValueError("Invalid element connectivity")
    if np.any(np.diff(np.sort(value, axis=1), axis=1) == 0):
        raise ValueError("An element repeats a vertex")
    return value.astype(np.int64)


def signed_tet_volumes(points, elements):
    import numpy as np
    points = checked_points(points)
    t = points[checked_elements(elements, len(points), 4)]
    return np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                     t[:, 3] - t[:, 0]) / 6.0


def boundary_triangles(elements):
    """Outward boundary of a positive-orientation conforming tetrahedral mesh."""
    import numpy as np
    elements = np.asarray(elements)
    if elements.ndim != 2 or elements.shape[1] != 4 or not len(elements):
        raise ValueError("Expected nonempty tetrahedra")
    faces = {}
    # Oriented boundary of [a,b,c,d]: [b,c,d]-[a,c,d]+[a,b,d]-[a,b,c].
    for a, b, c, d in elements:
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(face))
            faces.setdefault(key, []).append(face)
            if len(faces[key]) > 2:
                raise ValueError("Nonmanifold tetrahedral face")
    for incident in faces.values():
        if len(incident) == 2:
            first, second = incident
            if any(second == first[i:] + first[:i] for i in range(3)):
                raise ValueError("Shared tetrahedral faces have equal orientation")
    return np.asarray([incident[0] for incident in faces.values() if len(incident) == 1], dtype=np.int32)


def build_mesh(case: str, resolution: int, tet_pattern: str = "six", resolution_z: int | None = None):
    """Deterministic SI meshes. Resolution is cells per edge, not vertex count."""
    import numpy as np
    if type(resolution) is not int or resolution < 1:
        raise ValueError("resolution must be a positive integer")
    if resolution_z is not None and (case != "volume" or type(resolution_z) is not int or resolution_z < 1):
        raise ValueError("resolution_z requires a positive volume-mesh layer count")
    if case == "surface":
        axis = np.linspace(-0.7, 0.7, resolution + 1)
        points = np.array([[x, y, 1.1] for y in axis for x in axis], dtype=np.float64)
        elements = []
        for y in range(resolution):
            for x in range(resolution):
                a = y * (resolution + 1) + x
                b, c, d = a + 1, a + resolution + 1, a + resolution + 2
                elements.extend(((a, b, d), (a, d, c)))
        return points, np.asarray(elements, dtype=np.int32)
    if case != "volume":
        raise ValueError("Unknown deformable case")
    axis = np.linspace(-0.25, 0.25, resolution + 1)
    layers = resolution if resolution_z is None else resolution_z
    axis_z = axis if resolution_z is None else np.linspace(-0.25, 0.25, layers + 1)
    points = np.array([[x, y, z + 0.95] for z in axis_z for y in axis for x in axis], dtype=np.float64)
    stride = resolution + 1
    elements = []
    for z in range(layers):
        for y in range(resolution):
            for x in range(resolution):
                a = (z * stride + y) * stride + x
                b, c, d = a + 1, a + stride, a + stride + 1
                e, f, g, h = a + stride**2, a + stride**2 + 1, a + stride**2 + stride, a + stride**2 + stride + 1
                if tet_pattern == "six":
                    elements.extend(((a, b, d, h), (a, d, c, h), (a, c, g, h),
                                     (a, g, e, h), (a, e, f, h), (a, f, b, h)))
                elif tet_pattern == "five-alternating":
                    corners = [a, b, d, c, e, f, h, g]
                    if (x + y + z) % 2:
                        corners = [corners[i] for i in [1, 0, 3, 2, 5, 4, 7, 6]]
                    elements.extend(tuple(corners[i] for i in tet) for tet in
                        [(0,1,3,4), (1,2,3,6), (1,4,5,6), (3,4,6,7), (1,3,4,6)])
                else:
                    raise ValueError("Unknown tet pattern")
    elements = np.asarray(elements, dtype=np.int32)
    volumes = signed_tet_volumes(points, elements)
    if tet_pattern == "five-alternating":
        reverse = volumes < 0
        elements[reverse, :2] = elements[reverse, :2][:, ::-1]
        volumes = signed_tet_volumes(points, elements)
    if not np.all(volumes > 0) or not np.isclose(volumes.sum(), 0.5**3, rtol=1e-12):
        raise ValueError("Generated tetrahedra lack consistent positive orientation/volume")
    boundary_triangles(elements)  # also rejects duplicate/nonconforming face incidence
    return points, elements


def rigid_fit_residual(reference, observed) -> float:
    """RMS after the best proper rigid transform; translation/rotation do not count."""
    import numpy as np
    a, b = checked_points(reference), checked_points(observed)
    if a.shape != b.shape:
        raise ValueError("Node correspondence changed")
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    residual = a @ (u @ correction @ vt) - b
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def assess_trajectory(case, simulation, collision, velocities, simulation_elements,
                      collision_elements, times, step_numbers, config) -> dict:
    """Judge raw arrays; caller-provided finite flags cannot hide NaN/Inf."""
    import numpy as np
    checks = {}
    try:
        simulation, collision, velocities = [np.asarray(v, dtype=np.float64) for v in (simulation, collision, velocities)]
        times, step_numbers = np.asarray(times, dtype=float), np.asarray(step_numbers)
        count = len(simulation)
        shape_ok = (simulation.ndim == collision.ndim == velocities.ndim == 3
                    and simulation.shape == velocities.shape and collision.shape[0] == count
                    and simulation.shape[2] == collision.shape[2] == 3
                    and simulation.shape[1] > 0 and collision.shape[1] > 0
                    and times.shape == step_numbers.shape == (count,) and count >= 2)
        checks["complete_array_shapes"] = bool(shape_ok and count == config["steps"] + 1)
        checks["finite_raw_trajectory"] = all(np.isfinite(v).all() for v in (simulation, collision, velocities, times, step_numbers))
        if not shape_ok or not checks["finite_raw_trajectory"]:
            return {"passed": False, "checks": checks}
        width = 3 if case == "surface" else 4
        sim_elements = checked_elements(simulation_elements, simulation.shape[1], width)
        col_elements = checked_elements(collision_elements, collision.shape[1], width)
        checks["step_and_time_progress"] = bool(np.all(np.diff(step_numbers) == 1)
            and np.allclose(np.diff(times), 1 / config["physics_hz"], rtol=1e-5, atol=1e-9))
        residuals = [rigid_fit_residual(simulation[0], frame) for frame in simulation]
        centroids = simulation.mean(axis=1)
        floor_z = collision[:, :, 2].min(axis=1)
        displacement = np.linalg.norm(simulation - simulation[0], axis=2).max()
        checks["falls"] = bool(centroids[0, 2] - centroids[-1, 2] > config["minimum_centroid_drop_m"])
        checks["nonrigid_deformation_observed"] = max(residuals) > config["minimum_deformation_rms_m"]
        checks["nonzero_nodal_velocity_observed"] = bool(np.linalg.norm(velocities, axis=2).max() > 1e-5)
        checks["floor_penetration_within_smoke_gate"] = bool(floor_z.min() >= -config["penetration_tolerance_m"])
        metrics = {"maximum_node_displacement_m": float(displacement),
                   "initial_node_centroid_m": centroids[0].tolist(), "final_node_centroid_m": centroids[-1].tolist(),
                   "peak_nonrigid_rms_m": max(residuals), "final_nonrigid_rms_m": residuals[-1],
                   "minimum_floor_clearance_m": float(floor_z.min()),
                   "final_floor_clearance_m": float(floor_z[-1]),
                   "tail_maximum_nodal_speed_m_s": float(np.linalg.norm(velocities[-30:], axis=2).max()),
                   "tail_node_centroid_component_span_m": float(np.ptp(centroids[-30:], axis=0).max()),
                   "centroid_definition": "arithmetic node mean, not a mass-weighted center of mass"}
        if case == "surface":
            geometry = load_local("_isaac_deformable_geometry", PROJECT / "src/deformable_sim/validation.py")
            gaps = [geometry.point_triangle_distance(np.array([0, 0, 0.35]), frame[col_elements]) - 0.35
                    for frame in collision]
            checks["sphere_penetration_within_smoke_gate"] = min(gaps) >= -config["penetration_tolerance_m"]
            checks["final_near_sphere_geometry"] = abs(gaps[-1]) <= config["contact_band_m"]
            checks["final_sheet_draped"] = bool(np.ptp(collision[-1, :, 2]) > 0.15
                and collision[-1, :, 2].max() > 0.45 and residuals[-1] > config["minimum_deformation_rms_m"])
            metrics.update(minimum_sphere_surface_clearance_m=min(gaps), final_sphere_surface_clearance_m=gaps[-1],
                           final_vertical_span_m=float(np.ptp(collision[-1, :, 2])))
            contact_mask = np.abs(gaps) <= config["contact_band_m"]
        else:
            initial_volumes = signed_tet_volumes(simulation[0], sim_elements)
            checks["runtime_initial_tets_positive"] = bool(np.all(initial_volumes > 0))
            if not checks["runtime_initial_tets_positive"]:
                return {"passed": False, "checks": checks, "metrics": metrics}
            ratios = np.asarray([signed_tet_volumes(frame, sim_elements) / initial_volumes for frame in simulation])
            checks["no_inverted_or_collapsed_tets"] = bool(ratios.min() > 1e-3)
            checks["final_near_floor_geometry"] = abs(float(floor_z[-1])) <= config["contact_band_m"]
            metrics.update(minimum_signed_volume_ratio=float(ratios.min()), maximum_signed_volume_ratio=float(ratios.max()),
                           runtime_initial_minimum_signed_tet_volume_m3=float(initial_volumes.min()))
            contact_mask = np.abs(floor_z) <= config["contact_band_m"]
        touched = np.flatnonzero(contact_mask)
        metrics.update(near_contact_samples=int(len(touched)),
                       first_near_contact_time_s=float(times[touched[0]]) if len(touched) else None)
        scalars = [{"step": int(step_numbers[i]), "time_s": float(times[i]), "node_centroid_m": centroids[i].tolist(),
                    "nonrigid_rms_m": residuals[i], "floor_clearance_m": float(floor_z[i])}
                   for i in range(count)]
        return {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()},
                "metrics": metrics, "scalar_trajectory": scalars,
                "contact_evidence": "geometry proximity/penetration only; no contact-force or impulse query",
                "settling": "tail speed/span reported only; no physical calibration or settling-accuracy claim"}
    except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
        return {"passed": False, "checks": {k: bool(v) for k, v in checks.items()}, "error": str(error)}


def assess_control(states, velocities, elements, times, steps, config):
    """Independent static/freefall controls, not a weakened impact acceptance."""
    import numpy as np
    q, v = np.asarray(states), np.asarray(velocities)
    t = np.asarray(times) - times[0]
    g = config["gravity_m_s2"]
    expected = q[0][None] + t[:, None, None] * v[0][None]
    expected[:, :, 2] -= .5 * g * t[:, None] * (t[:, None] + 1 / config["physics_hz"])
    error = float(np.max(np.abs(q - expected)))
    initial_volumes = signed_tet_volumes(q[0], elements)
    ratios = np.array([signed_tet_volumes(frame, elements) / initial_volumes for frame in q])
    checks = dict(complete_samples=len(q)==config["steps"]+1, finite=bool(np.isfinite(q).all() and np.isfinite(v).all()),
                  time_advances=bool(np.allclose(np.diff(times), 1 / config["physics_hz"], rtol=1e-6, atol=1e-9) and np.all(np.diff(steps)==1)),
                  initial_elements_positive=bool(np.all(initial_volumes > 0)),
                  analytical_motion=error<.01, positive_elements=bool(ratios.min()>1e-3))
    return dict(passed=all(checks.values()), checks=checks, metrics=dict(maximum_analytical_position_error_m=error,
        minimum_signed_volume_ratio=float(ratios.min())), scalar_trajectory=[],
        contact_evidence="Collision-disabled control; no contact validation", settling="Not evaluated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("surface", "volume"), required=True)
    parser.add_argument("--source-root", type=Path, default=PROJECT / "upstream/IsaacSim")
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--experience", default="", help="Only the pinned built isaacsim.exp.full.kit is accepted")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--resolution", type=int, help="cells per edge; surface default 24, volume default 4")
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument("--position-iterations", type=int, default=16)
    parser.add_argument("--young-modulus", type=float)
    parser.add_argument("--control", choices=("impact", "freefall", "rest"), default="impact")
    parser.add_argument("--speculative-ccd", action="store_true")
    parser.add_argument("--self-collision", action="store_true")
    parser.add_argument("--max-depenetration-velocity", type=float)
    parser.add_argument("--elasticity-damping", type=float)
    parser.add_argument("--tet-pattern", choices=["six", "five-alternating"], default="six")
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--poisson-ratio", type=float, default=.3)
    parser.add_argument("--volume-side", type=float, default=.5)
    parser.add_argument("--volume-center-z", type=float, default=.95)
    parser.add_argument("--friction", type=float, default=.5)
    parser.add_argument("--contact-offset", type=float)
    parser.add_argument("--rest-offset", type=float)
    parser.add_argument("--linear-damping", type=float)
    parser.add_argument("--settling-damping", type=float)
    parser.add_argument("--settling-threshold", type=float)
    parser.add_argument("--sleep-threshold", type=float)
    args = parser.parse_args(argv)
    args.resolution = args.resolution if args.resolution is not None else (24 if args.case == "surface" else 4)
    if not 2 <= args.resolution <= (64 if args.case == "surface" else 12):
        parser.error("resolution outside surface [2,64] / volume [2,12]")
    if not 60 <= args.physics_hz <= 960 or not 60 <= args.steps <= 7200:
        parser.error("physics-hz must be [60,960], steps [60,7200]")
    if args.young_modulus is not None and not 0 < args.young_modulus <= 1e9:
        parser.error("young-modulus must be finite and in (0,1e9]")
    if not 1 <= args.position_iterations <= 255:
        parser.error("position-iterations must be [1,255]")
    for value in [args.max_depenetration_velocity, args.elasticity_damping]:
        if value is not None and not 0 <= value <= 1e6:
            parser.error("Damping/depenetration value must be finite in [0,1e6]")
    if args.control != "impact" and args.case != "volume":
        parser.error("Static/freefall controls currently require volume")
    import math
    if not 0 <= args.gravity <= 100 or not 0 <= args.poisson_ratio < .5:
        parser.error("Invalid gravity/Poisson ratio")
    if not all(math.isfinite(v) for v in [args.volume_side, args.volume_center_z, args.friction]) or not .01 <= args.volume_side <= 2 or not 0 <= args.friction <= 2:
        parser.error("Invalid volume dimensions/friction")
    for value in [args.contact_offset, args.rest_offset, args.linear_damping, args.settling_damping, args.settling_threshold, args.sleep_threshold]:
        if value is not None and not (math.isfinite(value) and 0 <= value <= 100):
            parser.error("Offset/damping must be finite in [0,100]")
    if args.contact_offset is not None and args.rest_offset is not None and args.contact_offset <= args.rest_offset:
        parser.error("contact-offset must exceed rest-offset")
    if args.experience and Path(args.experience).name != "isaacsim.exp.full.kit":
        parser.error("This protocol requires the complete official isaacsim.exp.full.kit experience")
    return args


def author_mesh(stage, path, points, elements, case, Usd, UsdGeom, Vt):
    mesh = (UsdGeom.Mesh if case == "surface" else UsdGeom.TetMesh).Define(stage, path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points.tolist()))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([points.min(axis=0).tolist(), points.max(axis=0).tolist()]))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([[0.08, 0.65, 0.85]]))
    mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    if case == "surface":
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(elements)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(elements.reshape(-1).tolist()))
        mesh.GetSubdivisionSchemeAttr().Set("none")
        mesh.GetDoubleSidedAttr().Set(True)
    else:
        mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray(elements.tolist()))
        mesh.GetSurfaceFaceVertexIndicesAttr().Set(UsdGeom.TetMesh.ComputeSurfaceFaces(mesh, Usd.TimeCode.Default()))
    return mesh


def save_static_snapshot(path, positions, elements, case, config, Usd, UsdGeom, UsdLux, Gf, Vt):
    """New stage, no copied physics APIs or misleading live simulation state."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {"artifact_kind": "static_snapshot", "protocol_id": PROTOCOL,
        "case": case, "description": "Measured final simulation nodes; visual preview only, no physics schemas"}
    faces = elements if case == "surface" else boundary_triangles(elements)
    author_mesh(stage, BODY, positions, faces, "surface", Usd, UsdGeom, Vt)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.GetSizeAttr().Set(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
    floor.AddScaleOp().Set(Gf.Vec3f(6, 6, 0.2))
    floor.CreateDisplayColorAttr(Vt.Vec3fArray([[0.18, 0.23, 0.30]]))
    if case == "surface":
        sphere = UsdGeom.Sphere.Define(stage, "/World/Obstacle")
        sphere.GetRadiusAttr().Set(0.35)
        sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.35))
        sphere.CreateDisplayColorAttr(Vt.Vec3fArray([[0.85, 0.30, 0.08]]))
    UsdLux.DistantLight.Define(stage, "/World/Light").CreateIntensityAttr(1500)
    stage.GetRootLayer().Save()
    return {"path": str(path), "kind": "static_snapshot", "physics_schemas_present": False,
            "node_source": "final simulation tensor positions", "simulation_time_step_s": 1 / config["physics_hz"]}


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.argv = [sys.argv[0]]  # SimulationApp must not forward our CLI to Kit.
    common_path = Path(__file__).with_name("isaac_platform_smoke.py")
    common = load_local("_isaac_deformable_common", common_path)
    output = (args.output_dir or PROJECT / "results/isaac-deformable-baseline" / args.case).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = (args.json or output / "report.json").resolve()
    config = {"case": args.case, "resolution_cells_per_edge": args.resolution, "steps": args.steps,
              "physics_hz": args.physics_hz, "position_iterations": args.position_iterations, "physics_device": "cuda:0", "gravity_m_s2": 0.0 if args.control == "rest" else args.gravity,
              "control": args.control, "speculative_ccd": args.speculative_ccd, "self_collision": args.self_collision,
              "max_depenetration_velocity": args.max_depenetration_velocity, "elasticity_damping": args.elasticity_damping, "tet_pattern": args.tet_pattern,
              "young_modulus_pa": args.young_modulus if args.young_modulus is not None else (50000.0 if args.case == "surface" else 20000.0),
              "poisson_ratio": args.poisson_ratio, "density_kg_m3": 1000.0, "static_friction": args.friction, "dynamic_friction": args.friction,
              "contact_offset": args.contact_offset, "rest_offset": args.rest_offset, "linear_damping": args.linear_damping,
              "settling_damping": args.settling_damping, "settling_threshold": args.settling_threshold, "sleep_threshold": args.sleep_threshold,
              "penetration_tolerance_m": 0.02, "contact_band_m": 0.04, "minimum_centroid_drop_m": 0.1,
              "minimum_deformation_rms_m": 0.00014 if args.case == "surface" else 0.00005,
              "geometry": {"floor_top_z_m": 0, "floor_size_m": [6, 6, 0.2],
                           "sheet_side_m": 1.4, "sheet_initial_z_m": 1.1,
                           "sphere_radius_m": 0.35, "sphere_center_m": [0, 0, 0.35], "sphere_is_fixed": True}
                           if args.case == "surface" else {"floor_top_z_m": 0, "floor_size_m": [6, 6, 0.2],
                           "block_side_m": args.volume_side, "block_center_m": [0, 0, args.volume_center_z]},
              "gate_scope": "gross numerical smoke; gates/illustrative material are not physical accuracy claims",
              "material_defaults": "Other PhysX schema defaults unchanged; authored schema snapshot retained"}
    required = ("source_identity", "runtime_binding", "build_receipt", "official_runtime", "scene_authoring",
                "initial_scene_export", "cuda_tensor_execution", "deformable_dynamics", "trajectory_export", "static_snapshot_export")
    report = {"schema_version": 1, "protocol_id": PROTOCOL, "case": args.case, "run_id": str(uuid.uuid4()),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "status": "running",
              "exit_code": 1, "full_platform_passed": False, "material_accuracy_verified": False,
              "clean_process_reload_verified": False, "configuration": config, "artifacts": {},
              "script_sha256": common.sha256(Path(__file__)), "common_helper_sha256": common.sha256(common_path),
              "geometry_helper_sha256": common.sha256(PROJECT / "src/deformable_sim/validation.py"),
              "checks": {name: {"status": "not_run"} for name in required},
              "limitations": ["Partial DEF-01 baseline only; no original solver or platform parity claim.",
                  "Geometric contact diagnostics do not establish force fidelity, dynamic rigid coupling or momentum conservation.",
                  "No self-collision, friction calibration, tearing, plasticity or real-material validation credit.",
                  "Initial USD is authored pre-play; final USD is a static visual snapshot. Neither is clean-process reloaded here.",
                  "Other deformable solver settings use pinned PhysX defaults; this script does not tune or infer them."]}
    app = app_utils = None
    started = time.perf_counter()

    def check(name, passed, **details):
        report["checks"][name] = {"status": "passed" if passed else "failed", **details}
        common.write_json(report_path, report)
        if not passed:
            raise RuntimeError(f"Required check failed: {name}")

    try:
        git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        git = lambda *a: subprocess.check_output(["git", "-C", str(args.source_root), *a], text=True, timeout=10, env=git_env).strip()
        commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=normal")
        check("source_identity", commit == common.EXPECTED_COMMIT and not dirty, commit=commit,
              dirty_including_untracked=bool(dirty), source_root=str(args.source_root.resolve()))
        binding = common.runtime_binding(args.source_root, os.environ.get("ISAAC_PATH"), sys.executable, args.experience)
        check("runtime_binding", binding.pop("passed"), **binding)
        receipt = common.verify_build_receipt(args.build_receipt, args.source_root, binding)
        check("build_receipt", receipt.pop("passed"), **receipt)
        if Path(binding["experience"]).name != "isaacsim.exp.full.kit":
            raise RuntimeError("Full official experience is required")
        launch = {"headless": True, "renderer": "RaytracedLighting", "multi_gpu": False,
                  "active_gpu": 0, "physics_gpu": 0, "fast_shutdown": True, "sync_loads": True,
                  "extra_args": ["--/exts/isaacsim.core.simulation_manager/default_engine=physx",
                                 "--/app/useFabricSceneDelegate=0", "--/app/extensions/pythonFullLeakCheck=false"]}
        report["launch_configuration"] = launch
        common.write_json(report_path, report)
        from isaacsim import SimulationApp
        app = SimulationApp(launch, experience=binding["experience"])
        import numpy as np
        import warp as wp
        import omni.kit.app
        import carb.settings
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.experimental.materials import SurfaceDeformableMaterial, VolumeDeformableMaterial
        from isaacsim.core.experimental.objects import Cube, Sphere, DistantLight
        from isaacsim.core.experimental.prims import DeformablePrim, GeomPrim
        from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
        from isaacsim.core.version import get_version
        from omni.physx.scripts import deformableUtils
        from pxr import Usd, UsdGeom, UsdLux, Gf, Vt

        kit = omni.kit.app.get_app()
        version = list(get_version())
        module = Path(sys.modules[SimulationApp.__module__].__file__).resolve()
        pinned = args.source_root / "source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py"
        observed_app = Path(kit.get_app_filename())
        app_matches = observed_app.name in ("isaacsim.exp.full.kit", "isaacsim.exp.full")
        if observed_app.is_absolute():
            app_matches &= observed_app.resolve() == Path(binding["experience"]).resolve()
        extension_ids = {name: kit.get_extension_manager().get_enabled_extension_id(name) for name in
                         ("omni.physx", "isaacsim.core.simulation_manager", "isaacsim.core.experimental.prims",
                          "isaacsim.core.experimental.materials")}
        report["runtime"] = {"isaac_version": version, "kit_version": kit.get_kit_version(),
            "experience": str(observed_app), "active_engine": SimulationManager.get_active_physics_engine(),
            "extension_ids": extension_ids, "warp_version": wp.__version__, "numpy_version": np.__version__,
            "simulation_app_module": str(module), "python_executable": sys.executable,
            "fast_shutdown_observed": carb.settings.get_settings().get("/app/fastShutdown")}
        check("official_runtime", version[0] == "6.1.0" and app_matches and bool(extension_ids["omni.physx"])
              and SimulationManager.get_active_physics_engine() == "physx" and common.sha256(module) == common.sha256(pinned)
              and kit.get_kit_version() == receipt["kit_version_expected"] and report["runtime"]["fast_shutdown_observed"] is True, **report["runtime"])

        points, elements = build_mesh(args.case, args.resolution, args.tet_pattern)
        if args.case == "volume":
            points = (points - [0, 0, .95]) * (args.volume_side / .5) + [0, 0, args.volume_center_z]
        report["generated_mesh"] = {"nodes": len(points), "elements": len(elements), "nodes_per_element": elements.shape[1],
            "generator": "regular triangular grid" if args.case == "surface" else "conforming regular grid: " + args.tet_pattern,
            "external_mesher": False}
        if args.case == "volume":
            v = signed_tet_volumes(points, elements)
            report["generated_mesh"].update(minimum_signed_tet_volume_m3=float(v.min()), maximum_signed_tet_volume_m3=float(v.max()),
                                            total_signed_volume_m3=float(v.sum()), positive_orientation=bool(np.all(v > 0)))
        stage_utils.create_new_stage()
        stage = stage_utils.get_current_stage(backend="usd")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetTimeCodesPerSecond(args.physics_hz)
        stage.SetDefaultPrim(stage_utils.define_prim("/World", "Xform"))
        scene = PhysxScene("/World/PhysicsScene")
        scene.set_steps_per_second(args.physics_hz)
        scene.set_gravity([0, 0, -config["gravity_m_s2"]])
        scene.set_solver_type("TGS")
        scene.set_enabled_gpu_dynamics(True)
        scene.set_broadphase_type("GPU")
        SimulationManager.set_device("cuda:0")
        floor = Cube("/World/Floor", sizes=1.0, positions=[0, 0, -0.1], scales=[6, 6, 0.2])
        floor_collision = GeomPrim("/World/Floor", apply_collision_apis=True)
        if args.control != "impact":
            stage.GetPrimAtPath("/World/Floor").GetAttribute("physics:collisionEnabled").Set(False)
        if args.case == "surface":
            obstacle = Sphere("/World/Obstacle", radii=0.35, positions=[0, 0, 0.35])
            obstacle_collision = GeomPrim("/World/Obstacle", apply_collision_apis=True)
        light = DistantLight("/World/Light")
        light.set_intensities(1500)
        mesh = author_mesh(stage, BODY, points, elements, args.case, Usd, UsdGeom, Vt)
        setup = (deformableUtils.set_physics_surface_deformable_body if args.case == "surface"
                 else deformableUtils.set_physics_volume_deformable_body)
        if not setup(stage, BODY):
            raise RuntimeError("Official PhysX deformable authoring returned false")
        if not mesh.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI"):
            raise RuntimeError("PhysX deformable solver schema missing")
        iteration_attr = mesh.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount")
        if not iteration_attr.Set(args.position_iterations) or iteration_attr.Get() != args.position_iterations:
            raise RuntimeError("Deformable position iteration setting was not authored")
        mesh.GetPrim().GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(args.speculative_ccd)
        mesh.GetPrim().GetAttribute("physxDeformableBody:selfCollision").Set(args.self_collision)
        if args.max_depenetration_velocity is not None:
            mesh.GetPrim().GetAttribute("physxDeformableBody:maxDepenetrationVelocity").Set(args.max_depenetration_velocity)
        if args.linear_damping is not None:
            assert mesh.GetPrim().GetAttribute("physxDeformableBody:linearDamping").Set(args.linear_damping)
        for name, value in [("settlingDamping", args.settling_damping), ("settlingThreshold", args.settling_threshold), ("sleepThreshold", args.sleep_threshold)]:
            if value is not None:
                assert mesh.GetPrim().GetAttribute("physxDeformableBody:"+name).Set(value)
        if args.contact_offset is not None or args.rest_offset is not None:
            assert mesh.GetPrim().ApplyAPI("PhysxCollisionAPI")
            for attr, value in [("contactOffset", args.contact_offset), ("restOffset", args.rest_offset)]:
                if value is not None:
                    assert mesh.GetPrim().GetAttribute("physxCollision:" + attr).Set(value)
        body = DeformablePrim(BODY, deformable_type=args.case, reset_xform_op_properties=False)
        material_type = SurfaceDeformableMaterial if args.case == "surface" else VolumeDeformableMaterial
        material = material_type("/World/DeformableMaterial", static_frictions=config["static_friction"],
            dynamic_frictions=config["dynamic_friction"], youngs_moduli=config["young_modulus_pa"],
            poissons_ratios=config["poisson_ratio"], densities=config["density_kg_m3"])
        body.apply_physics_materials(material)
        if args.elasticity_damping is not None:
            mat_prim = stage.GetPrimAtPath("/World/DeformableMaterial")
            assert mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
            assert mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(args.elasticity_damping)
        schemas = list(mesh.GetPrim().GetAppliedSchemas())
        required_schema = "OmniPhysicsSurfaceDeformableSimAPI" if args.case == "surface" else "OmniPhysicsVolumeDeformableSimAPI"
        check("scene_authoring", required_schema in schemas and "OmniPhysicsDeformableBodyAPI" in schemas,
              applied_schemas=schemas, deformable_type=body.deformable_type,
              material_schemas=list(stage.GetPrimAtPath("/World/DeformableMaterial").GetAppliedSchemas()))
        initial = output / "initial.usda"
        check("initial_scene_export", bool(stage.Export(str(initial))), path=str(initial),
              meaning="authored initial physics scene before playback; not yet clean-process reloaded")
        snapshot = common.authored_snapshot(stage, UsdGeom)
        manifest = output / "initial.manifest.json"
        common.write_json(manifest, {"protocol_id": PROTOCOL, "run_id": report["run_id"], "configuration": config,
            "source_commit": commit, "build_receipt_sha256": receipt["receipt_sha256"],
            "scene_sha256": common.sha256(initial), "authored_snapshot": snapshot})
        report["artifacts"].update(initial_scene=str(initial), initial_manifest=str(manifest))

        app_utils.play()
        app.update()  # initialize actual tensor entities; recorded but excluded from timed loop
        device = SimulationManager.get_device()
        if not body.is_physics_tensor_entity_valid():
            raise RuntimeError("No real PhysX deformable tensor entity after playback initialization")
        tensor_devices = {}

        def host(array, name):
            tensor_devices[name] = {"device": str(array.device), "is_cuda": bool(array.device.is_cuda),
                                    "shape": list(array.shape), "dtype": str(array.dtype)}
            if not array.device.is_cuda or str(array.device) != "cuda:0":
                raise RuntimeError(f"{name} is not a CUDA array: {array.device}")
            return array.numpy()[0].copy()  # synchronous actual readback, never fabricated geometry

        with use_backend("tensor", raise_on_fallback=True, raise_on_unsupported=True):
            element_arrays = body.get_element_indices()
        topologies = [host(array, name + "_elements") for name, array in zip(("simulation", "collision", "rest"), element_arrays)]
        states, collisions, velocities, times, steps = [], [], [], [], []
        rest_initial = None

        def sample():
            nonlocal rest_initial
            with use_backend("tensor", raise_on_fallback=True, raise_on_unsupported=True):
                position_arrays = body.get_nodal_positions()
                velocity_array = body.get_nodal_velocities()
            simulation, collision, rest = [host(array, name + "_positions") for name, array in
                zip(("simulation", "collision", "rest"), position_arrays)]
            velocity = host(velocity_array, "simulation_velocities")
            states.append(simulation)
            collisions.append(collision)
            velocities.append(velocity)
            times.append(float(SimulationManager.get_simulation_time()))
            steps.append(int(SimulationManager.get_num_physics_steps()))
            for name, value, topology in zip(("simulation", "collision", "rest"), (simulation, collision, rest), topologies):
                checked_points(value)
                checked_elements(topology, len(value), 3 if args.case == "surface" else 4)
            checked_points(velocity)
            if rest_initial is None:
                rest_initial = rest.copy()
            elif not np.array_equal(rest, rest_initial):
                raise RuntimeError("Rest-node array changed during this elastic baseline")

        sample()
        runtime = {"physics_device": str(device), "physics_device_name": device.name,
            "gpu_dynamics": bool(scene.get_enabled_gpu_dynamics()), "broadphase": scene.get_broadphase_type(),
            "rigid_scene_solver_type": scene.get_solver_type(), "physics_dt_s": scene.get_dt(),
            "deformable_solver": "PhysX surface/volume deformable implementation; internal solver not inferred from rigid TGS setting",
            "fabric_enabled": SimulationManager.is_fabric_enabled(), "array_evidence": tensor_devices,
            "simulation_mesh_paths": body.simulation_mesh_paths, "collision_mesh_paths": body.collision_mesh_paths,
            "num_nodes_per_body": list(body.num_nodes_per_body), "num_elements_per_body": list(body.num_elements_per_body),
            "warmup_physics_step": steps[0], "warmup_simulation_time_s": times[0]}
        report["runtime"].update(runtime)
        check("cuda_tensor_execution", bool(device.is_cuda) and str(device) == "cuda:0"
              and runtime["gpu_dynamics"] and runtime["broadphase"] == "GPU"
              and abs(scene.get_dt() - 1 / args.physics_hz) < 1e-9
              and body.deformable_type == args.case, **runtime)

        loop_error = None
        wp.synchronize_device(device)
        physics_start = time.perf_counter()
        try:
            for _ in range(args.steps):
                SimulationManager.step(update_fabric=SimulationManager.is_fabric_enabled())
                sample()
        except Exception as error:
            loop_error = f"{type(error).__name__}: {error}"
        wp.synchronize_device(device)
        elapsed = time.perf_counter() - physics_start
        trajectory_path = output / "trajectory.npz"
        np.savez_compressed(trajectory_path, simulation_positions=np.asarray(states), collision_positions=np.asarray(collisions),
            nodal_velocities=np.asarray(velocities), simulation_elements=topologies[0], collision_elements=topologies[1],
            rest_elements=topologies[2], rest_positions=rest_initial, authored_positions=points, authored_elements=elements,
            simulation_time_s=np.asarray(times), physics_step=np.asarray(steps))
        report["artifacts"]["trajectory"] = str(trajectory_path)
        check("trajectory_export", trajectory_path.is_file(), path=str(trajectory_path), sha256=common.sha256(trajectory_path),
              samples=len(states), contains_actual_node_arrays=True)
        assessment = (assess_trajectory(args.case, states, collisions, velocities, topologies[0], topologies[1], times, steps, config)
                      if args.control == "impact" else assess_control(states, velocities, topologies[0], times, steps, config))
        scalar_path = output / "trajectory-metrics.json"
        common.write_json(scalar_path, {"samples": assessment.pop("scalar_trajectory", []), "loop_error": loop_error})
        report["artifacts"]["trajectory_metrics"] = str(scalar_path)
        dynamics_passed = assessment.pop("passed") and loop_error is None
        report["checks"]["deformable_dynamics"] = {"status": "passed" if dynamics_passed else "failed", **assessment,
            "loop_error": loop_error, "elapsed_seconds": elapsed, "sampled_steps": len(states) - 1,
            "timing_scope": "synchronized PhysX steps plus Python/tensor queries/host readback; excludes startup and file export"}
        common.write_json(report_path, report)
        if states and np.isfinite(states[-1]).all():
            static = output / "final-static.usda"
            preview = save_static_snapshot(static, states[-1], topologies[0], args.case, config, Usd, UsdGeom, UsdLux, Gf, Vt)
            report["artifacts"]["final_static_snapshot"] = str(static)
            check("static_snapshot_export", static.is_file(), **preview, sha256=common.sha256(static))
        else:
            check("static_snapshot_export", False, reason="Final actual state is invalid; no substitute preview generated")
    except Exception as error:
        report["fatal_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        for name in required:
            if report["checks"][name]["status"] == "not_run":
                report["checks"][name] = {"status": "failed", "execution_status": "not_run",
                    "reason": "Required phase could not execute; see fatal_error. No skipped-pass credit."}
        passed = all(report["checks"][name]["status"] == "passed" for name in required) and "fatal_error" not in report
        report.update(status="pending_shutdown" if app is not None else "failed", exit_code=1,
                      shutdown_status="pending" if app is not None else "not_started",
                      elapsed_seconds_before_shutdown=time.perf_counter() - started)
        common.write_json(report_path, report)
        if app is not None:
            stop_error = None
            try:
                if app_utils is not None:
                    app_utils.stop()
            except Exception as error:
                stop_error = str(error)
                report["shutdown_stop_error"] = stop_error
            # Always attempt close, including after a failed timeline stop.
            try:
                passed = passed and stop_error is None
                report.update(status="pending_official_fast_exit", shutdown_status="official_fast_exit_requested",
                              shutdown_full_cleanup_verified=False, exit_code=0 if passed else 1)
                common.write_json(report_path, report)
                app.close(exit_code=report["exit_code"])
                report.update(shutdown_status="completed", status="passed_for_selected_baseline_case" if passed else "failed",
                              exit_code=0 if passed else 1)
            except Exception as error:
                report.update(status="failed", shutdown_status="failed", shutdown_error=str(error), exit_code=1)
            report["elapsed_seconds_total"] = time.perf_counter() - started
            common.write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "case": args.case, "status": report["status"],
                      "exit_code": report["exit_code"], "full_platform_passed": False}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
