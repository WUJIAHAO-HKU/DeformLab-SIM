"""Geometry diagnostics and falsifiable verdicts for isolated contact tests."""
from __future__ import annotations

import math
import numpy as np


def maximum_component_spread(values: list[list[float]]) -> float:
    """Return the largest max-minus-min span across repeated vector components."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError('Expected a nonempty finite matrix of repeated vectors.')
    return float(np.ptp(array, axis=0).max())


def point_triangle_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    """Return the minimum point-to-triangle distance [m], including edges."""
    point = np.asarray(point, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.shape[0] == 0:
        raise ValueError('Expected at least one triangle.')
    if not np.isfinite(point).all() or not np.isfinite(triangles).all():
        raise ValueError('Non-finite geometry.')
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    denominator = d00 * d11 - d01 * d01
    valid = denominator > 1e-24
    u = np.divide(d11 * dot(ap, ab) - d01 * dot(ap, ac), denominator, out=np.zeros_like(denominator), where=valid)
    v = np.divide(d00 * dot(ap, ac) - d01 * dot(ap, ab), denominator, out=np.zeros_like(denominator), where=valid)
    inside = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    normal = np.cross(ab, ac)
    plane_distance2 = np.divide(dot(ap, normal) ** 2, dot(normal, normal), out=np.full_like(denominator, np.inf), where=valid)
    distances2 = np.where(inside, plane_distance2, np.inf)
    for start, end in [(a, b), (b, c), (c, a)]:
        edge = end - start
        length2 = dot(edge, edge)
        t = np.clip(np.divide(dot(point - start, edge), length2, out=np.zeros_like(length2), where=length2 > 0), 0, 1)
        delta = point - (start + t[:, None] * edge)
        distances2 = np.minimum(distances2, dot(delta, delta))
    return float(np.sqrt(max(0.0, distances2.min())))


def assess_case(report: dict, thresholds: dict) -> dict:
    """Separate evidence of response from numerical-quality gates."""
    samples = report['samples']
    required = ['rigid_velocity_m_s', 'cloth_com_velocity_m_s', 'total_momentum_kg_m_s',
                'relative_momentum_drift', 'vertex_envelope_penetration_m',
                'sphere_ballistic_position_error_m', 'sphere_ballistic_velocity_error_m_s',
                'maximum_cloth_displacement_m']
    finite = bool(samples) and all(s.get('finite', False) for s in samples)
    for sample in samples:
        for name in required:
            value = sample.get(name)
            if value is None or not np.isfinite(np.asarray(value, dtype=float)).all():
                finite = False
    if not finite:
        return {'passed': False, 'checks': {'finite_trajectory': False}}
    contacts = max(s['soft_contact_count'] for s in samples)
    checks = {'finite_trajectory': True, 'contact_buffer_not_saturated': contacts < report['model']['soft_contact_capacity']}
    if report['case']['contact']:
        checks.update({
            'contacts_observed': contacts > 0,
            'momentum_within_gate': max(s['relative_momentum_drift'] for s in samples) <= thresholds['max_relative_momentum_drift'],
            'vertex_envelope_within_gate': max(s['vertex_envelope_penetration_m'] for s in samples) <= thresholds['max_vertex_envelope_penetration_m'],
        })
    else:
        checks.update({
            'contacts_disabled': contacts == 0,
            'ballistic_position': max(s['sphere_ballistic_position_error_m'] for s in samples) <= thresholds['off_position_error_m'],
            'ballistic_velocity': max(s['sphere_ballistic_velocity_error_m_s'] for s in samples) <= thresholds['off_velocity_error_m_s'],
            'cloth_stays_at_rest': max(s['maximum_cloth_displacement_m'] for s in samples) <= thresholds['off_cloth_displacement_m'],
            'cloth_velocity_stays_at_rest': max(float(np.linalg.norm(s['cloth_com_velocity_m_s'])) for s in samples) <= thresholds['off_velocity_error_m_s'],
        })
    return {'passed': all(checks.values()), 'checks': checks}


def assess_pair(on: dict, off: dict, thresholds: dict) -> dict:
    """Require matched controls and opposite endpoint responses from both systems."""
    if not on.get('samples') or not off.get('samples'):
        return {'passed': False, 'checks': {'nonempty_pair': False}}
    a, b = on['samples'][-1], off['samples'][-1]
    rigid_delta = float(a['rigid_velocity_m_s'][2] - b['rigid_velocity_m_s'][2])
    cloth_delta = float(b['cloth_com_velocity_m_s'][2] - a['cloth_com_velocity_m_s'][2])
    finite = math.isfinite(rigid_delta) and math.isfinite(cloth_delta)
    checks = {
        'correct_on_off_roles': on['case']['contact'] is True and off['case']['contact'] is False,
        'both_executions_finite': assess_case(on, thresholds)['checks'].get('finite_trajectory', False) and assess_case(off, thresholds)['checks'].get('finite_trajectory', False),
        'matched_initial_physical_state': on['initial_state_sha256'] == off['initial_state_sha256'],
        'matched_duration': abs(a['time_s'] - b['time_s']) < 1e-10,
        'valid_off_control': assess_case(off, thresholds)['passed'],
        'on_contacts_observed': max(s['soft_contact_count'] for s in on['samples']) > 0,
        'rigid_response_above_gate': finite and rigid_delta >= thresholds['final_rigid_vz_increase_m_s'],
        'cloth_response_above_gate': finite and cloth_delta >= thresholds['final_cloth_vz_decrease_m_s'],
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'final_rigid_vz_increase_m_s': rigid_delta,
            'final_cloth_vz_decrease_m_s': cloth_delta}
