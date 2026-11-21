"""Recompute the independent Franka and search-schedule confirmations."""
from pathlib import Path
import collections
import json
import math
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data/confirmation-66'


def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def bootstrap(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261119)
    samples = a[rng.integers(0, len(a), size=(100000, len(a)))].mean(axis=1)
    return {'paired_n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'percentile95': np.quantile(samples, [.025, .975]).tolist(), 'unit': 'fraction'}


def wilson(k, n):
    z = 1.959963984540054
    p = k / n; den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0., center - half), min(1., center + half)]


def pairs(study):
    rows = [x for x in read('66-runs.json') if x['study'] == study]
    cells = collections.defaultdict(dict)
    for row in rows:
        cells[row['cell']][row['method']] = row
    result = []
    for published in read('reported-statistics.json')['paired']:
        if published['study'] != study:
            continue
        first, second = published['first'], published['reference']
        common = [(x[first], x[second]) for x in cells.values()]
        out = {'first': first, 'reference': second, 'physical_paired_n': len(common)}
        for key in ('loop_s', 'data_ready_s'):
            eligible = [(a, b) for a, b in common if a.get(key) is not None and b.get(key) is not None and a['timing_valid'] and b['timing_valid']]
            if eligible:
                measured = bootstrap([1 - a[key] / b[key] for a, b in eligible])
                reference = published[key + '_relative_reduction']
                assert measured['paired_n'] == reference['paired_n']
                assert np.allclose([measured['mean'], *measured['percentile95']], [reference['mean'], *reference['percentile95']], atol=1e-12, rtol=0)
                out[key + '_relative_reduction'] = measured
        result.append(out)
    return result


def franka_confirmation():
    inputs = read('franka-score-inputs.json')
    records = [x for x in read('66-runs.json') if x['study'] == 'franka']
    assert len(inputs) == len(records) == 36
    names = {'B': 'Consistent State Mapping', 'D': 'DeformLabSim', 'Dense': 'Full Boundary Contact'}
    tolerance_counts = {str(t): {m: 0 for m in names.values()} for t in (0., .05, .1, .25, .5, 1.)}
    verified = []
    for row in records:
        name = f"{row['cell']}-{row['method_id']}"
        r = inputs[name]; slot = r['slot']
        with np.load(DATA / 'franka-trajectories' / f'{name}.npz', allow_pickle=False) as z:
            t = z['time_s']; x = z['centroid_m']; hand = z['hand_xyz']; low = z['low_m']
            final = t >= 9.5; lift = (t >= 3.7) & (t <= 5.); contact = (t >= 5.) & (t <= 9.5)
            rel = x - hand
            drift = float(np.linalg.norm(rel[lift] - rel[lift][0], axis=1).max())
            clear = float((low[lift] - slot['support_top_m']).min())
            carry = float(x[np.argmin(abs(t - 5.)), 0] - x[0, 0])
            margin = float(z['margin_m'][final].min())
            jac = float(z['minimum_J'].min())
            rail = float(z['rail_penetration_m'][contact].max())
            speed = float(np.quantile(np.linalg.norm(np.diff(x[final], axis=0), axis=1) * 960, .95))
            height = float(np.abs(low[final] - slot['support_top_m']).max())
            hand_clear = float((hand[final, 2] - z['top_m'][final]).min())
            finger = float(z['finger_positions'][final].min())
            criteria = dict(unsupported_pick=clear > .035, retained_in_gripper=drift < .007,
                carried_to_slot=carry > .04, whole_object_in_opening=margin >= 0,
                deposited_on_floor=height <= .001, released_and_retreated=finger >= .038 and hand_clear >= .1,
                stable_after_withdrawal=speed < .025, valid_elements=jac > 0, rail_penetration_below_1mm=rail <= .001)
            err = float(np.median(np.linalg.norm(x[:, :2] - np.asarray(slot['center_xy']), axis=1)[final]))
        assert criteria == r['task_result']['criteria'], name
        assert bool(all(criteria.values())) == row['complete_task_success']
        assert np.isclose(1000 * err, row['center_error_mm'], atol=1e-8, rtol=0)
        assert np.isclose(1e6 * margin, row['final_minimum_clearance_um'], atol=1e-8, rtol=0)
        for tol in tolerance_counts:
            fixed = all(v for k, v in criteria.items() if k != 'whole_object_in_opening')
            tolerance_counts[tol][row['method']] += int(fixed and margin >= -float(tol) / 1000)
        verified.append(row)
    grouped = {}
    for name in names.values():
        g = [r for r in verified if r['method'] == name]
        k = sum(r['complete_task_success'] for r in g)
        grouped[name] = dict(n=len(g), successes=k, wilson95=wilson(k, len(g)),
            center_error_mm_median=float(np.median([r['center_error_mm'] for r in g])),
            clearance_mm_median=float(np.median([r['final_minimum_clearance_um'] / 1000 for r in g])))
    cells = collections.defaultdict(dict)
    for row in verified:
        cells[row['cell']][row['method_id']] = row
    work = [1 - x['D']['supplemental_row_rounds'] / x['Dense']['supplemental_row_rounds'] for x in cells.values()]
    assert [grouped[n]['successes'] for n in names.values()] == [5, 12, 12]
    assert list(tolerance_counts['0.1'].values()) == [7, 12, 12]
    return dict(groups=grouped, tolerance_successes=tolerance_counts, pairs=pairs('franka'),
                supplemental_constraint_reduction_range=[min(work), max(work)], physical_executions=36)


def refresh_confirmation():
    rows = [x for x in read('66-runs.json') if x['study'] == 'refresh']
    assert len(rows) == 30
    grouped = {}
    for name in dict.fromkeys(r['method'] for r in rows):
        g = [r for r in rows if r['method'] == name]
        assert len(g) == 6
        assert all(r['complete_task_success'] and all(r['criteria'].values()) for r in g)
        assert all(r['gauge_penetration_um'] == 0 and r['minimum_J'] > 0 for r in g)
        for r in g:
            assert np.isclose(r['loop_s'], r['cost']['loop_wall_s'], atol=1e-9, rtol=0)
            assert np.isclose(r['data_ready_s'], r['cost']['generation_ready_wall_s'], atol=1e-9, rtol=0)
            assert np.isclose(r['refresh_fraction'], r['refresh_counters']['refresh_fraction'], atol=1e-12, rtol=0)
        grouped[name] = dict(n=6, successes=6, refresh_fraction_median=float(np.median([r['refresh_fraction'] for r in g])),
                            loop_s_median=float(np.median([r['loop_s'] for r in g])),
                            data_ready_s_median=float(np.median([r['data_ready_s'] for r in g])))
    return dict(groups=grouped, pairs=pairs('refresh'), physical_executions=30)
