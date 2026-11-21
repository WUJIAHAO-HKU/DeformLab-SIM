"""Recompute the placement and thickness-control results from case records."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

DATA = Path(__file__).resolve().parents[1] / 'data'


def read(path):
    return json.loads((DATA / path).read_text(encoding='utf-8'))


def continuous_placement():
    rows = read('placement/per-case-results.json')
    reference = read('placement/descriptive-results.json')['strata']
    assert len(rows) == 36
    grouped = defaultdict(list)
    for row in rows:
        assert row['execution_valid'], row['case_id']
        grouped[row['condition'], row['arm']].append(row)
    output = {}
    for (condition, arm), group in sorted(grouped.items()):
        assert len(group) == 3
        passes = sum(row['placement_passed'] for row in group)
        regrasp = sum(row['regrasp_passed'] for row in group)
        error = median(row['measurements']['placement_error_m'] for row in group)
        expected = reference[condition][arm]
        assert passes == expected['placement'] and regrasp == expected['regrasp']
        assert abs(error - expected['metrics']['placement_error_m']['median']) < 1e-10
        output.setdefault(condition, {})[arm] = dict(placement=passes, regrasp=regrasp, median_placement_error_m=error)
    for values in output.values():
        baseline = values['B']['median_placement_error_m']
        values['deformlabsim_reduction_pct'] = 100 * (1 - values['D']['median_placement_error_m'] / baseline)
    return output


def thickness_control():
    rows = read('thickness/36-runs.json')
    reference = {(row['observer'], row['hidden_bias_m'], row['arm']): row['median_m']
                 for row in read('thickness/reported-statistics.json')['groups']}
    assert len(rows) == 36
    grouped = defaultdict(list)
    for row in rows:
        assert row['execution_valid'], row['name']
        grouped[row['observer'], row['hidden_bias_m'], row['arm']].append(row)
    output = []
    for (observer, bias_m, arm), group in sorted(grouped.items()):
        assert len(group) == 2
        thickness = median(row['metrics']['hold']['thickness_m']['mean_absolute_error_m'] for row in group)
        penetration = median(row['metrics']['hold']['bench_nodal_penetration_m'] for row in group)
        assert abs(thickness - reference[observer, bias_m, arm]) < 1e-10
        output.append(dict(observer=observer, bias_m=bias_m, arm=arm, trials=len(group),
                           median_hold_thickness_error_m=thickness, median_hold_support_penetration_m=penetration))
    return output
