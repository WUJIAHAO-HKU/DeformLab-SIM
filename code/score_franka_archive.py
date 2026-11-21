"""Rescore a released Franka ZIP using its full mesh and robot trajectories."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from transfer_task import score

DATA = Path(__file__).resolve().parents[1] / 'data'
INDEX = {'franka': 'confirmation-66/franka-score-inputs.json',
         'franka-demo': 'franka/score-inputs.json'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    outcomes = {}
    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.endswith('/trajectory.npz')]
        studies = {name.split('/')[0] for name in members}
        if len(studies) != 1 or not studies <= INDEX.keys():
            parser.error('Provide one of the two released Franka trajectory ZIPs.')
        index = json.loads((DATA / INDEX[studies.pop()]).read_text(encoding='utf-8'))
        if sorted(name.split('/')[1] for name in members) != sorted(index):
            parser.error('The ZIP does not contain all expected cases exactly once.')
        for name in sorted(members):
            case = name.split('/')[1]
            slot = index[case]['slot']
            expected = index[case]['task_result']
            rails = [(np.asarray(rail['center']), np.asarray(rail['full_size'])) for rail in slot['rails']]
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                outcome, _ = score(data['time_s'], data['volume_nodes'], data['hand_poses'],
                                   data['joint_positions'], data['minimum_J'], data['volume_nodes'][0].mean(axis=0),
                                   slot['support_top_m'], np.asarray(slot['center_xy']),
                                   np.asarray(slot['half_opening_m']), rails)
            assert outcome['criteria'] == expected['criteria'], case
            assert outcome['success'] == expected['success'], case
            for metric in ('final_center_error_m', 'final_minimum_margin_m'):
                assert np.isclose(outcome[metric], expected[metric], atol=1e-9, rtol=0), (case, metric)
            outcomes[case] = outcome
    print(json.dumps(outcomes, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
