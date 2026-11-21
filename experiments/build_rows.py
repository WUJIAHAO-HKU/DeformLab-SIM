"""Compile the recorded schedule variant to PTX using a local CUDA toolkit."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'data/confirmation-66/protocol.json').read_text(encoding='utf-8'))
    schedules = {item['id']: item for item in protocol['refresh']['methods']}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material', choices=('c0', 'c1'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--schedule', choices=schedules)
    group.add_argument('--franka-policy', choices=('sparse', 'dense'))
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    native = root / 'experiments/native'
    flags = ['-DCOMPONENT_ROW_DIAGNOSTICS=0']
    if args.franka_policy:
        source = native / 'franka/gpu-source/robot_surface_parallel_rows.cu'
        flags.append(f'-DCOMPONENT_DENSE_POLICY={int(args.franka_policy == "dense")}')
    else:
        if args.material is None:
            parser.error('--schedule requires --material c0 or c1')
        source = native / 'gpu-source/robot_surface_parallel_rows.cu'
        schedule = schedules[args.schedule]
        flags.extend([f'-I{native / "material-reference" / args.material}', '-DCOMPONENT_DENSE_POLICY=0',
                      '-DCOMPONENT_DISCOVERY_PERIOD=1', '-DCOMPONENT_UNIFORM_DENOMINATOR=1',
                      f'-DEVENT_MODE={schedule["mode"]}',
                      f'-DEVENT_DISABLE_MATERIAL_RESPONSE={schedule["disable_material"]}',
                      f'-DEVENT_DISABLE_PREFINAL={schedule["disable_prefinal"]}'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.nvcc, '-ptx', f'-arch={args.arch}', '-O3', *flags,
                    str(source), '-o', str(args.out)], check=True)


if __name__ == '__main__':
    main()
