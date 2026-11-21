# DeformLabSim

DeformLabSim augments support contact during iterative deformable simulation. It adds constraints at boundary nodes that penetrate a finite stationary support and retains each response by physical node identity. The experiments measure task completion, contact work, and elapsed time under matched execution settings.

In the 36-run independent Franka study, Consistent State Mapping completed 5 of 12 placements. DeformLabSim and Full Boundary Contact both completed 12 of 12. DeformLabSim used a mean 1.86% less loop time than Full Boundary Contact across ten timing-eligible pairs. Identical-command replays at 2,197 nodes used 24.8–26.4% less loop time.

## Reproduce the reported results

Use Python 3.10 or later. Run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -B code/reproduce.py --output reproduced-results.json
python -B code/reproduce.py --study franka_confirmation
python -B code/reproduce.py --study refresh_confirmation
```

The offline commands use archived observations, solver counters, geometry features, and timing records. They reproduce the task scores and paired statistics without Isaac Sim or a GPU. See [Reproduction guide](docs/REPRODUCE.md) for the experiment index, raw downloads, and simulator requirements.

## Repository layout

- `data/` holds the reported numerical inputs, case definitions, and per-run results.
- `code/` holds the offline scoring, aggregation, and contact policy checks.
- `experiments/` holds the paper-specific native sources, configuration, and experiment drivers.
- `runtime/` holds the platform's custom task and native integration sources. Obtain Isaac Sim and third-party assets separately under their upstream terms.
- [FigureIndex.csv](FigureIndex.csv) maps paper figures and tables to reproducible measurements.

## Full trajectories

Download the numerical archives from the [RA-L data release](https://github.com/WUJIAHAO-HKU/DeformLab-SIM/releases/tag/ral-2026.11). Each archive contains a member manifest. The [asset index](data/raw-data-assets.json) gives archive sizes and SHA-256 checksums. The raw release contains the full numerical trajectories for the 66 independent confirmation runs and the original 18 Franka demonstration runs. The schedule experiments include synchronized RGB-D and gauge trajectories.

## Third-party components

Version pins and notices are in [`runtime/upstream-lock.json`](runtime/upstream-lock.json) and [`runtime/THIRD_PARTY_NOTICES.md`](runtime/THIRD_PARTY_NOTICES.md). The Newton-derived examples retain their upstream license. The native experiment interface requires the version-matched simulator and a locally built adapter. See the reproduction guide for the distinction between offline verification and a fresh GPU run.
