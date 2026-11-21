# Reproduction guide

## Offline scoring

Use the root README commands. `python -B code/reproduce.py --help` lists the available studies. All paths resolve relative to the checked-out source. Optional material-reference assembly requires SciPy. [FigureIndex.csv](../FigureIndex.csv) gives the dataset and command for each measured figure or table.

- `franka_confirmation` scores all 36 trials and applies the original task and overflow-tolerance rules to each trajectory. It recomputes success and center error, relative constraint counts, and the paired loop-time interval.
- `refresh_confirmation` audits all 30 measured schedule executions and recomputes search rates and paired loop and data-ready time. `franka` and `franka_tolerance_sensitivity` refer to the earlier 18-run demonstration cohort.
- `gauge`, `discovery`, `scaling`, `heldout`, and `shared_cap` reproduce the corresponding bench, mechanism, and runtime summaries. `physical_accounting` checks momentum and work records. `reported_shape` exposes the archived independent shape comparison at its reported aggregate resolution.
- `continuous_placement` scores the 36 earlier visual placement and regrasp trials in `data/placement/`. `thickness_control` computes the thickness and support-penetration summaries for the 36 trials in `data/thickness/`.

The two independent confirmation cohorts use 100,000 bootstrap draws with seed 20261119. Draws resample physical case pairs. Completion scores retain every formal execution. Runtime pairs require eligible resource measurements from both runs. Timing values are elapsed wall time in seconds, and the archive preserves the study-specific timing intervals.

## Full trajectory archives

Place the ZIP downloads from the [RA-L data release](https://github.com/WUJIAHAO-HKU/DeformLab-SIM/releases/tag/ral-2026.11) in `downloads/`. Verify them against the manifest.

```sh
python -B code/verify_downloads.py downloads
```

Each ZIP can be extracted independently. The first directory component is the study and the second is the paired case and method. Franka trajectories contain time, node positions, rest nodes, tetrahedra, measured joints and hand poses. Schedule trajectories contain the per-step mesh, velocities and counters, separate gauge motion, and the aligned observation/action sequences. Read NPZ data with `numpy.load(..., allow_pickle=False)`. Earlier bench cohorts are preserved at the resolution listed in the figure index, typically as per-trial metrics, traces, or physical ledgers.

## Simulator sources

The paper used a Linux NVIDIA workstation with Isaac Sim 6.1.0 and CUDA 12.6. The independent backend used Newton 1.5.2 and Warp 1.16.0. Install the upstream simulator and assets according to their documentation. Place the Isaac Sim source and build under `runtime/upstream/IsaacSim` when using the original project layout, or set the runtime options explicitly where an entry point supports them.

The `experiments/native/gpu-source/` files and material references correspond to the confirmation campaign. Compile the numerical row kernels with the locally installed CUDA toolkit.

```sh
python -B experiments/build_rows.py --material c0 --schedule EveryRound --arch sm_89 --out outputs/rows.ptx
```

For a fresh Franka trial, supply an Isaac Sim `python.sh`, a verified runtime build receipt, and the matching compiled adapter library and PTX. Run the following on Linux with the relevant project assets installed.

```sh
DEFORMLABSIM_ROOT="$PWD/runtime" bash experiments/run_franka.sh F01 D
```

Set `ISAAC_PYTHON`, `ISAAC_BUILD_RECEIPT`, and `CONTACT_BUILD_DIR` before running. `CONTACT_BUILD_DIR` must contain `sparse/` and `dense/`, each with `libphysx_robot_surface.so` and `rows.ptx`. Use `MethodNames.json` to interpret the preserved backend IDs. `experiments/franka/cases/` and `experiments/refresh/cases/` record the paired inputs. The gauge driver and original runtime sources are provided for inspection and rebuilding against the version-matched SDK. The local build receipt, generated native binaries, and simulator are prerequisites for a fresh rollout. The public release was verified by rescoring the archived trials. GPU rebuilding and fresh rollout validation require that separate installation.
