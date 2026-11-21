# Third-party notices

`scripts/smoke_cloth.py` adapts the model and stepping sequence from Newton v1.5.2
`newton/examples/cloth/example_cloth_hanging.py` (Copyright (c) 2025 The Newton Developers),
under Apache License 2.0. The local adaptation adds configuration, instrumentation,
numerical smoke checks, and artifact export. The original solver is unchanged.

The license is preserved in `third_party/newton/LICENSE.md`.
Upstream URL: https://github.com/newton-physics/newton/tree/v1.5.2

The separate upstream reference checkouts retain their own licenses.
See `upstream-lock.json` for source commits and `requirements-baseline.lock` for packages.

`scripts/smoke_rigid.py` adapts the scene and stepping from Newton v1.5.2
`newton/examples/basic/example_basic_shapes.py` (Copyright (c) 2025 The Newton Developers),
also under Apache License 2.0. The local adaptation adds configuration, trajectory
recording, timing, and numerical smoke checks. See the same preserved license above.


The native desktop in `desktop/` uses PySide6 6.10.2 and VTK 9.5.2.
Its component sources, preserved license texts and build process are described
in `desktop/THIRD_PARTY.txt`, `desktop/licenses/`, and `desktop/build.py`.
The workbench reuses the public Newton 1.5.2 model/solver stepping APIs used by
the recorded baselines; its GPU solver is unchanged. Isaac Sim is a reference
checkout and is not included in the native desktop executable.
