# pet-simulation: development plan

A unified Python pipeline for PET simulation that wraps MCGPU-PET and
GATE 10 as backends behind a simulator-agnostic Run API.

## Status

Phase 1 (core data model): **complete**. 151 tests.
Phase 2a (MCGPU-PET input writers): **complete**. +20 tests.
Phase 2b (MCGPU-PET execution layer): **complete**. +19 tests.
Phase 3 (GATE backend): **complete**. +20 tests.

Total: **211 tests passing** in the offline test environment.

Remaining manual validation on the simulation server:

  1. Byte-exact diff of `write_vox` output against the distributed
     MCGPU-PET sample (`phantom_9x9x9cm.vox`). The 3 hot-spot voxels
     in the sample are at unknown positions; either locate them in
     the sample file or reproduce the all-air slices alone.
  2. End-to-end MCGPU-PET run via `MCGPUBackend.run_full()` on the
     sample scene. Expect ~4.8 s wall time and ~3710 trues with the
     default seed.
  3. End-to-end GATE run via `GATEBackend.run_full()` on the same
     scene. Expect agreement within ~10% (per the MCGPU-PET paper).

## Architecture

```
petsim/
├── phantom.py     Phantom: voxel geometry + materials (float64 densities)
├── source.py      Source: activity distribution (float64 activity_Bq)
├── scanner.py     Scanner: geometry + binning + forward-model fields
├── sinogram.py    Sinogram: trues/scatter/randoms container
├── materials.py   MaterialRegistry: names → ICRP 110 / GATE names
├── run.py         Run: bundle all + manifest + save/load orchestration
└── backends/
    ├── __init__.py
    ├── mcgpu.py           write_vox (byte-exact) + write_in (structural)
    ├── mcgpu_runtime.py   MCGPUBackend: stage + invoke + parse
    └── gate.py            GATEBackend: opengate API + .mhd I/O
```

## Public API

Simulator-agnostic:

```python
from petsim import Phantom, Source, Scanner, Run

phantom = Phantom.cube(shape=(9,9,9), voxel_size=(1.0,1.0,1.0), ...)
source = Source.with_total_activity(phantom, material="water",
                                    total_activity_Bq=1e6)
scanner = Scanner.from_preset("mcgpu_sample")
run = Run(phantom=phantom, source=source, scanner=scanner, seed=42)
```

MCGPU-PET:

```python
from petsim.backends import MCGPUBackend

backend = MCGPUBackend(
    executable="./MCGPU-PET.x",
    materials_dir="./MCGPU-PET/sample_simulation/materials",
)
sinogram, result = backend.run_full(run, workdir="/tmp/my_run")
```

GATE 10:

```python
from petsim.backends import GATEBackend

backend = GATEBackend()
sinogram, result = backend.run_full(run, workdir="/tmp/my_gate_run")
```

## Known deferred work

- **Byte-exact `.in` reproduction**: infeasible due to sample file
  quirks (placeholder isotope mean life of 70000 s, dose-ROI values
  leftover from a different phantom). `write_in` produces a
  functionally correct file that MCGPU-PET accepts.
- **Caching of `.vox` writes**: writing the 9×9×9 sample takes <1 ms
  but large phantoms (e.g. 256³) will be slow. Cache keyed on
  `(phantom.hash, source.hash)` is straightforward to add when needed.
- **LOR rebinning across backends**: MCGPU-PET's span-compressed
  sinogram and GATE's raw sinogram may have different axial extents.
  The current design surfaces this via `Scanner.binning_convention` and
  raises clearly when a mismatch is detected. A real rebinner
  (SSRB, FORE) is out of scope for Phase 3.
- **GATE ring repeater**: the `opengate` ring-placement API varies by
  version. The current `gate_ring_repeater()` returns a descriptor
  dict that users tweak for their installed version. This should be
  pinned once you've confirmed the exact call signature on your
  simulation server.
- **Multi-ring scanner templates for GATE**: the current GATE backend
  builds a single-ring detector. Clinical-grade templates (Biograph,
  Vision, etc.) can be added as presets without API changes.

## Design decisions (condensed)

- `Run.seed` is a first-class field, not metadata.
- `Scanner.binning_convention` / `tof_enabled` / `normalization`
  make the forward model explicit. `is_compatible_with()` checks
  two scanners before reusing a sinogram across runs.
- Coordinate convention (documented in `phantom.py`): origin at
  corner voxel, axes +x/+y/+z in numpy index order, cm throughout;
  backends convert to their native units (MCGPU: cm, GATE: mm).
- Float64 for densities and activities. Required for byte-exact
  `.vox` text formatting (float32 roundtrip through `repr()` produces
  spurious digits).
- Counts are read from binary output arrays, never parsed from
  simulator log files.
- GATE uses its own material LUT (G4_AIR, G4_WATER, etc.); petsim
  material names are mapped via `_PETSIM_TO_GATE_MATERIAL_MAP` with
  unknown names passed through for custom `materials.db` users.