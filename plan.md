# pet-simulation — project plan

A unified Python pipeline for generating PET Monte Carlo simulation data,
with interchangeable backends (MCGPU-PET for speed, GATE as reference).

## Scope

- **In:** Python-native scene description (phantom + source + scanner),
  backend abstraction, sinogram output, ML-ready run bundles.
- **Out:** image reconstruction, DICOM I/O, real-scanner interfaces, GUI.

---

## Inverse problem framing

Every simulation produces a record of three things:

    y = A(x) + noise

  - `x` = ground truth = `Phantom` (geometry + materials) + `Source` (activity)
  - `A` = forward model = `Scanner` (geometry + binning + assumptions)
  - `y` = measurement = `Sinogram` (trues + scatter, optionally randoms)

All three are stored per run. Losing `A` would mean the sinogram bins have
no physical meaning; losing `x` would make ML training pointless. The
`Run` class holds all three plus metadata.

### Explicit forward-model assumptions

`Scanner` carries three fields that make A unambiguous:

  - `binning_convention` — free-form string ID for the LOR indexing scheme
    (e.g. `"mcgpu_span11_mrd79"`, `"gate_span3_mrd31"`). Two sinograms
    with different conventions are NOT directly comparable.
  - `tof_enabled` — whether time-of-flight info is preserved.
  - `normalization` — what corrections have been applied (`"none"`,
    `"attenuation_corrected"`, etc.).

`Scanner.is_compatible_with(other)` checks whether two scanners produce
bin-for-bin comparable sinograms. This is the early-warning system that
stops silent mismatches between backends.

### Coordinate convention

Fixed petsim-wide, documented in `phantom.py`:

  - **Origin:** voxel (0, 0, 0), the grid corner (not center).
  - **Axes:** +x, +y, +z in array index order.
  - **Units:** centimeters in all public APIs. Backends convert to their
    native units (MCGPU: cm; GATE: mm) at write time.
  - **Voxel center:** at `((ix+0.5)*dx, (iy+0.5)*dy, (iz+0.5)*dz)`.

Phantom, Source, and Scanner all adhere to this. Backends are responsible
for converting to and from their conventions.

---

## Storage format

Every run is a directory:

    runs/001_water_cylinder/
    ├── run.yaml          ← manifest (backend, seed, wall time, counts, git hash)
    ├── phantom.npz       ← x: material_ids, densities, voxel_size
    ├── source.npz        ← x: activity_Bq, isotope, voxel_size
    ├── scanner.yaml      ← A: geometry, binning, forward-model assumptions
    ├── sinogram.npz      ← y: trues, scatter, randoms (any subset)
    └── simulator_output/ ← raw backend files (gitignored)

YAML for human-readable config, NPZ for arrays. `Run.save(dir)` /
`Run.load(dir)` is the top-level API.

Rationale for sinogram-over-image: images bake in a reconstruction choice;
scatter correction operates in sinogram space; the sinogram is the raw
measurement. Image ↔ sinogram transforms are downstream work.

---

## Architecture

```
petsim/
├── phantom.py       — Phantom: voxel geometry + materials
├── source.py        — Source: activity distribution aligned with a Phantom
├── scanner.py       — Scanner: geometry + binning + forward-model fields
├── sinogram.py      — Sinogram: trues/scatter/randoms container
├── materials.py     — MaterialRegistry: names → ICRP 110 files / GATE names
├── run.py           — Run: bundle of all four + manifest, save/load orchestration
└── backends/
    ├── mcgpu.py     — MCGPU-PET backend (Phase 2)
    └── gate.py      — GATE backend (Phase 3)
```

Each top-level class is a pure data class with no backend-specific code.
Backends sit below the data model and are the only code that touches
external tools.

---

## Phase 1 — core data model — COMPLETE

**Status:** all criteria met, 151 tests passing.

**What got built:**
  - `Phantom`, `Source`, `Scanner`, `Sinogram`, `MaterialRegistry`, `Run`.
  - Factory methods for common shapes (cube, cylinder, sphere, uniform).
  - Cross-component validation: source grid must match phantom, sinogram
    scanner must match run scanner.
  - Forward-model explicitness: `binning_convention`, `tof_enabled`,
    `normalization` on Scanner plus `is_compatible_with()` method.
  - `seed` as a first-class field on Run (not buried in metadata).
  - Coordinate convention fixed and documented in `phantom.py`.
  - Reference scene test: water cylinder with hot spot insert
    round-trips byte-exact through save/load.

**Validation criteria (all met):**
  - Full scene round-trips save/load exactly.
  - Cylinder/sphere volumes within 5% of analytical.
  - Material registry covers 17 ICRP 110 tissues + air + water.
  - Forward-model fields survive round-trip and block incompatible
    sinogram comparisons.
  - Seed is reproducibly persisted.

---

## Phase 2 — MCGPU-PET backend

**Goal:** `backend.run(run: Run) -> Run` that writes a completed sinogram
into `run.sinogram`.

**Deliverables:**
  - `petsim/backends/mcgpu.py` with `MCGPUBackend` class.
  - `write_inputs(run, workdir)` — generate `.vox` and `.in` files.
  - `run(run, workdir)` — invoke the executable, capture timing.
  - `load_results(run, workdir)` — parse `.raw.gz` files into `Sinogram`,
    populate metadata from log file (wall time only; counts come from
    summing the arrays, not parsing logs — they're not stable API).

**Key implementation choices:**
  - Pass seed through explicitly so backend reproducibility is verifiable.
  - Read all counts (trues, scatter, scatter fraction) by summing the
    binary arrays, not parsing the text log.
  - Store executable path and `MCGPU-PET` git hash in the run metadata
    for provenance.
  - MCGPU-PET has a hardcoded `MAX_MATERIALS=15` limit. Validate the
    phantom's material count before writing inputs.

**Validation criteria:**
  - Byte-exact reproduction of the bundled `phantom_9x9x9cm.vox` from a
    Python `Phantom` + `Source`.
  - End-to-end run on the sample simulation: trues count within 10% of
    ~3700 (MC statistical variation).
  - Same seed → statistically identical output (within MC noise floor).
  - Different seed → statistically independent output.
  - Clean error if the phantom references a material whose `.mcgpu.gz`
    file is missing (use `MaterialRegistry.verify_mcgpu_files()`).
  - 64³ smoke test completes without GPU kernel timeout (server must be
    in multi-user.target; document this).

**Deferred to Phase 2b or later:**
  - `.vox` file caching by content hash. Small datasets (10-100 runs)
    don't need it; revisit at 1000+ runs if I/O becomes a bottleneck.

---

## Phase 3 — GATE backend

**Goal:** `GATEBackend` that consumes the same `Run` and produces the
same `Sinogram` format.

**Deliverables:**
  - `petsim/backends/gate.py` translating Python objects to opengate.
  - Coincidence-to-sinogram binning, since GATE gives events not sinograms.
  - Preserving `binning_convention` so users can detect incompatibility
    with MCGPU output before mis-comparing.

**Validation criteria:**
  - Reproduce existing `gate-pet/bruker_pet_sim.py` results within 5%.
  - Swap test: same `Run` through both backends produces sinograms that
    declare themselves incompatible via `Scanner.is_compatible_with()`
    until explicit rebinning is applied.
  - Seed-pair noise realizations work (same phantom+source, different
    seeds → independent noise).

**Deferred to Phase 3b:**
  - `Sinogram.rebin(target_convention)` — the actual LOR-remapping code
    that makes MCGPU and GATE output directly comparable. Hard problem,
    only tackled when we have both backends producing output side by side.

---

## Risks and known gotchas

  - **GPU kernel timeout:** if the GPU is connected to a display, CUDA
    kills kernels after 5 seconds. Document the `systemctl isolate
    multi-user.target` workaround; consider detecting and warning.
  - **MAX_MATERIALS=15:** MCGPU-PET hardcoded limit. Validate early.
  - **Binning conventions:** MCGPU and GATE do not naturally agree.
    Surfaced via `binning_convention` field; rebinning deferred to
    Phase 3b.
  - **Log parsing fragility:** never parse counts from `.out` files.
    Read binary sinograms and sum them.
  - **Float precision in round-trip:** voxel_size is stored as float64
    (not float32) to avoid ~1e-9 drift breaking equality tests.
  - **Coordinate drift:** fixed by the convention in `phantom.py`.
    Backends must convert; tests must verify conversion is lossless.

---

## What's explicitly out of scope (for sanity)

  - Image reconstruction (that's parallelproj / STIR territory).
  - DICOM I/O.
  - Real-scanner interfaces.
  - Analytic (non-voxelized) sources. MCGPU-PET doesn't support them;
    if we ever need them, add a `Source.mode` field then.
  - CUDA kernel modifications. The simulator is a black box.
  - A unified top-level `SimulationConfig` class. `Run` already plays
    that role; adding another wrapper would be two classes doing one job.