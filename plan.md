# pet-simulation: Project Plan

A unified Python interface for generating PET simulation data across MCGPU-PET and GATE, with a storage format designed around the inverse problem structure `y = A(x)`.

---

## Context and motivation

The goal of this project is to build a dataset pipeline for a deep-learning-based PET scatter correction model. Training such a model requires large numbers of paired sinograms (scatter-contaminated and scatter-free or scatter-labeled), which come from Monte Carlo simulation.

Two simulators are in play:

- **MCGPU-PET** — a fast GPU-based Monte Carlo simulator (FDA/DIDSR, CC0). Pure MC transport runs at ~4.5M histories/second on a TITAN RTX. Good for generating large volumes of training data, but uses a clumsy file-based configuration workflow (text `.vox` file, gzipped cross-section files, plain-text `.in` config).
- **GATE** (accessed via the `opengate` Python package) — the community-standard PET simulator. Much more configurable, crystal-level detector physics, but slower. Best for validated reference simulations.

Currently each is handled ad-hoc. Generating 100 varied MCGPU-PET simulations for training would mean 100 hand-edited voxel files. Integrating GATE simulations alongside would mean a parallel workflow.

Real scanner DICOM integration is explicitly **out of scope** for this project — it's a separate workstream.

## What we're building

A Python library that provides one consistent API for both simulators. The user constructs three objects — a phantom, a source, and a scanner — and hands them to a backend. The backend handles format translation, runs the simulator, and returns a standardized `Sinogram` object that downstream training code consumes without caring which simulator produced it.

```python
phantom = Phantom.cylinder(radius=5, height=10, material="water")
source  = Source(activity_map=..., isotope="F18")
scanner = Scanner.from_preset("bruker_albira")

backend = MCGPUBackend()
run = backend.run(phantom, source, scanner, out_dir="runs/001")
# Later:
run = Run.load("runs/001")
sino = run.sinogram
```

Swapping in `GATEBackend()` runs the equivalent GATE simulation with no other changes.

## The inverse problem framing

Every simulation run produces data that fits the standard inverse problem structure:

```
y = A(x) + scatter + noise
```

- **`x`** — the ground truth activity distribution (phantom + source)
- **`A`** — the forward model: scanner geometry, attenuation, binning scheme
- **`y`** — the measured sinogram

Reconstruction algorithms need all three. Scatter correction happens in sinogram space (on `y`), but evaluating a correction requires access to `x` (to compute ground truth) and `A` (to understand what each bin means). Therefore every run stores all three.

## Key design decisions

**Sinogram is the primary output.** Images are downstream of reconstruction — an ill-posed inverse problem that bakes in algorithmic choices. Sinograms are the raw measurement; any downstream consumer can reconstruct them their own way. Scatter correction specifically operates in sinogram space.

**Separate phantom and source.** MCGPU-PET fuses them into a single `.vox` file; GATE keeps them separate. We follow GATE's convention in the Python layer because it mirrors physical reality (same patient, different tracers). Only the MCGPU backend fuses them at write time.

**Every run stores `x`, `A`, and `y` separately.** One directory per run. Phantom, scanner, and sinogram are distinct files. This supports fast sinogram-only loading for DL training while preserving full reproducibility.

**Material registry as a cross-cutting concern.** A single dict maps human-readable names (`"brain"`, `"water"`, `"bone"`) to both a GATE material definition and an MCGPU `.mcgpu.gz` file path. Users never touch raw material files.

**Treat MCGPU-PET as a black box.** Don't modify the CUDA source during Phases 1–3. Specific modifications (e.g. raising `MAX_MATERIALS=15`, disabling the GPU display timeout) become a separate workstream if needed.

**Python API only.** No CLI, no GUI. The API is the interface.

## Scope boundaries

**In scope:**
- Phantom construction, serialization, deserialization
- Material management across simulators
- Running MCGPU-PET and GATE through a unified interface
- Persistent storage of `x`, `A`, `y` with full reproducibility
- Validation against the existing sample simulations

**Out of scope:**
- DICOM/real scanner data loading
- Deep learning training code
- Image reconstruction
- Scanner calibration
- GUI or web interface

---

## Architecture overview

```
pet-simulation/
├── petsim/                          ← Python package (to create)
│   ├── __init__.py
│   ├── phantom.py                   ← Phase 1
│   ├── source.py                    ← Phase 1
│   ├── scanner.py                   ← Phase 1
│   ├── sinogram.py                  ← Phase 1
│   ├── materials.py                 ← Phase 1
│   ├── run.py                       ← Phase 1 (storage layout)
│   └── backends/
│       ├── mcgpu.py                 ← Phase 2
│       └── gate.py                  ← Phase 3
├── MCGPU-PET/                       ← Existing, untouched (forked from FDA)
├── gate-pet/                        ← Existing, current ad-hoc GATE scripts
├── runs/                            ← Simulation outputs (gitignored)
├── tests/                           ← Validation tests per phase
├── pyproject.toml                   ← Existing (uv)
└── PLAN.md                          ← This file
```

---

## Storage format

One directory per run, fully self-contained:

```
runs/001_water_cylinder/
├── run.yaml                ← manifest: backend, seed, git hash, wall time
├── phantom.npz             ← x: material_ids, densities, voxel_size
├── source.npz              ← x: activity_map, isotope, total_activity_Bq
├── scanner.yaml            ← A: geometry, energy window, binning
├── sinogram.npz            ← y: trues, scatter, randoms
└── simulator_output/       ← gitignored; raw backend files
    ├── MCGPU-PET.in        ← generated input (MCGPU case)
    ├── MCGPU-PET.out       ← simulator log
    └── *.raw.gz            ← raw simulator outputs
```

**Why separate files:** lets you load only `sinogram.npz` during training iterations without pulling in the full phantom. Major speedup at dataset scale.

**Why npz + YAML:** npz is numpy-native, compressed, fast. YAML is human-readable for scanner and manifest so you can `cat` to inspect without Python.

**Why `run.yaml`:** records backend, random seed, git commit hash of petsim at run time, wall time, count totals, pointers to other files. Makes runs reproducible and auditable.

**Why `simulator_output/` is gitignored:** reconstructible from `run.yaml` + seed + inputs; and they're large.

**Loading API:**
```python
run = Run.load("runs/001/")
run.phantom      # → Phantom
run.source       # → Source
run.scanner      # → Scanner
run.sinogram     # → Sinogram
run.metadata     # → dict

# Fast path for training:
sino = Sinogram.load("runs/001/sinogram.npz")
```

**Size estimate:** ~50–150 MB per run depending on phantom resolution and whether simulator raw outputs are kept. 1000 runs ≈ 100 GB, manageable on workstation disk.

---

## Phase 1: Core data model

Build the simulator-agnostic data classes and storage layer. No simulator I/O yet.

### What gets built

- **`Phantom`** class: numpy 3D arrays for material IDs (int) and densities (float), plus voxel spacing. Factory methods: `cylinder()`, `sphere()`, `cube()`, `from_numpy()`.
- **`Source`** class: 3D activity array aligned with a phantom grid, plus isotope name and total activity in Bq.
- **`Scanner`** class: detector geometry (radius, axial length, energy window, energy resolution, timing) and scanner-specific parameters. Supports presets. Required for interpreting sinogram bins.
- **`Sinogram`** class: numpy arrays for trues/scatter/randoms (any may be `None`), plus a metadata dict. Carries a reference to the `Scanner` that produced it and optionally the `Phantom` (for attenuation correction).
- **`MaterialRegistry`**: dict-based registry mapping names to material definitions for both simulators.
- **`Run`** class: orchestrates loading/saving a full simulation directory according to the storage format above.

### Validation criteria

- Instantiate each of the main classes with plausible inputs; no exceptions.
- `Phantom.cylinder()` voxel count inside the cylinder matches the analytical expectation within 5% (voxel discretization error).
- `Source` correctly scales a constant activity map to hit a target total Bq (verified by summing).
- `Sinogram` round-trips through `save()`/`load()` without data loss.
- `Run` round-trips a full scene (phantom + source + scanner + sinogram) through save/load with byte-exact match for arrays and value equality for metadata.
- Material registry lookup works for all 17 ICRP 110 tissues in the MCGPU material zip.
- Unit tests cover all factory methods and all public attributes.

**Phase 1 is done when:** I can describe a water-cylinder-with-a-hot-spot scene purely in Python objects, save it to disk using the storage format, load it back, and verify all arrays and metadata are preserved exactly. No simulator has been touched.

---

## Phase 2: MCGPU-PET backend

Wrap MCGPU-PET behind the Phase 1 interface.

### What gets built

- `MCGPUBackend.write_inputs(phantom, source, scanner, out_dir)`:
  - Writes the `.vox` text file with MCGPU header and one line per voxel (material, density, activity), fusing phantom + source.
  - Writes an `.in` config file templated from scanner parameters.
  - Copies or symlinks required `.mcgpu.gz` material files.
- `MCGPUBackend.run(phantom, source, scanner, out_dir) -> Run`:
  - Calls `write_inputs`, then invokes `MCGPU-PET.x` via `subprocess`.
  - Streams stdout to a log file.
  - Raises clear exceptions on non-zero exit or known error strings.
  - Saves phantom, source, scanner to disk per storage format.
  - Parses output and saves `sinogram.npz`.
  - Writes `run.yaml` with backend name, seed, wall time, count totals, git hash of petsim.
  - Returns a fully-populated `Run` object.
- `MCGPUBackend.load_results(out_dir) -> Sinogram`:
  - Reads gzipped `.raw` files with correct dtype and shape.
  - Parses `.out` log for metadata.

### Validation criteria

- **Byte-exact reproduction of the sample:** generate the 9×9×9 water phantom through the Python API; the resulting `.vox` file matches the bundled `phantom_9x9x9cm.vox` modulo formatting whitespace.
- **End-to-end sample run:** running `backend.run()` on the reproduced sample produces a `Sinogram` with total trues within 10% of the original sample run (~3700 trues). Stochastic variation expected.
- **Error handling:** passing an invalid phantom (unknown material) raises a clear Python exception instead of silently producing garbage.
- **Larger phantom smoke test:** a 64×64×64 cylindrical phantom with a hot spot produces a non-empty sinogram with the hot spot visible at the correct location in the voxel emission map.
- **Metadata parsing:** `run.metadata["scatter_fraction"]`, `["total_trues"]`, `["simulated_histories"]`, `["wall_time_seconds"]` match values in the MCGPU log.
- **Storage round-trip:** `Run.load(out_dir)` after `backend.run(...)` returns data equal to what was produced in memory.

**Phase 2 is done when:** I can generate, run, load, and re-load the sample simulation entirely through Python, and parameter sweeps (varying activity, phantom size, acquisition time) work without manual file editing.

---

## Phase 3: GATE backend

Wrap GATE (via `opengate`) behind the same Phase 1 interface.

### What gets built

- `GATEBackend.build_simulation(phantom, source, scanner, out_dir) -> gate.Simulation`:
  - Translates `Phantom` into GATE volumes (`ImageVolume` for voxelized data; `Box`/`Sphere`/`Tubs` for analytical shapes).
  - Translates `Scanner` into the crystal → module → rsector → envelope hierarchy, with presets starting from the existing `gate-pet/bruker_pet_sim.py`.
  - Translates `Source` into a `GenericSource` with `back_to_back` particle type, correct activity, and position sampling tied to the activity map.
  - Attaches the standard digitizer chain: Hits → Singles → EnergyWindow → Coincidences.
- `GATEBackend.run(phantom, source, scanner, out_dir) -> Run`:
  - Calls `sim.run(start_new_process=True)`.
  - Histograms ROOT coincidence events into a sinogram matching the shape produced by the MCGPU backend.
  - Saves to disk per storage format.

### Validation criteria

- **Existing Bruker script reproduction:** the current `gate-pet/bruker_pet_sim.py` is expressible through the new API and produces event counts within 5% of the original and the same coincidence rate.
- **Format parity:** a `Sinogram` from GATE and a `Sinogram` from MCGPU, for the same phantom/source/scanner, have identical shapes, dtypes, metadata keys, and file layout on disk. Values differ (different physics models) but containers are interchangeable.
- **Swap test:** any code that works with an MCGPU `Run` also works with a GATE `Run`, with zero code changes.
- **Material cross-check:** a phantom referencing `"brain"` uses the GATE brain material in the GATE backend and the MCGPU brain cross-section file in the MCGPU backend, both automatically.
- **Storage round-trip:** same as Phase 2.

**Phase 3 is done when:** the same Python script can generate MCGPU and GATE sinograms of the same scene by changing one line, and the results can be compared head-to-head programmatically.

---

## Cross-cutting validation: the reference scene

Every phase validates against the same reference scene — a water cylinder with a hot-spot insert, 1 second simulated acquisition. By end of Phase 3, both backends produce `Run` directories for this scene that are programmatically comparable.

## Risk register

- **MCGPU-PET material cap:** hardcoded `MAX_MATERIALS=15` limits complex phantoms. Not a Phase 1–3 blocker. CUDA source modification if needed later.
- **GPU display timeout:** 5-second CUDA kernel timeout when the GPU drives a display kills larger simulations. Workaround: `sudo systemctl isolate multi-user.target`. Long-term: split long runs into batches automatically.
- **ROOT parsing complexity:** GATE's ROOT output is nested. Budget extra time for Phase 3 output parsing. Try `uproot` first, fall back to opengate's built-in sinogram writer if available.
- **Sinogram binning conventions:** MCGPU-PET uses a specific span/ring-difference scheme; GATE has its own. Decide early whether `Sinogram` enforces a single canonical binning or carries the original as metadata. Current plan: enforce canonical in the output format, document the translation per backend, store the raw binning in `scanner.yaml`.
- **Reproducibility of stochastic runs:** both simulators are Monte Carlo; exact value match across runs is impossible. All validation criteria specify tolerances (5–10%) or statistical properties rather than byte-exact sinogram equality.

---

## Development conventions

- Each phase is a separate branch, merged to `main` only after validation criteria pass.
- Each new class gets a unit test file before the class is considered done.
- The reference scene has an integration test that runs end-to-end after every major change.
- Simulation outputs (`runs/`, `simulator_output/`, `.raw.gz`, `.root`) are gitignored. Only Python source, tests, and this plan are versioned.
- The `MCGPU-PET/` subdirectory is a plain folder (not a submodule). Any source modifications get their own commit with a clear message.
- Attribution: the README credits the original MCGPU-PET authors per the CC0 citation request.