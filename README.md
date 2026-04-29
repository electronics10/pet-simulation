# PET Simulation Pipeline

Monte Carlo PET simulation pipeline for scatter correction research.
Generates paired sinograms (trues + scatter) from two simulators —
MCGPU-PET and GATE 10 — to serve as training data for a deep learning
scatter correction model.

---

## Background

In PET imaging, a significant fraction of detected photon pairs are
**scattered** — they have changed direction before detection and carry
incorrect spatial information. Scatter correction is an open problem,
and one approach is to train a model to predict the scatter component
from the measured sinogram.

To generate training data we need:
- A Monte Carlo simulator that can separately label **true** and
  **scattered** coincidences
- A controlled way to vary phantom geometry and activity distribution
- Enough runs to train a model

This project provides that pipeline. It wraps two simulators
(MCGPU-PET for GPU-accelerated fast simulation, GATE 10 for
physics-accurate reference simulation) behind a single Python API.

---

## Lab Device: Bruker Albira Scanner

| Parameter           | Value               |
| ------------------- | ------------------- |
| Scanner geometry    | CylindricalPET      |
| Rmax / Rmin         | 82 mm / 58 mm       |
| Axial height        | 105 mm              |
| Crystal material    | LYSO                |
| Crystal size        | 10 × 10 × 10 mm     |
| Crystal array       | 8 × 8 per module    |
| Modules per rsector | 3 (axial)           |
| Rsectors            | 8 (transaxial ring) |
| Energy resolution   | 15% at 511 keV      |
| Energy window       | 350–650 keV         |
| Coincidence window  | 10 ns               |

Derived ring structure: **24 rings axial** (8 crystals × 3 modules),
**64 crystals per ring** (8 crystals × 8 rsectors).

---

## Repository Layout

```
pet-simulation/
├── petsim/                        # Main library
│   ├── __init__.py                # Public API
│   ├── phantom.py                 # Voxelized geometry + materials
│   ├── source.py                  # Activity distribution (Bq per voxel)
│   ├── scanner.py                 # Hardware spec (backend-agnostic)
│   ├── sinogram_binning.py        # Sinogram layout spec (backend-agnostic)
│   ├── sinogram.py                # Output container (trues + scatter arrays)
│   ├── run.py                     # Bundles all 5 above into one saveable unit
│   ├── materials.py               # MCGPU material registry (.mcgpu.gz files)
│   └── backends/
│       ├── __init__.py
│       ├── mcgpu.py               # MCGPUConfig + write_vox / write_in
│       ├── mcgpu_runtime.py       # MCGPUBackend: stage → run → parse → save
│       └── gate.py                # GATEConfig + GATEBackend: build → run → parse → save
├── bin/
│   └── MCGPU-PET.x               # Compiled GPU binary (see Setup)
├── materials/                     # MCGPU cross-section tables (.mcgpu.gz)
│   ├── air_5-515keV.mcgpu.gz
│   ├── water_5-515keV.mcgpu.gz
│   └── ...
├── GateMaterials.db               # GATE custom material definitions (LYSO etc.)
├── runs/                          # Generated simulation output
│   └── 0001/
│       ├── run.yaml               # Manifest: seed, backend config, timestamps
│       ├── phantom.npz            # Voxelized geometry
│       ├── source.npz             # Activity distribution
│       ├── scanner.yaml           # Scanner hardware spec
│       ├── binning.yaml           # Sinogram layout choice
│       ├── sinogram.npz           # Trues + scatter arrays
│       └── mu_map.npy             # 511 keV linear attenuation map (GATE only)
├── main.py
└── README.md
```

---

## Setup

### 1. Prerequisites

- Ubuntu (24.04)
- git
- [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/electronics10/pet-simulation.git
cd pet-simulation
uv sync
```

### 2. MCGPU-PET binary

Acknowledgement:

> J.L. Herraiz, A. Lopez-Montes, and A. Badal, "MCGPU-PET: An Open-Source Real-Time Monte Carlo PET Simulator", Computer Physics Communications 296 (2024) 109008; https://doi.org/10.1016/j.cpc.2023.109008

`MCGPU-PET.x` is a GPU/CUDA-dependent binary. Build it from source:

```bash
git clone https://github.com/DIDSR/MCGPU-PET
cd MCGPU-PET
make                    # requires CUDA toolkit (nvcc)
cp MCGPU-PET.x /path/to/pet-simulation/bin/
cp -r sample_simulation/materials/* /path/to/pet-simulation/materials/
```

### 3. GATE (opengate)

Installed via `uv sync`. opengate downloads the required Geant4 data
files automatically on first run.

---

## Core Concepts

### Three backend-agnostic layers

The key design principle: **Scanner and SinogramBinning are separate**
and neither belongs to any backend.

```
Scanner          — what the hardware IS
                   (detector radius, rings, energy window, crystal geometry)

SinogramBinning  — how you choose to bin the data
                   (span, MRD, n_radial_bins, n_angular_bins)

Run              — bundles Phantom + Source + Scanner + SinogramBinning + seed
                   ready to hand to any backend
```

The same `Run` can be handed to `MCGPUBackend` or `GATEBackend`.
Backend-specific knobs (GPU number, physics list, etc.) live in
`MCGPUConfig` and `GATEConfig` respectively.

### Sinogram formats: backend-specific

The two backends currently produce sinograms with **different shapes**:

| Backend | Shape | Interpretation |
|---|---|---|
| MCGPU | `(n_z_planes, n_angular, n_radial)` | Native Michelogram (span-compressed ring pairs) |
| GATE | `(n_rings, n_rings, n_angular, n_radial)` | 4D ring-pair, lossless |

`Sinogram` carries an `axes` field documenting what each axis means:
GATE sinograms have `axes=("ring1", "ring2", "angular", "radial")`,
MCGPU has empty `axes` (legacy 3D).

For the Bruker preset, GATE produces shape `(24, 24, 32, 32)` ≈ 590k bins
≈ 2.4 MB per sinogram — small enough that lossless storage is fine.

### What gets saved per run

Each `run_dir/` contains the following files after a successful run:

| File | Contents | Backend |
|---|---|---|
| `run.yaml` | Manifest with seed, backend name, full backend config, timestamps, voxel size | both |
| `phantom.npz` | Voxelized geometry (material IDs + densities) | both |
| `source.npz` | Activity distribution (Bq per voxel) | both |
| `scanner.yaml` | Hardware spec | both |
| `binning.yaml` | Sinogram layout | both |
| `sinogram.npz` | Trues + scatter float32 arrays + axes labels | both |
| `mu_map.npy` | 511 keV linear attenuation map (cm⁻¹) | GATE |

These files are sufficient to **exactly reproduce the simulation**
years later, because `run.yaml` stores the complete `MCGPUConfig` or
`GATEConfig` used.

### Workdir vs run_dir

`run_full()` separates temporary scratch files from permanent output:

- `run_dir` — permanent: gets the essential output files
- `workdir` — temporary: all simulator scratch files, **auto-deleted on success**,
  **preserved on crash** so you can inspect input files and logs

---

## Usage

### Run MCGPU-PET

```python
from petsim import Phantom, Source, Scanner, SinogramBinning, Run
from petsim.backends import MCGPUBackend
from petsim.sinogram_binning import preset_binning

phantom = Phantom.cube(
    shape=(9, 9, 9), voxel_size=(1.0, 1.0, 1.0),
    inner_material="water", inner_density=1.0,
    outer_material="air", outer_density=0.0012,
    inner_size_vox=5,
)
source = Source.with_total_activity(
    phantom, material="water", total_activity_Bq=1e6, isotope="F18"
)
scanner = Scanner.from_preset("mcgpu_sample")
binning = preset_binning("mcgpu_sample")
run = Run(phantom, source, scanner, binning, seed=42)

backend = MCGPUBackend(executable="./bin/MCGPU-PET.x",
                       materials_dir="./materials")
sinogram, result = backend.run_full(run, run_dir="./runs/mcgpu_0001")
```

### Run GATE (Bruker scanner)

```python
from petsim.backends.gate import GATEBackend

scanner = Scanner.from_preset("bruker_albira")
binning = SinogramBinning.default_for(scanner)
run = Run(phantom, source, scanner, binning, seed=42)

backend = GATEBackend(materials_db="./GateMaterials.db")
sinogram, result = backend.run_full(run, run_dir="./runs/gate_0001")
```

### Share a MaterialRegistry across both backends

```python
from petsim.materials import MaterialRegistry, Material
from petsim.backends import MCGPUBackend
from petsim.backends.gate import GATEBackend

registry = MaterialRegistry(mcgpu_materials_dir="./materials")
registry.register(Material(
    name="tumor",
    nominal_density=1.04,
    mcgpu_file="soft_tissue_ICRP110_5-515keV.mcgpu.gz",
    gate_name="G4_TISSUE_SOFT_ICRP",
))

mcgpu_backend = MCGPUBackend(
    executable="./bin/MCGPU-PET.x",
    materials_registry=registry,
)
gate_backend = GATEBackend(
    materials_db="./GateMaterials.db",
    materials_registry=registry,
)
```

### Load an old run

```python
from petsim import Run, Sinogram

old_run = Run.load("./runs/mcgpu_0001")
sino = Sinogram.load("./runs/mcgpu_0001/sinogram.npz")
x = sino.trues
y = sino.scatter
```

---

## Scanner and Binning Presets

### Scanner presets (`petsim/scanner.py`)

| Preset | Description |
|---|---|
| `mcgpu_sample` | Toy scanner from MCGPU-PET sample simulation (80 rings, 336 crystals/ring) |
| `bruker_albira` | Lab device (24 rings, 64 crystals/ring, full crystal geometry for GATE) |

### Binning presets (`petsim/sinogram_binning.py`)

| Preset | span | MRD | n_radial | n_angular |
|---|---|---|---|---|
| `mcgpu_sample` | 11 | 79 | 147 | 168 |
| `bruker_albira` | 3 | 23 | 32 | 32 |

`SinogramBinning.default_for(scanner)` computes sensible defaults for
any scanner.

---

## Materials

Material name resolution is centralised in `petsim/materials.py`.
The `MaterialRegistry` is the single source of truth, used by both backends.

Each `Material` entry carries:
- `name` — the petsim name you use in Python (`"water"`, `"lyso"`, ...)
- `nominal_density` — default density in g/cm³
- `mcgpu_file` — filename of the `.mcgpu.gz` cross-section table (MCGPU)
- `gate_name` — Geant4 material name string (GATE)

Default materials cover air, water, and 14 ICRP 110 biological tissues.

### MCGPU: `.mcgpu.gz` files

MCGPU pre-tabulates cross-sections; one `.gz` file per material.

### GATE: `GateMaterials.db`

GATE uses Geant4, which computes cross-sections at runtime from atomic
composition. Standard materials (`G4_WATER`, `G4_AIR`, etc.) are built
into Geant4. Custom materials like LYSO must be defined in
`GateMaterials.db`.

The two formats deliver the same physics; just differently.

---

## Status

### Working

- **MCGPU backend**: full pipeline (voxelized phantom + voxelized source + native
  Michelogram output with trues/scatter split). Produced sinograms correctly
  for the standard sample simulation.
- **GATE backend Pass 1 (physics)**: voxelized phantom (ImageVolume) + voxelized
  source (VoxelSource). Without the phantom volume, photons fly through
  G4_AIR and never scatter — this was the root cause of zero-scatter
  output in earlier iterations.
- **GATE backend scatter labelling**: a separate `HitsCollectionActor` on
  the phantom volume records every interaction. Coincidences classified as
  trues / scatter / random by EventID match × phantom-scatter flag.
  Validated: water cube in air gives ~20% scatter fraction, consistent
  with NEMA NU4 published values for similar small-animal phantoms.
- **GATE backend Pass 3 (auxiliary outputs)**: `mu_map.npy` and
  `voxel_size_mm` in the run manifest.

### Known issue: GATE histogrammer (Pass 2 in progress)

The 4D ring-pair histogrammer produces correct **count totals** and
correct **ring-pair occupancy** (anti-diagonal pattern for centered
sources) but the **radial offset signal does not vary with view angle
as expected**. A point-source displacement test shows ~88% of
coincidences land on diametrically opposite crystals (`c2 = c1 + 32`)
regardless of source position — suggesting the source displacement is
not propagating through `VoxelSource` to the simulation.

Possible causes (not yet verified):
- `VoxelSource` may interpret MHD `Offset` differently than the MetaImage
  spec dictates.
- `VoxelSource` may centre the source on its `position.translation`
  field instead of using the MHD offset.
- The `_write_mhd_activity_image` coordinate convention may need
  rechecking (corner-of-voxel vs center-of-voxel).

The histogrammer logic itself (LOR midpoint, perpendicular angle, signed
radial distance) is sound — this has been verified on synthetic LOR data.
The issue is upstream in the source positioning.

**To resume debugging:** see `test_point_source.py`. The next step is
to test a large source displacement (e.g. +30 mm) and watch whether the
`c2 - c1` distribution shifts. If it doesn't, switch to
`pet_source.position.translation` as the source-positioning mechanism
instead of relying on the MHD offset.

### Recommendation while debugging is paused

For ML pipeline development, **use MCGPU output**. The MCGPU sinogram
format is correct (its native Michelogram), trues/scatter labelling
works, and 1000× faster simulation makes batch generation tractable.
GATE will become the cross-validation simulator and lab-shared
infrastructure for sister tasks (denoising, motion correction, PVC)
once the point-source issue is resolved.

---

## TODO

### GATE histogrammer (paused, see Status)

Resume with point-source displacement diagnostic. Likely fix is
switching from MHD-offset positioning to explicit
`pet_source.position.translation`.

### Cross-backend sinogram format reconciliation (deferred)

GATE produces 4D `(ring1, ring2, angular, radial)`; MCGPU produces 3D
Michelogram `(n_z, angular, radial)`. The two are not bin-comparable.
Three options exist (see git history for the full discussion):

- **Option A:** Re-bin GATE to MCGPU's Michelogram. Locks pipeline to
  MCGPU's idiosyncratic conventions.
- **Option B:** Force MCGPU to span=1, then re-bin its output to 4D.
  Standard, publishable, larger files.
- **Option C:** Keep both formats, treat GATE as physics-validation
  reference only. Train ML model on MCGPU exclusively.

Option C is the current de-facto state. Option B is the long-term
target. Decide before scaling up data generation.

### Batch generation

Write a script to generate N runs with randomized phantoms (varying
shape, size, activity distribution, materials) for ML training data.
Latin hypercube sampling over the parameter space is a candidate.

### Geometry convention reconciliation (minor)

`detector_radius_cm` means slightly different things in the two
backends:
- MCGPU: cylindric detector radius
- GATE: inner edge of crystal block (so crystal centers sit at
  `radius + crystal_x/2`)

For Bruker this is a ~5 mm offset. Affects scatter-fraction comparison
between backends. Adopt "crystal-center radius" as canonical when
reconciling.

### LYSO density (minor)

`GateMaterials.db` defines LYSO at 5.37 g/cm³, below real LYSO (~7.1).
Composition is also off (real LYSO is Lu-dominated by mass).
Affects detection efficiency in GATE. The Python `MaterialRegistry`
records 7.10 g/cm³, but inside GATE the `.db` value wins — fix in `.db`.

---

## History and handover notes

### GATE 9 (abandoned)

An earlier handover (Kishore) used GATE 9.4.1 via Docker with `.mac`
macro files. The digitizer API changed in GATE v9.3 and the handover
macros (`Bruker_PET.mac`, `Bruker_PET1.mac`) are broken. **Do not use.**
The `.mac` files are kept only as a geometry reference for the Bruker
scanner specs. GATE 10 (`opengate`) is the Python-native replacement.

### petsim design decisions

- **Scanner does not own binning.** Scanner is hardware; SinogramBinning
  is a choice. The same scanner can produce different sinogram layouts.
  Binning lives in `Run.binning` and is passed to both backends.
- **Sinogram is rank-agnostic.** Stores arrays of arbitrary rank plus an
  optional `axes` field. Backends choose what shape to produce.
- **Auto-save on success, preserve on crash.** `run_full()` saves the
  essential files to `run_dir` on success and deletes the temp workdir.
  On crash, the workdir is preserved for debugging.
- **n_z_slices in MCGPU is an input parameter**, not the output shape.
  It is computed as `2*n_rings - 1` and written to the `.in` file.
  The actual output z-planes are auto-detected from file size at parse time.
- **GATE phantom must be a real volume.** A voxelized source alone is
  not enough — without an `ImageVolume` made of real materials, photons
  fly through G4_AIR with no scatter. This was the root cause of weeks
  of debugging.