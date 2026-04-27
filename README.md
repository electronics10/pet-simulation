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
│       └── sinogram.npz           # Trues + scatter arrays
├── main.py
└── README.md
```

---

## Setup

### 1. Prerequisite

- Ubuntu (24.04)
- git
- [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/electronics10/pet-simulation.git
cd pet-simulation
uv sync
```

### 2. MCGPU-PET binary check

Acknowledgement

> J.L. Herraiz, A. Lopez-Montes, and A. Badal, "MCGPU-PET: An Open-Source Real-Time Monte Carlo PET Simulator", Computer Physics Communications 296 (2024) 109008; https://doi.org/10.1016/j.cpc.2023.109008

I use MCGPU-PET (https://github.com/DIDSR/MCGPU-PET) in this project. The binary `MCGPU-PET.x` file is used as the direct backend, and it is GPU/CUDA dependent. Therefore, one may need to setup its own `MCGPU-PET.x`.

```bash
git clone https://github.com/DIDSR/MCGPU-PET
cd MCGPU-PET
make                    # requires CUDA toolkit (nvcc)
cp MCGPU-PET.x /path/to/pet-simulation/bin/
cp -r sample_simulation/materials/* /path/to/pet-simulation/materials/
```

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

### Sinogram layout: the Michelogram

PET sinograms use a **Michelogram** compression scheme to reduce storage.
Ring pairs are grouped into segments by their ring difference, and
compressed axially using a span parameter.

`SinogramBinning` captures the four parameters that define this layout:

| Parameter | Meaning |
|---|---|
| `n_radial_bins` | Radial bins per plane (transverse resolution) |
| `n_angular_bins` | Angular bins per plane (typically `n_crystals/2`) |
| `span` | Axial compression factor (odd integer; larger = more compression) |
| `max_ring_difference` (MRD) | Maximum ring pair distance included |

For the `mcgpu_sample` preset: span=11, MRD=79 → output shape `(1293, 168, 147)`.
For the `bruker_albira` default: span=3, MRD=23 → output shape `(47, 32, 32)`.

### What gets saved per run

Each `run_dir/` contains exactly six files after a successful run:

| File | Contents |
|---|---|
| `run.yaml` | Manifest with seed, backend name, full backend config, timestamps |
| `phantom.npz` | Voxelized geometry (material IDs + densities) |
| `source.npz` | Activity distribution (Bq per voxel) |
| `scanner.yaml` | Hardware spec |
| `binning.yaml` | Sinogram layout |
| `sinogram.npz` | Trues + scatter float32 arrays |

These six files are sufficient to **exactly reproduce the simulation**
years later, because `run.yaml` stores the complete `MCGPUConfig` or
`GATEConfig` used.

### Workdir vs run_dir

`run_full()` separates temporary scratch files from permanent output:

- `run_dir` — permanent: gets the 6 essential files
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
# → ./runs/mcgpu_0001/ now has 6 essential files
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

Both backends use `MaterialRegistry` for material name resolution.
By default each backend builds its own, but you can share one instance —
useful if you've added custom materials or want a single place to manage them.

```python
from petsim.materials import MaterialRegistry, Material
from petsim.backends import MCGPUBackend
from petsim.backends.gate import GATEBackend

registry = MaterialRegistry(mcgpu_materials_dir="./materials")

# Register a custom tissue not in the defaults
registry.register(Material(
    name="tumor",
    nominal_density=1.04,
    mcgpu_file="soft_tissue_ICRP110_5-515keV.mcgpu.gz",  # reuse closest file
    gate_name="G4_TISSUE_SOFT_ICRP",
))

mcgpu_backend = MCGPUBackend(
    executable="./bin/MCGPU-PET.x",
    materials_registry=registry,      # share the registry
)
gate_backend = GATEBackend(
    materials_db="./GateMaterials.db",
    materials_registry=registry,      # same registry
)
```

### Load and reproduce an old run

```python
from petsim import Run
from petsim.backends import MCGPUBackend, MCGPUConfig

old_run = Run.load("./runs/mcgpu_0001")

# Exact config is recorded in run.yaml
config = MCGPUConfig(**old_run.metadata["mcgpu_config"])

backend = MCGPUBackend(executable="./bin/MCGPU-PET.x",
                       materials_dir="./materials")
sinogram, _ = backend.run_full(old_run, run_dir="./runs/mcgpu_0001_repro",
                                config=config)
```

### Access training arrays

```python
from petsim import Sinogram

sinogram = Sinogram.load("./runs/mcgpu_0001/sinogram.npz")
x = sinogram.trues    # (1293, 168, 147) float32 — model input
y = sinogram.scatter  # (1293, 168, 147) float32 — model target
```

---

## Scanner and Binning Presets

### Scanner presets (`petsim/scanner.py`)

| Preset | Description |
|---|---|
| `mcgpu_sample` | Toy scanner from MCGPU-PET sample simulation (80 rings, 336 crystals/ring) |
| `bruker_albira` | Lab device (24 rings, 64 crystals/ring, full crystal geometry for GATE) |

### Binning presets (`petsim/sinogram_binning.py`)

| Preset | span | MRD | n_radial | n_angular | Output shape (n_z, ang, rad) |
|---|---|---|---|---|---|
| `mcgpu_sample` | 11 | 79 | 147 | 168 | (1293, 168, 147) |
| `bruker_albira` | 3 | 23 | 32 | 32 | (47, 32, 32) |

`SinogramBinning.default_for(scanner)` computes sensible defaults
(span=3, MRD=n_rings-1, n_radial=n_angular=n_crystals/2) for any scanner.

---

## Materials

Material name resolution is centralised in `petsim/materials.py`.
The `MaterialRegistry` is the single source of truth, used by both backends.

### MaterialRegistry

Each `Material` entry carries:
- `name` — the petsim name you use in Python (`"water"`, `"lyso"`, ...)
- `nominal_density` — default density in g/cm³
- `mcgpu_file` — filename of the `.mcgpu.gz` cross-section table (MCGPU)
- `gate_name` — Geant4 material name string (GATE)

When you write `Phantom.cube(inner_material="water")`, both backends
resolve `"water"` through the registry — MCGPU gets
`water_5-515keV.mcgpu.gz`, GATE gets `G4_WATER`.

Default materials cover air, water, and 14 ICRP 110 biological tissues
(brain, lung, liver, muscle, etc.). See `_DEFAULT_MATERIALS` in
`materials.py` for the full list.

### MCGPU: `.mcgpu.gz` files

MCGPU-PET cannot compute cross-sections at runtime. It reads
**pre-tabulated cross-section tables** compressed as `.mcgpu.gz` files,
one per material, from the `materials/` directory.

### GATE: `GateMaterials.db`

GATE uses Geant4, which **computes cross-sections at runtime** from
atomic composition. Standard materials (`G4_WATER`, `G4_AIR`, etc.)
are built into Geant4's NIST database — no file needed. Custom
materials like LYSO (the Bruker's crystal scintillator) must be defined
in `GateMaterials.db` — a plain-text composition + density file.

`GATEBackend` copies the `.db` file into its workdir automatically.
The crystal material name (e.g. `"lyso"` in `scanner.crystal_material`)
is resolved via `registry.gate_name("lyso")` → `"LYSO"`.

The two formats are physically the same thing — cross-section data —
just delivered differently: MCGPU pre-tabulates, Geant4 computes at runtime.

---

## TODO

### Sinogram format: MCGPU's Michelogram vs simulator-agnostic list-mode (critical decision)

`_histogram_coincidences` in `gate.py` is a **placeholder** producing direct-segment only.
GATE and MCGPU sinograms currently have **different shapes and bin conventions**.

**The core issue:** MCGPU uses a nonstandard Michelogram compression (1293 z-planes for
mcgpu_sample, not the textbook 1153). Matching MCGPU's exact format means locking the
entire pipeline to one tool's idiosyncratic conventions.

**Three architectural options exist. Choose one before implementing the histogrammer:**

#### Option A: Match MCGPU's native Michelogram exactly

**Pros:**
- MCGPU data is in its native format (no re-binning needed)
- Direct comparison of bin-by-bin sinograms

**Cons:**
- MCGPU's convention is nonstandard and undocumented (reverse-engineering required)
- Locks the pipeline to MCGPU's quirks for all future work
- GATE output must be re-binned to match MCGPU's scheme
- Hard to publish: reviewers may question MCGPU's format

**Implementation:** Empirically reverse-engineer MCGPU's ring-pair → z-plane mapping by
running single-z-slice phantoms through MCGPU and observing where counts land. Then
implement ring-pair → z-plane + angular/radial binning in `_histogram_coincidences`.

#### Option B: Use simulator-agnostic 4D format `(ring1, ring2, angular, radial)`

**Pros:**
- Standard, reproducible, defensible for publication (no simulator lock-in)
- GATE histogrammer is ~30 lines (trivial)
- Easy to add new simulators later
- Clear, well-defined coordinates

**Cons:**
- MCGPU output must be post-processed (span=1 only, or re-binning from span>1)
- Larger output shape: `(24, 24, 32, 32) = 589KB` per sinogram for Bruker (still fine)
- Can't use MCGPU's span>1 compression during simulation (only span=1)

**Implementation:** Keep `SinogramBinning` but deprecate `span` and `max_ring_difference`.
Force `span=1` in MCGPUConfig. GATE produces 4D directly. Both backends output the same
shape and semantics.

#### Option C: Keep separate formats; use GATE as validation only

**Pros:**
- No architectural changes needed
- MCGPU runs in native span>1 (if that matters for speed)
- GATE is a "reference physics check" not an interchangeable source

**Cons:**
- GATE and MCGPU sinograms are not comparable bin-by-bin
- Can only train on MCGPU data
- Requires two separate models if you want GATE results

**Implementation:** Leave the placeholder as-is. Run GATE + MCGPU on same phantoms,
verify total counts and scatter fractions are similar (physics validation), move on.

#### Recommendation

**Option B** is most robust for a research pipeline. It decouples you from MCGPU's
idiosyncrasies and makes the work more publishable. The 589KB per sinogram is negligible.
If span compression matters for speed later, you can add it as a post-processing step
after training.

**To implement Option B:**

1. Modify `SinogramBinning` to remove/deprecate `span` and `MRD` fields
2. Ensure MCGPU always uses `span=1` (update MCGPUConfig defaults)
3. Write `_histogram_coincidences` to bin GATE list-mode into `(ring1, ring2, ang, rad)`
   - For each coincidence pair: extract ring indices from (x,y,z) positions
   - Compute angular bin from LOR azimuth: `atan2(y2-y1, x2-x1) % pi`
   - Compute radial bin from perpendicular distance to FOV axis
4. Validate: run same water cube through both, verify shapes match and counts are comparable

**Choose before next session.** The decision affects the entire histogrammer design.

### Voxelized source for GATE

Currently uses a point source at the FOV center. The ITK export
(`_write_mhd_activity_image`) is already implemented but not
yet wired up. Replace the `GenericSource` point source in
`GATEBackend.build()` with a `VoxelizedSource` reading from
`source_activity.mhd`.

### Scatter labelling for GATE

GATE can label scattered coincidences via `EventID` matching — compare
the EventID of both detected photons; if they differ, it's a scatter.
Currently `sinogram.scatter` is `None` for GATE runs. This needs to
be implemented in `parse_sinogram()`.

### Batch generation

Write a script to generate N runs with randomized phantoms (varying
shape, size, activity distribution) for ML training data.

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
- **MCGPUConfig owns only MCGPU-specific runtime knobs** (GPU number,
  PSF size, image resolution). It no longer owns span/MRD/bin counts.
- **Auto-save on success, preserve on crash.** `run_full()` saves the
  6 essential files to `run_dir` on success and deletes the temp workdir.
  On crash, the workdir is preserved for debugging.
- **n_z_slices in MCGPU is an input parameter**, not the output shape.
  It is computed as `2*n_rings - 1` and written to the `.in` file.
  The actual output z-planes (1293 for mcgpu_sample) are auto-detected
  from file size at parse time.
