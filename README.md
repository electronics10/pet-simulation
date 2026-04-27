# PET Simulation Pipeline — Setup & Structure

## Directory Layout

```
pet-simulation/
├── petsim/                   # Main library
│   ├── __init__.py
│   ├── phantom.py
│   ├── source.py
│   ├── scanner.py
│   ├── sinogram.py
│   ├── run.py
│   ├── materials.py
│   └── backends/
│       ├── __init__.py
│       ├── mcgpu.py
│       └── mcgpu_runtime.py
├── bin/                      # Compiled simulator binaries
│   └── MCGPU-PET.x          # Must be compiled from MCGPU-PET repo
├── materials/               # Cross-section files for MCGPU-PET
│   ├── air_5-515keV.mcgpu.gz
│   ├── water_5-515keV.mcgpu.gz
│   └── ...                  # Other .mcgpu.gz files from MCGPU-PET
├── runs/                    # Generated simulation data
│   ├── 0001/
│   │   ├── run.yaml         # Manifest (seed, backend config, timestamps)
│   │   ├── phantom.npz      # Voxelized geometry
│   │   ├── source.npz       # Activity distribution
│   │   ├── scanner.yaml     # Scanner hardware spec
│   │   └── sinogram.npz     # Measured sinogram (trues + scatter)
│   ├── 0002/
│   │   └── ...
│   └── ...
├── main.py                  # Example script
└── README.md
```

## Setup

### 1. Compile MCGPU-PET

Clone and compile the MCGPU-PET repository:

```bash
git clone https://github.com/DIDSR/MCGPU-PET
cd MCGPU-PET
make                        # Requires CUDA toolkit
cp MCGPU-PET.x ../../bin/   # Copy to pet-simulation/bin/
```

### 2. Copy Material Files

```bash
cp -r MCGPU-PET/sample_simulation/materials/* ../../materials/
```

The `materials/` directory should contain `.mcgpu.gz` files for:
- `air_5-515keV.mcgpu.gz`
- `water_5-515keV.mcgpu.gz`
- And any other tissue materials you plan to use

### 3. Install petsim

```bash
cd pet-simulation
pip install -e .
# or with uv:
uv pip install -e .
```

## Usage

### Basic Simulation

```python
from petsim import Phantom, Source, Scanner, Run
from petsim.backends import MCGPUBackend, MCGPUConfig

# Define the scene
phantom = Phantom.cube(shape=(9, 9, 9), voxel_size=(1.0, 1.0, 1.0), ...)
source = Source.with_total_activity(phantom, material="water", ...)
scanner = Scanner.from_preset("mcgpu_sample")
run = Run(phantom, source, scanner, seed=42)

# Run the simulation
backend = MCGPUBackend(
    executable="./bin/MCGPU-PET.x",
    materials_dir="./materials",
)
sinogram, result = backend.run_full(run, run_dir="./runs/0001")

# Done. ./runs/0001/ now has the 5 essential files.
```

### Loading a Saved Run

```python
from petsim import Run

run = Run.load("./runs/0001")
print(run.sinogram.total_trues)

# To reproduce it exactly:
old_config_dict = run.metadata.get("mcgpu_config")
if old_config_dict:
    from petsim.backends import MCGPUConfig
    config = MCGPUConfig(**old_config_dict)
    # Rerun with same parameters
```

## The Five Essential Files

Each `run_dir/` contains exactly five files (plus hidden `.tmp/` during execution):

- **`run.yaml`**: Manifest with seed, backend name, MCGPUConfig, timestamps
- **`phantom.npz`**: Voxel geometry (material IDs + densities)
- **`source.npz`**: Activity distribution
- **`scanner.yaml`**: Hardware spec (radius, rings, energy window, ...)
- **`sinogram.npz`**: Output measurement (trues + scatter arrays)

These five files are sufficient to:
- Reproduce the exact simulation years later
- Analyze the data
- Train ML models
- Cross-validate with other simulators

## Structure: Hardware vs. Runtime

**`Scanner`** (in `scanner.py`) = pure hardware spec:
- Detector radius, axial length
- Number of rings, crystals per ring
- Energy window, energy resolution
- Crystal-level geometry (for GATE)

**`MCGPUConfig`** (in `backends/mcgpu.py`) = MCGPU-PET runtime:
- Sinogram binning: span, MRD, n_radial_bins, n_angular_bins
- GPU knobs: gpu_number, threads_per_block, density_scale_factor
- Output control: PSF size, coincidence/output reporting modes

The same Scanner can be simulated with different MCGPUConfigs (different binning). The manifest records which config was used, enabling exact reproduction.