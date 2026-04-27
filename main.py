"""
Minimal working script to generate one PET sinogram.

This script demonstrates the complete pipeline:
  1. Define phantom (geometry)
  2. Define source (activity)
  3. Define scanner (detector)
  4. Bundle into Run
  5. Execute with MCGPUBackend
  6. Parse output sinogram
"""

from petsim import Phantom, Source, Scanner, Run
from petsim.backends import MCGPUBackend

WORK_DIR = "./runs/test_01"

# ============================================================================
# Step 1: Define the phantom (geometry + materials)
# ============================================================================
phantom = Phantom.cube(
    shape=(9, 9, 9),
    voxel_size=(1.0, 1.0, 1.0),       # cm per voxel
    inner_material="water",
    inner_density=1.0,
    outer_material="air",
    outer_density=0.0012,
    inner_size_vox=5,
)
print(f"Phantom: {phantom}")

# ============================================================================
# Step 2: Define the source (activity distribution)
# ============================================================================
source = Source.with_total_activity(
    phantom,
    material="water",
    total_activity_Bq=1e6,
    isotope="F18",
)
print(f"Source: {source}")

# ============================================================================
# Step 3: Define the scanner (detector geometry + binning)
# ============================================================================
scanner = Scanner.from_preset("mcgpu_sample")

# CRITICAL: The preset has n_z_slices=159 (input parameter to MCGPU-PET),
# but MCGPU-PET's span compression produces 1293 output slices.
# This must match what the simulator actually outputs, or parsing fails.
# scanner.n_z_slices = 1293
scanner.n_z_slices = 18303


print(f"Scanner: {scanner}")
print(f"Sinogram shape: {scanner.sinogram_shape}")

# ============================================================================
# Step 4: Bundle into a Run
# ============================================================================
run = Run(phantom=phantom, source=source, scanner=scanner, seed=42)
print(f"Run: {run}")

# ============================================================================
# Step 5: Execute the simulation
# ============================================================================
backend = MCGPUBackend(
    executable="./MCGPU-PET/MCGPU-PET.x",
    materials_dir="./MCGPU-PET/sample_simulation/materials",
)

print("\n" + "="*70)
print("Running MCGPU-PET simulation...")
print("="*70)

sinogram, result = backend.run_full(run, workdir=WORK_DIR)

print("\n" + "="*70)
print("Simulation complete!")
print("="*70)

# ============================================================================
# Step 6: Inspect the output
# ============================================================================
print(f"\nSinogram: {sinogram}")
print(f"  Total trues: {sinogram.total_trues}")
print(f"  Total scatter: {sinogram.total_scatter}")
if sinogram.scatter_fraction is not None:
    print(f"  Scatter fraction: {sinogram.scatter_fraction:.1%}")
print(f"\nWall time: {result.wall_time_s:.2f} s")
print(f"Return code: {result.returncode}")

# ============================================================================
# Step 7: Save for later use
# ============================================================================
run.sinogram = sinogram
run.save(WORK_DIR + "/saved")
print(f"\nSaved run to" + WORK_DIR + "/saved")