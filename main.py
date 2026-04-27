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

from pathlib import Path
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
# Use the preset as-is; no manual n_z_slices override needed.
# The parser will auto-detect the output shape.
print(f"Scanner: {scanner}")

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
print(f"  Sinogram shape: {sinogram.shape}")
print(f"  Total trues: {sinogram.total_trues}")
print(f"  Total scatter: {sinogram.total_scatter}")
if sinogram.scatter_fraction is not None:
    print(f"  Scatter fraction: {sinogram.scatter_fraction:.1%}")

print(f"\nWall time: {result.wall_time_s:.2f} s")
print(f"Return code: {result.returncode}")

# ============================================================================
# Step 7: Access the arrays directly for ML
# ============================================================================
print("\n" + "="*70)
print("Arrays for ML training:")
print("="*70)
print(f"trues shape:   {sinogram.trues.shape}, dtype: {sinogram.trues.dtype}")
print(f"scatter shape: {sinogram.scatter.shape if sinogram.scatter is not None else None}")
print(f"scatter dtype: {sinogram.scatter.dtype if sinogram.scatter is not None else None}")

# Example: use directly for training
trues_array = sinogram.trues      # shape (18303, 168, 147)
scatter_array = sinogram.scatter  # shape (18303, 168, 147)
print(f"\nReady for training:")
print(f"  Input (total):  {trues_array.shape}")
print(f"  Target (scatter): {scatter_array.shape}")
print(f"  You can now train: model.fit(trues, scatter)")