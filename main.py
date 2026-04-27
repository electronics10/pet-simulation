"""
Minimal working example with the cleanly-separated API.

Three layers, all backend-agnostic:
  1. Scanner       — hardware (radius, rings, energy window)
  2. SinogramBinning — layout choice (span, MRD, n_radial, n_angular)
  3. Run           — bundles everything, ready for any backend

Backend-specific runtime (MCGPUConfig, GATEConfig) is the only thing
backends define on their own.
"""

from petsim import Phantom, Source, Scanner, SinogramBinning, Run
from petsim.backends import MCGPUBackend

# ============================================================================
# Define the simulation — backend-agnostic
# ============================================================================
phantom = Phantom.cube(
    shape=(9, 9, 9),
    voxel_size=(1.0, 1.0, 1.0),
    inner_material="water",
    inner_density=1.0,
    outer_material="air",
    outer_density=0.0012,
    inner_size_vox=5,
)

source = Source.with_total_activity(
    phantom, material="water", total_activity_Bq=1e6, isotope="F18"
)

# Hardware: backend-agnostic
scanner = Scanner.from_preset("mcgpu_sample")

# Layout: backend-agnostic. The "default_for" factory works for any scanner.
# binning = SinogramBinning.default_for(scanner)  # works for any scanner

# For the mcgpu_sample, use the canonical layout it was designed for:
from petsim.sinogram_binning import preset_binning
binning = preset_binning("mcgpu_sample")

# Bundle
run = Run(
    phantom=phantom,
    source=source,
    scanner=scanner,
    binning=binning,
    seed=42,
)
print(f"Scanner: {scanner}")
print(f"Binning: {binning}")
print(f"Run: {run}")

# ============================================================================
# Run with MCGPU
# ============================================================================
backend = MCGPUBackend(
    executable="./bin/MCGPU-PET.x",
    materials_dir="./materials",
)

print("\nRunning simulation...")
sinogram, result = backend.run_full(run, run_dir="./runs/0001")

print(f"\nSinogram: {sinogram}")
print(f"  shape:           {sinogram.shape}")
print(f"  total trues:     {sinogram.total_trues}")
print(f"  total scatter:   {sinogram.total_scatter}")
print(f"  scatter fraction: {sinogram.scatter_fraction:.1%}")
print(f"\nWall time: {result.wall_time_s:.2f} s")
print(f"Saved to ./runs/0001/")

# ============================================================================
# Bruker example — same pipeline, different scanner+binning
# ============================================================================
print("\n" + "="*70)
print("Same pipeline with Bruker scanner — just change scanner + binning:")
print("="*70)

bruker_scanner = Scanner.from_preset("bruker_albira")
bruker_binning = SinogramBinning.default_for(bruker_scanner)
print(f"Bruker scanner: {bruker_scanner}")
print(f"Bruker binning: {bruker_binning}")
# (Not actually running this — would need a Bruker-sized phantom.
#  The point is the API is identical regardless of scanner.)