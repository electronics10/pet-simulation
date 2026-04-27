"""
Minimal working example with the cleanly-separated API.

Three layers, all backend-agnostic:
  1. Scanner         — hardware (radius, rings, energy window)
  2. SinogramBinning — layout choice (span, MRD, n_radial, n_angular)
  3. Run             — bundles everything, ready for any backend

Backend-specific runtime (MCGPUConfig, GATEConfig) is the only thing
backends define on their own.
"""

from petsim import Phantom, Source, Scanner, SinogramBinning, Run
from petsim.backends import MCGPUBackend
from petsim.backends.gate import GATEBackend
from petsim.sinogram_binning import preset_binning

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

# ============================================================================
# MCGPU-PET run — mcgpu_sample scanner
# ============================================================================
scanner = Scanner.from_preset("mcgpu_sample")
binning = preset_binning("mcgpu_sample")
run = Run(phantom=phantom, source=source, scanner=scanner, binning=binning, seed=42)

print(f"Scanner: {scanner}")
print(f"Binning: {binning}")

mcgpu_backend = MCGPUBackend(
    executable="./bin/MCGPU-PET.x",
    materials_dir="./materials",
)
print("\nRunning MCGPU-PET simulation...")
sinogram, result = mcgpu_backend.run_full(run, run_dir="./runs/mcgpu_0001")

print(f"Sinogram: {sinogram}")
print(f"  shape:            {sinogram.shape}")
print(f"  total trues:      {sinogram.total_trues}")
print(f"  total scatter:    {sinogram.total_scatter}")
print(f"  scatter fraction: {sinogram.scatter_fraction:.1%}")
print(f"  wall time:        {result.wall_time_s:.2f} s")
print(f"Saved to ./runs/mcgpu_0001/")

# ============================================================================
# GATE run — Bruker scanner (same phantom, different scanner + binning)
# ============================================================================
bruker_scanner = Scanner.from_preset("bruker_albira")
bruker_binning = SinogramBinning.default_for(bruker_scanner)
bruker_run = Run(
    phantom=phantom,
    source=source,
    scanner=bruker_scanner,
    binning=bruker_binning,
    seed=42,
)

print(f"\nScanner: {bruker_scanner}")
print(f"Binning: {bruker_binning}")

gate_backend = GATEBackend(materials_db="./GateMaterials.db")
print("\nRunning GATE simulation...")
gate_sinogram, gate_result = gate_backend.run_full(
    bruker_run, run_dir="./runs/gate_0001"
)

print(f"Sinogram: {gate_sinogram}")
print(f"  shape:            {gate_sinogram.shape}")
print(f"  total trues:      {gate_sinogram.total_trues}")
print(f"  wall time:        {gate_result.wall_time_s:.2f} s")
print(f"Saved to ./runs/gate_0001/")