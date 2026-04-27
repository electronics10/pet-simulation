"""
Minimal working script with the refactored Scanner + MCGPUConfig split.

Scanner = hardware (radius, rings, energy window, ...)
MCGPUConfig = MCGPU's runtime knobs + sinogram binning (span, MRD, bins)

Use the default MCGPUConfig for the mcgpu_sample scanner. For the Bruker
scanner, override the binning to something sensible for 24 rings.
"""

from petsim import Phantom, Source, Scanner, Run
from petsim.backends import MCGPUBackend, MCGPUConfig

WORK_DIR = "./runs/test_01"

# ============================================================================
# Phantom + Source (unchanged)
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
# Scanner (hardware only — no binning fields anymore)
# ============================================================================
scanner = Scanner.from_preset("mcgpu_sample")
print(f"Scanner: {scanner}")

# ============================================================================
# MCGPUConfig (binning + runtime knobs live here now)
# ============================================================================
# Defaults are tuned for mcgpu_sample, so we just use them as-is.
# For other scanners, override:
#   config = MCGPUConfig(span=3, max_ring_difference=23,
#                        n_radial_bins=32, n_angular_bins=32)
config = MCGPUConfig()
print(f"Config: span={config.span}, MRD={config.max_ring_difference}, "
      f"bins=({config.n_angular_bins}, {config.n_radial_bins})")

# ============================================================================
# Bundle and run
# ============================================================================
run = Run(phantom=phantom, source=source, scanner=scanner, seed=42)

backend = MCGPUBackend(
    executable="./MCGPU-PET/MCGPU-PET.x",
    materials_dir="./MCGPU-PET/sample_simulation/materials",
)

print("\nRunning MCGPU-PET simulation...")
sinogram, result = backend.run_full(run, workdir=WORK_DIR, config=config)

# ============================================================================
# Inspect output
# ============================================================================
print(f"\nSinogram: {sinogram}")
print(f"  shape: {sinogram.shape}")
print(f"  total trues: {sinogram.total_trues}")
print(f"  total scatter: {sinogram.total_scatter}")
print(f"  scatter fraction: {sinogram.scatter_fraction:.1%}")
print(f"\nWall time: {result.wall_time_s:.2f} s")

# ============================================================================
# Save the full run for later use
# ============================================================================
run.sinogram = sinogram
run.save(WORK_DIR + "/saved")
print(f"\nSaved run to {WORK_DIR}/saved/")

# Arrays for ML training
print(f"\nReady for training:")
print(f"  trues:   {sinogram.trues.shape} {sinogram.trues.dtype}")
print(f"  scatter: {sinogram.scatter.shape} {sinogram.scatter.dtype}")