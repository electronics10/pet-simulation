from petsim import*
from petsim.sinogram_binning import preset_binning
from petsim.backends import*

# # Define scanner
# scanner = Scanner.from_preset("mcgpu_sample")
# print(scanner)

# # Define phantom
# phantom = Phantom.sphere(
#     shape = (3, 3, 3),
#     voxel_size = (0.1, 0.1, 0.1),
#     radius_cm = 0.1,
#     inner_material = "water",
#     inner_density = 1
# )
# phantom.show_slices(fig_name="phantom")
# print(phantom)

# # Define source corresponding to phantom
# source = Source.with_total_activity(phantom=phantom, material="water", total_activity_Bq=1e6)
# print(source)

# # mcgpu run
# binning = preset_binning("mcgpu_sample")
# run = Run(phantom=phantom, source=source, scanner=scanner, binning=binning, seed=42)
# print(run)

# # mcgpu_backend = MCGPUBackend(
# #     executable="./bin/MCGPU-PET.x",
# #     materials_dir="./materials",
# # )
# # print("\nRunning MCGPU-PET simulation...")
# # sinogram, result = mcgpu_backend.run_full(run, run_dir="./runs/mcgpu_0001")

# # print(f"Sinogram: {sinogram}")
# # print(f"  shape:            {sinogram.shape}")
# # print(f"  total trues:      {sinogram.total_trues}")
# # print(f"  total scatter:    {sinogram.total_scatter}")
# # print(f"  scatter fraction: {sinogram.scatter_fraction:.1%}")
# # print(f"  wall time:        {result.wall_time_s:.2f} s")
# # print(f"Saved to ./runs/mcgpu_0001/")

phantom = Phantom.load("runs/mcgpu_0001/phantom.npz")
print(phantom)

#sdfs 