# """Point source debug test for the GATE histogrammer.

# Setup
# -----
# - Phantom: 31×31×31 voxels of pure air at 1 mm isotropic spacing.
#   No attenuation, so scatter should be ~0.
# - Source: single hot voxel at (+10 mm, 0, 0) relative to phantom center.
#   Phantom center sits on the world origin (the MHD writer centers it),
#   so the source is at world coords (+10 mm, 0, 0).

# Expected sinogram (with a correct LOR-based histogrammer)
# ---------------------------------------------------------
# For a point source at (s_x, 0, 0) on the z-axis, the (angular, radial)
# sinogram at the central ring pair (ring1 = ring2 = n_rings/2) traces a
# half-cosine:

#     radial_offset(view_angle) = s_x * cos(view_angle - π/2)

# Concretely, with view ∈ [0, π) and s_x = +10 mm:
# - view = 0     → radial = +10 mm   (peak on the right side)
# - view = π/2   → radial =   0 mm   (LOR through origin)
# - view = π     → radial = -10 mm   (peak on the left side, just inside the bin range)

# In a 32-bin radial axis spanning [-R_eff, +R_eff] ≈ [-67, +67] mm,
# ±10 mm corresponds to bins ~16 ± 2.4 = ~13.6 and ~18.4. The trace should
# be a smooth diagonal sweep across the (angular, radial) image, NOT a
# solid blob centered on radial=0 and NOT shifted to one side.

# Other things to verify
# ----------------------
# - Total scatter: ≈ 0 (pure air phantom — nothing to scatter off).
# - Total trues: thousands (depends on acquisition time and source rate).
# - Counts concentrated in ring1 = ring2 = center_ring; some bleed into
#   adjacent ring pairs because crystals have axial extent.
# - Random rate: small fraction (< few %).
# """

# import numpy as np
# import matplotlib.pyplot as plt

# from petsim import Phantom, Source, Scanner, SinogramBinning, Run
# from petsim.backends.gate import GATEBackend


# # ---------------------------------------------------------------------
# # 1. Build the test scene
# # ---------------------------------------------------------------------

# phantom = Phantom.uniform(
#     shape=(31, 31, 31),
#     voxel_size=(0.1, 0.1, 0.1),     # 1 mm voxels (cm units)
#     material="air",
#     density=0.0012,
# )

# # Phantom center (corner-relative) is at (1.55, 1.55, 1.55) cm.
# # Adding +10 mm in x puts the hot voxel at (2.55, 1.55, 1.55) cm.
# source = Source.zeros(phantom, isotope="F18").add_hot_spot(
#     position_cm=(2.55, 1.55, 1.55),
#     activity_Bq=1e6,
#     radius_cm=0.0,                  # exactly one voxel
# )

# scanner = Scanner.from_preset("bruker_albira", acquisition_time_s=0.5)
# binning = SinogramBinning.default_for(scanner)

# run = Run(
#     phantom=phantom,
#     source=source,
#     scanner=scanner,
#     binning=binning,
#     seed=42,
# )

# print(f"Phantom: {phantom}")
# print(f"Source:  {source}")
# print(f"Scanner: {scanner}")
# print(f"Binning: {binning}")
# print()

# # ---------------------------------------------------------------------
# # 2. Run GATE
# # ---------------------------------------------------------------------

# backend = GATEBackend(materials_db="./GateMaterials.db")
# print("Running GATE point-source test...\n")
# sino, _ = backend.run_full(run, run_dir="./runs/gate_point_source")

# print()
# print(f"Sinogram: {sino}")
# print(f"  shape:   {sino.shape}")
# print(f"  axes:    {sino.axes}")
# print(f"  trues:   {sino.total_trues:,}")
# print(f"  scatter: {sino.total_scatter:,}  (expect ~0 in pure air)")

# # ---------------------------------------------------------------------
# # 3. Visualize
# # ---------------------------------------------------------------------

# n_rings = scanner.n_rings
# center_ring = n_rings // 2

# # Direct plane (ring1 == ring2 == center_ring)
# direct_plane = sino.trues[center_ring, center_ring]

# # Total over all ring pairs (axial summary)
# all_pairs = sino.trues.sum(axis=(0, 1))

# # Ring-pair occupancy (where in (ring1, ring2) do counts live?)
# ring_occupancy = sino.trues.sum(axis=(2, 3))

# fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# im0 = axes[0].imshow(direct_plane, aspect="auto", cmap="hot", origin="lower")
# axes[0].set_xlabel("Radial bin")
# axes[0].set_ylabel("Angular bin (view)")
# axes[0].set_title(
#     f"Direct plane: ring1 = ring2 = {center_ring}\n"
#     f"(should show a half-cosine sweep)"
# )
# plt.colorbar(im0, ax=axes[0])

# im1 = axes[1].imshow(all_pairs, aspect="auto", cmap="hot", origin="lower")
# axes[1].set_xlabel("Radial bin")
# axes[1].set_ylabel("Angular bin (view)")
# axes[1].set_title("Summed over all ring pairs")
# plt.colorbar(im1, ax=axes[1])

# im2 = axes[2].imshow(ring_occupancy, aspect="auto", cmap="hot", origin="lower")
# axes[2].set_xlabel("ring2")
# axes[2].set_ylabel("ring1")
# axes[2].set_title(
#     f"Counts vs (ring1, ring2)\n(should peak on the diagonal at {center_ring})"
# )
# plt.colorbar(im2, ax=axes[2])

# plt.tight_layout()
# out_path = "./runs/gate_point_source/sinogram_diagnostic.png"
# plt.savefig(out_path, dpi=150)
# print(f"\nSaved diagnostic plot to {out_path}")

# # ---------------------------------------------------------------------
# # 4. Numerical sanity check
# # ---------------------------------------------------------------------

# R_eff = scanner.detector_radius_cm * 10.0 + (
#     scanner.crystal_size_mm[0] / 2.0 if scanner.crystal_size_mm else 0.0
# )
# n_ang = binning.n_angular_bins
# n_rad = binning.n_radial_bins

# print()
# print(f"R_eff (crystal-center radius): {R_eff:.2f} mm")
# print(f"Radial bin width: {2 * R_eff / n_rad:.2f} mm")
# print(f"Bin index for radial = +10 mm: {((10 / R_eff + 1) / 2 * n_rad):.1f}")
# print(f"Bin index for radial =   0 mm: {((0  / R_eff + 1) / 2 * n_rad):.1f}")
# print(f"Bin index for radial = -10 mm: {((-10/ R_eff + 1) / 2 * n_rad):.1f}")
# print()
# print("In the direct plane image, the bright trace should pass through")
# print(f"those three radial bins as the angular bin goes from 0 to {n_ang - 1}.")

# Sinusoidal check
# import numpy as np
# from petsim import Sinogram
# sino = Sinogram.load("./runs/gate_point_source/sinogram.npz")

# # Sum over rings to get (n_angular, n_radial)
# ang_rad = sino.trues.sum(axis=(0, 1))

# # Compute centroid radial bin per view
# radial_bins = np.arange(32)
# for v in range(0, 32, 4):
#     counts = ang_rad[v]
#     if counts.sum() > 0:
#         centroid = (counts * radial_bins).sum() / counts.sum()
#         print(f"view={v:2d}  total={counts.sum():.0f}  centroid_radial={centroid:.2f}")

# ROOT debug
# Read raw GATE output to inspect crystal indices
import uproot
import numpy as np

with uproot.open("./runs/gate_point_source/tmp/gate_output.root") as f:
    coinc = f["Coincidences"].arrays(
        ["PostPosition1_X", "PostPosition1_Y", "PostPosition2_X", "PostPosition2_Y"],
        library="np"
    )

x1, y1 = coinc["PostPosition1_X"], coinc["PostPosition1_Y"]
x2, y2 = coinc["PostPosition2_X"], coinc["PostPosition2_Y"]

# Crystal indices from azimuthal angle
n_crystals = 64
a1 = np.arctan2(y1, x1) % (2*np.pi)
a2 = np.arctan2(y2, x2) % (2*np.pi)
c1 = (a1 / (2*np.pi) * n_crystals).astype(int)
c2 = (a2 / (2*np.pi) * n_crystals).astype(int)

# Histogram of c2 - c1 mod n_crystals
diff = (c2 - c1) % n_crystals
print("c2-c1 distribution (should peak at 32 for centered source, shift for displaced):")
unique, counts = np.unique(diff, return_counts=True)
for u, c in zip(unique, counts):
    bar = "#" * (c * 50 // counts.max())
    print(f"  diff={u:3d}: {c:5d}  {bar}")

# Also check: do hits land EXACTLY at crystal centers?
# Distance from origin should be exactly R_eff = 67 mm if so.
r1 = np.sqrt(x1**2 + y1**2)
print(f"\nHit radius (should be ~67mm if snapped):")
print(f"  mean: {r1.mean():.2f} mm, std: {r1.std():.2f} mm")
print(f"  min:  {r1.min():.2f} mm, max:  {r1.max():.2f} mm")