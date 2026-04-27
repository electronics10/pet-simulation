"""
MCGPU-PET backend: translate petsim objects to MCGPU-PET input files.

This module provides the input-side of the MCGPU-PET backend:

  - `write_vox(phantom, source, path)` produces a .vox file that MCGPU-PET
    can parse directly. Output is byte-exact to MCGPU-PET's reference
    phantom format.

  - `write_in(run, materials, vox_filename, config, path)` produces the
    .in control file MCGPU-PET reads at startup. Functionally correct
    for MCGPU-PET's section-based parser.

The executable invocation and output parsing live in mcgpu_runtime.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..materials import MaterialRegistry
from ..phantom import Phantom
from ..run import Run
from ..source import Source


# =====================================================================
# .vox file format
# =====================================================================

VOX_HEADER_TEMPLATE = """\
[SECTION VOXELS HEADER v.2008-04-13]
{nx} {ny} {nz}   No. OF VOXELS IN X,Y,Z
{dx} {dy} {dz}   VOXEL SIZE (cm) ALONG X,Y,Z
 1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED
 2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED
 1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)
[END OF VXH SECTION]  # MCGPU-PET voxel format: Material  Density  Activity
"""


def write_vox(
    phantom: Phantom,
    source: Source,
    path: str | Path,
) -> None:
    """Write a .vox file combining phantom geometry and source activity.

    The output format follows MCGPU-PET's reference phantom_9x9x9cm.vox
    byte-for-byte. Each voxel becomes one line:

        {material_id} {density_g_per_cm3} {activity_Bq}

    Iteration order is X fastest (innermost), then Y, then Z (slowest).
    """
    if not source.matches(phantom):
        raise ValueError(
            f"Source grid {source.shape} {source.voxel_size} cm does not "
            f"match phantom grid {phantom.shape} {phantom.voxel_size} cm"
        )

    path = Path(path)
    nx, ny, nz = phantom.shape
    dx, dy, dz = phantom.voxel_size

    parts: list[str] = [
        VOX_HEADER_TEMPLATE.format(
            nx=nx, ny=ny, nz=nz,
            dx=float(dx), dy=float(dy), dz=float(dz),
        )
    ]

    mat_ids = phantom.material_ids
    dens = phantom.densities
    acts = source.activity_Bq

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                mid = int(mat_ids[x, y, z])
                den = float(dens[x, y, z])
                act = float(acts[x, y, z])
                parts.append(f"{mid} {den} {act}\n")
            parts.append("\n")  # end of X-cycle
        parts.append("\n")      # end of Y-cycle
    parts.append("\n")          # trailing blank at EOF

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts))


# =====================================================================
# .in file format and configuration
# =====================================================================
#
# MCGPU-PET-specific runtime parameters AND sinogram binning live in this
# config. Binning belongs here (not in Scanner) because it's a choice
# about how to histogram coincidences, not a property of the scanner
# hardware. Two MCGPU runs of the same scanner can use different binning.


@dataclass
class MCGPUConfig:
    """MCGPU-PET-specific runtime parameters and sinogram binning.

    Defaults mirror the MCGPU-PET sample simulation, which is designed
    for the mcgpu_sample scanner preset (80 rings, 336 crystals/ring).
    For other scanners, you'll typically want to override at least
    n_radial_bins, n_angular_bins, span, and max_ring_difference to
    sensible values for that geometry.
    """

    # ---- GPU / perf knobs --------------------------------------------
    gpu_number: int = 0
    threads_per_block: int = 32        # must be multiple of 32
    density_scale_factor: float = 1.0  # 1.0 = no scaling

    # ---- Phase-space output ------------------------------------------
    psf_filename: str = "MCGPU_PET.psf"
    psf_size: int = 150_000_000

    # Coincidence filter: 0=both trues+scatter, 1=trues only, 2=scatter only
    report_coincidence_mode: int = 0

    # Output file selection: 0=both PSF+sinogram, 1=PSF only, 2=sinogram only
    report_output_mode: int = 0

    # ---- Dose tally --------------------------------------------------
    tally_material_dose: bool = True
    tally_voxel_dose: bool = False
    dose_output_filename: str = "mc-gpu_dose.dat"

    # ---- Sinogram binning --------------------------------------------
    # These control how MCGPU-PET histograms coincidences. They are
    # intentionally independent of the Scanner: the same scanner can
    # produce sinograms with different binning conventions.
    n_radial_bins: int = 147           # NRAD
    n_angular_bins: int = 168          # NANGLES = NCRYSTALS/2 typically
    span: int = 11                     # Michelogram span
    max_ring_difference: int = 79      # MRD

    # ---- Output voxel image resolution -------------------------------
    image_resolution: int = 128   # bins per axis in the emission image
    n_energy_bins: int = 700      # bins in the detected-energy spectrum


# The .in file uses `#` for inline comments and a `[SECTION ... v.YYYY-MM-DD]`
# header style. MCGPU-PET's parser scans for the section strings, so field
# order within a section matters but surrounding whitespace doesn't.

IN_TEMPLATE = """\
# >>>> INPUT FILE FOR MCGPU-PET v0.1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
#  -- Generated by petsim ({petsim_tag})
#

#[SECTION SIMULATION CONFIG v.2016-07-05]
{seed}                               # RANDOM SEED (ranecu PRNG; 0 = seed from current time)
{gpu_number}                               # GPU NUMBER TO USE WHEN MPI IS NOT USED, OR TO BE AVOIDED IN MPI RUNS
{threads_per_block}                              # GPU THREADS PER CUDA BLOCK (multiple of 32)
{density_scale_factor}                             # FACTOR TO SCALE THE INPUT MATERIAL DENSITY (usually 1; very small = all air)


#[SECTION SOURCE PET SCAN v.2017-03-14]
{acquisition_time_s}                             # TOTAL PET SCAN ACQUISITION TIME [seconds]
 {mean_life_s}                        # ISOTOPE MEAN LIFE [s]
   1    0.0   # !!INPUT NOT USED: Activity read from input voxel file as a 3rd column after material and density!!   # TABLE MATERIAL NUMBER AND VOXEL ACTIVITY [Bq]: 1==1st_material ; 0==end_of_list
   0    0.0


#[SECTION PHASE SPACE FILE v.2016-07-05]
 {psf_filename}                  # OUTPUT PHASE SPACE FILE FILE NAME
 0.0  0.0  0.0  {axial_fov_cm}  {detector_radius_signed}   # CYLINDRIC DETECTOR CENTER, HEIGHT, AND RADIUS: X, Y, Z, H, RADIUS [cm] (IF RADIUS<0: AUTO-CENTER ON VOXELIZED GEOMETRY)
 {psf_size}                      # PHASE SPACE FILE SIZE (MAXIMUM NUMBER OF ELEMENTS)
 {report_coincidence_mode}                              # REPORT TRUES (1), SCATTER (2), OR BOTH (0)
 {report_output_mode}                              # REPORT PSF (1), SINOGRAM (2) OR BOTH (0)


#[SECTION DOSE DEPOSITION v.2012-12-12]
{tally_material_dose}                             # TALLY MATERIAL DOSE? [YES/NO]
{tally_voxel_dose}                              # TALLY 3D VOXEL DOSE? [YES/NO]
{dose_output_filename}                 # OUTPUT VOXEL DOSE FILE NAME
  1  {nx}                        # VOXEL DOSE ROI: X-index min max (first voxel has index 1)
  1  {ny}                        # VOXEL DOSE ROI: Y-index min max
  1  {nz}                        # VOXEL DOSE ROI: Z-index min max


#[SECTION ENERGY PARAMETERS v.2019-04-25]
{energy_resolution}          # ENERGY RESOLUTION OF THE CRYSTALS (fractional FWHM at 511 keV)
{energy_low_eV}      # ENERGY WINDOW LOW (eV)
{energy_high_eV}      # ENERGY WINDOW HIGH (eV)


#[SECTION SINOGRAM PARAMETERS v.2019-04-25]
{axial_fov_cm} # AXIAL FIELD OF VIEW (FOVz) in cm
{n_rings}     # NUMBER OF ROWS (detector rings)
{n_crystals_per_ring}    # TOTAL NUMBER OF CRYSTALS (per ring)
{n_angular_bins}    # NUMBER OF ANGULAR BINS (NCRYSTALS/2)
{n_radial_bins}    # NUMBER OF RADIAL BINS
{n_z_slices}    # NUMBER OF Z SLICES
{image_resolution}    # IMAGE RESOLUTION (NUMBER OF BINS IN THE IMAGE)
{n_energy_bins}    # NUMBER OF ENERGY BINS (NE)
{max_ring_difference}     # MAXIMUM RING DIFFERENCE (MRD)
{span}     # SPAN


#[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]
{vox_filename}          # VOXEL GEOMETRY FILE (penEasy 2008 format; .gz accepted)


#[SECTION MATERIAL FILE LIST v.2009-11-30]
{material_files_block}
#
#
# >>>> END INPUT FILE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""


def write_in(
    run: Run,
    materials: MaterialRegistry,
    vox_filename: str,
    config: Optional[MCGPUConfig] = None,
    path: str | Path = "MCGPU-PET.in",
    petsim_tag: str = "petsim",
) -> None:
    """Write an MCGPU-PET .in control file from a Run + MaterialRegistry.

    The Scanner provides hardware specs (rings, geometry, energy window).
    The MCGPUConfig provides binning and runtime knobs. NUMBER OF Z SLICES
    is computed from the scanner's n_rings as 2*n_rings - 1.

    Parameters
    ----------
    run : Run
        Provides scanner, source, phantom, and seed.
    materials : MaterialRegistry
        Resolves the phantom's material names to .mcgpu.gz files.
    vox_filename : str
        Relative path (as written into the .in file) to the .vox file.
    config : MCGPUConfig, optional
        Backend-specific knobs and binning. Defaults are tuned for the
        mcgpu_sample scanner preset.
    path : str or Path
        Where to write the .in file.
    petsim_tag : str
        Free-form identifier written into the header comment.
    """
    if config is None:
        config = MCGPUConfig()

    scanner = run.scanner
    phantom = run.phantom
    source = run.source

    # --- Material file list, in material-ID order ---
    material_lines: list[str] = []
    for i, name in enumerate(phantom.material_names, start=1):
        if name not in materials:
            raise KeyError(
                f"Material {name!r} (id {i}) not in MaterialRegistry; "
                f"available: {materials.names()}"
            )
        material = materials[name]
        rel_path = f"./materials/{material.mcgpu_file}"
        material_lines.append(f"{rel_path}             # {i}")
    material_files_block = "\n".join(material_lines)

    # --- Energy window in eV (MCGPU-PET's native unit) ---
    e_low_eV, e_high_eV = scanner.energy_window_eV

    # --- Seed: 0 means "use time-based" per MCGPU-PET convention ---
    seed_value = run.seed if run.seed is not None else 0

    # --- Detector radius: negate so MCGPU auto-centers on voxel geometry ---
    detector_radius_signed = -float(scanner.detector_radius_cm)

    # --- Dose ROI spans the whole phantom ---
    nx, ny, nz = phantom.shape

    # --- NUMBER OF Z SLICES is the input parameter to MCGPU-PET's
    #     rebinning algorithm. The standard value is 2*n_rings - 1.
    #     The actual output sinogram has many more z-planes after span
    #     compression — that's auto-detected at parse time.
    n_z_slices_input = 2 * scanner.n_rings - 1

    content = IN_TEMPLATE.format(
        petsim_tag=petsim_tag,
        seed=seed_value,
        gpu_number=config.gpu_number,
        threads_per_block=config.threads_per_block,
        density_scale_factor=float(config.density_scale_factor),
        acquisition_time_s=float(scanner.acquisition_time_s),
        mean_life_s=float(source.mean_lifetime_s),
        psf_filename=config.psf_filename,
        axial_fov_cm=float(scanner.detector_axial_length_cm),
        detector_radius_signed=detector_radius_signed,
        psf_size=config.psf_size,
        report_coincidence_mode=config.report_coincidence_mode,
        report_output_mode=config.report_output_mode,
        tally_material_dose="YES" if config.tally_material_dose else "NO",
        tally_voxel_dose="YES" if config.tally_voxel_dose else "NO",
        dose_output_filename=config.dose_output_filename,
        nx=nx, ny=ny, nz=nz,
        energy_resolution=float(scanner.energy_resolution),
        energy_low_eV=float(e_low_eV),
        energy_high_eV=float(e_high_eV),
        n_rings=scanner.n_rings,
        n_crystals_per_ring=scanner.n_crystals_per_ring,
        n_angular_bins=config.n_angular_bins,
        n_radial_bins=config.n_radial_bins,
        n_z_slices=n_z_slices_input,
        image_resolution=config.image_resolution,
        n_energy_bins=config.n_energy_bins,
        max_ring_difference=config.max_ring_difference,
        span=config.span,
        vox_filename=vox_filename,
        material_files_block=material_files_block,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)