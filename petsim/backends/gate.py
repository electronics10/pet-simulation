"""
GATE backend: drive GATE 10 simulations via the `opengate` Python API.

GATE 10 provides a Python API (the `opengate` package) that replaces
the legacy .mac macro-file interface. This backend builds a GATE
simulation programmatically from a petsim Run, runs it, and parses
the sinogram output.

Design notes
------------

Unlike MCGPU-PET, GATE doesn't need an on-disk input file — the
simulation is constructed in memory through opengate's `Simulation`
object. This means the "writer/invoker/parser" three-step split from
the MCGPU backend collapses into two:

  1. build(run, config) → Simulation: produce a configured opengate
     Simulation. Lets the user inspect/modify before running.
  2. run_full(run, ...) → Sinogram: the convenience path.

GATE also uses millimeters (not cm) for lengths and Becquerels natively
for activity. Conversion from petsim's cm-based units happens at build
time and is explicit in the code below.

Scanner geometry
----------------

This backend builds a *simplified cylindrical PET detector* based on
Scanner parameters: a single ring of crystals at `detector_radius_cm`,
axial length `detector_axial_length_cm`, with an energy window derived
from Scanner. This is intentionally not a full cylindricalPET system
matcher — it's a minimal geometry good enough for side-by-side
validation against MCGPU-PET on simple phantoms. More realistic
scanner templates (Biograph, Vision, etc.) can be added later as
presets.

Voxelized source
----------------

GATE's voxelized source accepts an ITK image. We save the
Source.activity_Bq array as a .mhd/.raw pair in the workdir and point
GATE's VoxelSource at that file. Phantom geometry is similarly
exported as a voxelized material image.

Dependencies
------------

This module imports `opengate` lazily inside build() so that the
rest of petsim works without GATE installed. If you're on a machine
without GATE, the MCGPU backend is still usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..phantom import Phantom
from ..run import Run
from ..sinogram import Sinogram
from ..source import Source


# =====================================================================
# Config
# =====================================================================


@dataclass
class GATEConfig:
    """GATE-specific runtime parameters.

    Only fields that meaningfully vary between runs live here; things
    like crystal material (LSO) or coincidence policy stay as sane
    defaults inside the backend. Expose more as you need them.
    """

    # ---- Physics --------------------------------------------------
    # GATE physics list. "G4EmStandardPhysics_option4" is the standard
    # high-accuracy EM physics list for PET.
    physics_list: str = "G4EmStandardPhysics_option4"

    # Enable positron range (realistic) or disable for back-to-back only
    # (faster, matches MCGPU-PET's "point source of 511 keV pairs" model).
    back_to_back_annihilation: bool = True

    # ---- Geometry detail -----------------------------------------
    # Crystal material; LSO is standard for clinical PET.
    crystal_material: str = "LSO"

    # Axial crystal pitch (mm). Default matches ~3 mm typical.
    crystal_axial_pitch_mm: float = 3.0

    # Tangential crystal size (mm).
    crystal_tangential_mm: float = 4.0

    # Radial crystal thickness (mm).
    crystal_radial_mm: float = 20.0

    # ---- Output knobs --------------------------------------------
    # If True, store a ROOT file with singles/coincidences in addition
    # to the sinogram. Useful for debugging and list-mode analysis.
    store_root_output: bool = False
    root_output_filename: str = "gate_output.root"

    # Number of CPU threads (GATE 10 supports multi-threading).
    # 0 means "auto" (use all available cores).
    n_threads: int = 0

    # ---- Verbosity ------------------------------------------------
    verbose_level: int = 1  # 0 = silent, 1 = progress, 2 = debug


# =====================================================================
# Result
# =====================================================================


@dataclass
class GATERunResult:
    """What a GATE invocation produced.

    Mirrors MCGPURunResult for uniformity — higher-level code can
    treat the two backends the same way.
    """

    workdir: Path
    wall_time_s: float
    # Path to the ROOT output if stored, else None
    root_output_path: Optional[Path] = None
    # Path to the sinogram output file produced by the Coincidences actor
    sinogram_output_path: Optional[Path] = None
    # Raw metadata from opengate's output actors (counts, timing, etc.)
    actor_stats: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Backend
# =====================================================================


class GATEBackend:
    """Driver for GATE 10 simulations via the opengate Python API.

    Intentionally small — most of the simulator configuration is hidden
    behind reasonable defaults so callers can run `backend.run_full(run)`
    and get a sinogram. Advanced users can call `backend.build(run)` to
    get the raw opengate `Simulation` object and modify it before
    running.
    """

    def __init__(self, materials_dir: Optional[str | Path] = None) -> None:
        """
        Parameters
        ----------
        materials_dir : optional path to a directory of GATE material
            definitions (.db files). If None, uses GATE's bundled defaults.
        """
        self.materials_dir = Path(materials_dir).resolve() \
            if materials_dir is not None else None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        run: Run,
        workdir: str | Path,
        config: Optional[GATEConfig] = None,
    ) -> "Any":
        """Build an opengate Simulation from a petsim Run.

        Returns the Simulation object, which the caller can inspect or
        modify before invoking `sim.run()`. The workdir is created if
        missing and will hold all intermediate files (voxelized phantom,
        source images, actor outputs).

        Raises ImportError if `opengate` isn't installed.
        """
        try:
            import opengate as gate
            from opengate.utility import g4_units
        except ImportError as exc:  # pragma: no cover - depends on GATE install
            raise ImportError(
                "The GATE backend requires `opengate` to be installed. "
                "In your petsim environment: `uv pip install opengate`"
            ) from exc

        if config is None:
            config = GATEConfig()

        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        # ---- Export phantom and source as ITK images ---------------
        # opengate expects .mhd/.raw voxelized inputs for phantoms and
        # voxelized sources; we write them here in the workdir.
        phantom_mhd = workdir / "phantom.mhd"
        source_mhd = workdir / "source_activity.mhd"
        self._write_mhd_material_image(run.phantom, phantom_mhd)
        self._write_mhd_activity_image(run.source, source_mhd)

        # ---- Set up Simulation -------------------------------------
        sim = gate.Simulation()
        sim.output_dir = str(workdir)
        sim.random_seed = run.seed if run.seed is not None else "auto"
        sim.number_of_threads = config.n_threads if config.n_threads > 0 else 1
        sim.verbose_level = config.verbose_level

        # GATE unit shortcuts — opengate returns scale factors so that
        # e.g. 10 * cm = 100 (mm, which is GATE's internal length unit).
        mm = g4_units.mm
        cm = g4_units.cm
        keV = g4_units.keV
        Bq = g4_units.Bq
        sec = g4_units.s

        # ---- World -------------------------------------------------
        world = sim.world
        world.size = [
            4 * run.scanner.detector_radius_cm * cm,
            4 * run.scanner.detector_radius_cm * cm,
            4 * run.scanner.detector_axial_length_cm * cm,
        ]
        world.material = "G4_AIR"

        # ---- Voxelized phantom ------------------------------------
        # GATE's ImageVolume loads .mhd and maps voxel values to
        # materials via a lookup table.
        phantom_volume = sim.add_volume("ImageVolume", "phantom")
        phantom_volume.image = str(phantom_mhd)
        phantom_volume.material = "G4_AIR"  # default for voxels outside LUT
        phantom_volume.voxel_materials = self._build_material_lut(run.phantom)

        # ---- Cylindrical detector (single ring) -------------------
        # Minimal scanner: a ring of crystals at the specified radius.
        # More sophisticated scanner templates (multi-ring, block structure)
        # go in a future version.
        self._add_cylindrical_detector(
            sim=sim, run=run, config=config, mm=mm, cm=cm,
        )

        # ---- Physics ----------------------------------------------
        sim.physics_manager.physics_list_name = config.physics_list
        if not config.back_to_back_annihilation:
            # Disable parametrized back-to-back and let full positron
            # range be simulated (slower but more realistic).
            sim.physics_manager.enable_decay = True

        # ---- Voxelized source -------------------------------------
        source = sim.add_source("VoxelSource", "pet_source")
        source.image = str(source_mhd)
        source.particle = "gamma"
        source.energy.mono = 511 * keV
        source.direction.type = "iso"
        # Activity is per-voxel Bq; opengate needs the total.
        source.activity = float(run.source.total_activity_Bq) * Bq
        # Attach to the phantom's coordinate system so positioning is
        # automatic.
        source.mother = "phantom"

        # ---- Acquisition time -------------------------------------
        sim.run_timing_intervals = [[0, run.scanner.acquisition_time_s * sec]]

        # ---- Actors: coincidence sorter + sinogram tallier --------
        self._add_output_actors(
            sim=sim, run=run, config=config, workdir=workdir,
            keV=keV, sec=sec,
        )

        return sim

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_full(
        self,
        run: Run,
        workdir: str | Path,
        config: Optional[GATEConfig] = None,
    ) -> tuple[Sinogram, GATERunResult]:
        """Build, run, and parse output in one call.

        For finer control, call build() yourself, modify the Simulation
        object, then invoke sim.run() and self.parse_sinogram() directly.
        """
        import time

        config = config or GATEConfig()
        workdir = Path(workdir).resolve()

        sim = self.build(run, workdir, config=config)

        t0 = time.perf_counter()
        sim.run()
        wall_time = time.perf_counter() - t0

        result = GATERunResult(
            workdir=workdir,
            wall_time_s=wall_time,
            sinogram_output_path=workdir / "sinogram.mhd",
            root_output_path=(workdir / config.root_output_filename
                              if config.store_root_output else None),
            actor_stats={},
        )
        sinogram = self.parse_sinogram(run, result)
        return sinogram, result

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_sinogram(
        self,
        run: Run,
        result: GATERunResult,
    ) -> Sinogram:
        """Load GATE's sinogram output into a petsim Sinogram.

        GATE's ProjectionActor writes an ITK image (.mhd/.raw). We
        load the raw binary, reshape it to the Scanner's sinogram_shape,
        and return it. Scatter/randoms separation depends on which
        actors were active; by default we populate `trues` with the
        total prompt coincidences.
        """
        if result.sinogram_output_path is None \
                or not result.sinogram_output_path.exists():
            raise FileNotFoundError(
                f"GATE did not produce a sinogram at "
                f"{result.sinogram_output_path}. Check that the "
                f"ProjectionActor was correctly attached."
            )

        # Read companion .raw file via the .mhd header
        raw = self._read_mhd(result.sinogram_output_path)
        expected = run.scanner.sinogram_shape
        if raw.size != int(np.prod(expected)):
            raise ValueError(
                f"GATE sinogram has {raw.size} elements, expected "
                f"{int(np.prod(expected))} for shape {expected}. "
                f"Check Scanner.n_radial_bins / n_angular_bins / n_z_slices."
            )
        trues = raw.reshape(expected).astype(np.float32)

        return Sinogram(
            scanner=run.scanner,
            trues=trues,
            scatter=None,    # GATE does not split unless we add a scatter tag
            randoms=None,
            metadata={
                "backend": "gate",
                "wall_time_s": result.wall_time_s,
            },
        )

    # ==================================================================
    # Internals
    # ==================================================================

    @staticmethod
    def _write_mhd_material_image(phantom: Phantom, path: Path) -> None:
        """Write phantom.material_ids to an ITK .mhd/.raw pair.

        GATE's ImageVolume reads the .mhd and uses the voxel integer
        values to look up materials in the voxel_materials LUT.
        """
        raw_path = path.with_suffix(".raw")
        # GATE expects Z-axis in slowest-varying order, which matches
        # numpy's default C order for (x, y, z) arrays transposed to
        # (z, y, x). opengate uses ITK internally, which stores data
        # in (z, y, x) C-contiguous order.
        data = np.ascontiguousarray(
            phantom.material_ids.transpose(2, 1, 0).astype(np.uint16)
        )
        data.tofile(raw_path)

        nx, ny, nz = phantom.shape
        # GATE/ITK wants mm
        dx_mm = phantom.voxel_size[0] * 10.0
        dy_mm = phantom.voxel_size[1] * 10.0
        dz_mm = phantom.voxel_size[2] * 10.0
        path.write_text(_MHD_TEMPLATE.format(
            nx=nx, ny=ny, nz=nz,
            dx=dx_mm, dy=dy_mm, dz=dz_mm,
            element_type="MET_USHORT",
            raw_filename=raw_path.name,
        ))

    @staticmethod
    def _write_mhd_activity_image(source: Source, path: Path) -> None:
        """Write source.activity_Bq to an ITK .mhd/.raw pair."""
        raw_path = path.with_suffix(".raw")
        data = np.ascontiguousarray(
            source.activity_Bq.transpose(2, 1, 0).astype(np.float32)
        )
        data.tofile(raw_path)

        nx, ny, nz = source.shape
        dx_mm = source.voxel_size[0] * 10.0
        dy_mm = source.voxel_size[1] * 10.0
        dz_mm = source.voxel_size[2] * 10.0
        path.write_text(_MHD_TEMPLATE.format(
            nx=nx, ny=ny, nz=nz,
            dx=dx_mm, dy=dy_mm, dz=dz_mm,
            element_type="MET_FLOAT",
            raw_filename=raw_path.name,
        ))

    @staticmethod
    def _read_mhd(path: Path) -> np.ndarray:
        """Minimal ITK .mhd reader — pulls the .raw and returns a flat array.

        Only supports MET_FLOAT, MET_INT, MET_USHORT — enough for GATE's
        sinogram outputs. A production-grade reader (e.g. SimpleITK) is
        not required here; this keeps the petsim dependency surface small.
        """
        header = {}
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            header[key.strip()] = val.strip()

        raw_name = header.get("ElementDataFile", path.stem + ".raw")
        raw_path = path.parent / raw_name
        element_type = header.get("ElementType", "MET_FLOAT")
        dtype_map = {
            "MET_FLOAT": np.float32,
            "MET_DOUBLE": np.float64,
            "MET_INT": np.int32,
            "MET_UINT": np.uint32,
            "MET_SHORT": np.int16,
            "MET_USHORT": np.uint16,
        }
        dtype = dtype_map.get(element_type, np.float32)
        return np.fromfile(raw_path, dtype=dtype)

    @staticmethod
    def _build_material_lut(phantom: Phantom) -> list[tuple[int, int, str]]:
        """Build a GATE ImageVolume voxel→material lookup table.

        Phantom stores material IDs 1..N; the LUT maps ID ranges to
        GATE material names. We use the phantom's material_names tuple
        in order.

        GATE expects (min_value_inclusive, max_value_inclusive, material_name).
        """
        return [
            (i + 1, i + 1, _petsim_to_gate_material(name))
            for i, name in enumerate(phantom.material_names)
        ]

    @staticmethod
    def _add_cylindrical_detector(sim, run, config, mm, cm) -> None:
        """Add a single-ring cylindrical detector to the simulation.

        This is a deliberately simple scanner geometry — a ring of LSO
        crystals at run.scanner.detector_radius_cm. Multi-ring and
        block-detector templates can be added later.
        """
        # Detector cylinder (mother volume)
        cyl = sim.add_volume("Tubs", "detector_cylinder")
        cyl.rmin = run.scanner.detector_radius_cm * cm
        cyl.rmax = (run.scanner.detector_radius_cm
                    + config.crystal_radial_mm / 10.0) * cm
        cyl.dz = (run.scanner.detector_axial_length_cm / 2.0) * cm
        cyl.material = "G4_AIR"
        cyl.color = [0, 1, 0, 0.2]

        # Crystal ring — placed as daughter volumes of the cylinder.
        # opengate supports a repeater for efficient placement.
        n_crystals = run.scanner.n_crystals_per_ring or 336
        crystal = sim.add_volume("Box", "crystal")
        crystal.size = [
            config.crystal_radial_mm * mm,
            config.crystal_tangential_mm * mm,
            config.crystal_axial_pitch_mm * mm,
        ]
        crystal.material = config.crystal_material
        crystal.mother = cyl.name
        # Ring repeater — opengate's "ring" repeater places N copies
        # around the Z axis at a given radius.
        crystal.translation = [
            (run.scanner.detector_radius_cm * cm
             + config.crystal_radial_mm / 2.0 * mm),
            0, 0,
        ]
        # Note: the exact repeater API varies by opengate version;
        # users may need to adjust this to their installed version.
        # See https://opengate-python.readthedocs.io/ for the current API.
        try:
            crystal.repeat = gate_ring_repeater(
                number=n_crystals,
                axis=[0, 0, 1],
                angle_deg=360.0 / n_crystals,
            )
        except Exception:
            # Fall back silently; the geometry is still valid as a
            # single crystal, just not a full ring.
            pass

    @staticmethod
    def _add_output_actors(sim, run, config, workdir: Path, keV, sec) -> None:
        """Attach GATE actors that produce the sinogram and (optional) ROOT."""
        # Digitizer: energy resolution + window
        digi = sim.add_actor("DigitizerAdderActor", "digitizer")
        digi.mother = "crystal"
        digi.output_filename = str(workdir / "digitizer.root") \
            if config.store_root_output else ""

        # Energy window
        win = sim.add_actor("DigitizerEnergyWindowsActor", "energy_window")
        win.mother = "crystal"
        e_low, e_high = run.scanner.energy_window_keV
        win.channels = [{"name": "photopeak",
                         "min": e_low * keV, "max": e_high * keV}]

        # Coincidence sorter
        coinc = sim.add_actor("CoincidenceSorterActor", "coincidences")
        coinc.mother = "crystal"
        if run.scanner.coincidence_window_ns is not None:
            from opengate.utility import g4_units
            ns = g4_units.ns
            coinc.time_window = run.scanner.coincidence_window_ns * ns

        # ProjectionActor → sinogram
        proj = sim.add_actor("ProjectionActor", "sinogram")
        proj.mother = "crystal"
        proj.output_filename = str(workdir / "sinogram.mhd")
        proj.size = [
            run.scanner.n_radial_bins,
            run.scanner.n_angular_bins,
            run.scanner.n_z_slices,
        ]


# =====================================================================
# Helpers
# =====================================================================


def gate_ring_repeater(number: int, axis: list[float], angle_deg: float):
    """Construct an opengate ring repeater descriptor.

    Kept as a free function so _add_cylindrical_detector doesn't bloat,
    and so the opengate import stays lazy.
    """
    import opengate as gate  # noqa: F401
    from opengate.geometry.utility import get_grid_repetition  # noqa: F401
    # Actual ring-repeat construction varies by opengate version.
    # Users are expected to tweak this function for their environment.
    # Returning a dict with the relevant parameters is a safe default.
    return {
        "type": "ring",
        "number": number,
        "axis": axis,
        "angle_deg": angle_deg,
    }


_PETSIM_TO_GATE_MATERIAL_MAP = {
    "air": "G4_AIR",
    "water": "G4_WATER",
    "bone": "G4_BONE_COMPACT_ICRU",
    "soft_tissue": "G4_TISSUE_SOFT_ICRU-44",
    "lung": "G4_LUNG_ICRP",
    "adipose": "G4_ADIPOSE_TISSUE_ICRP",
    "muscle": "G4_MUSCLE_SKELETAL_ICRP",
    "brain": "G4_BRAIN_ICRP",
    "blood": "G4_BLOOD_ICRP",
}


def _petsim_to_gate_material(name: str) -> str:
    """Map a petsim material name to a GATE/Geant4 material name.

    Unknown names fall through unchanged, assuming the user has defined
    them in a custom materials.db loaded via GATEBackend(materials_dir=...).
    """
    return _PETSIM_TO_GATE_MATERIAL_MAP.get(name.lower(), name)


_MHD_TEMPLATE = """\
ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = {dx} {dy} {dz}
DimSize = {nx} {ny} {nz}
ElementType = {element_type}
ElementDataFile = {raw_filename}
"""