"""
GATE backend: drive GATE 10 simulations via the `opengate` Python API.

Design notes
------------
Unlike MCGPU-PET, GATE doesn't need on-disk input files — the simulation
is built in memory via opengate's Simulation object. The pipeline is:

  1. build(run, workdir, config) → Simulation object (inspect/modify before running)
  2. run_full(run, run_dir, config) → (Sinogram, GATERunResult): the all-in-one path

Geometry
--------
The backend builds the full Bruker-style cylindricalPET geometry from
Scanner fields (n_rings, n_crystals_per_ring, crystal_size_mm, etc.).
For scanners without crystal-level detail, it falls back to a simplified
single-ring cylinder.

Materials
---------
Material name resolution uses the same MaterialRegistry as MCGPU-PET.
Each Material carries both mcgpu_file (for MCGPU) and gate_name (for GATE),
so `registry.gate_name("water")` returns `"G4_WATER"` and
`registry.gate_name("lyso")` returns `"LYSO"` (from GateMaterials.db).

Pass materials_db to GATEBackend to enable custom materials like LYSO.
Pass a pre-built MaterialRegistry directly if you want to share one
instance across both backends.

Sinogram output
---------------
GATE's CoincidenceSorterActor writes list-mode coincidences to a ROOT
file. We read that with uproot and histogram each event into a sinogram
using the Michelogram convention defined in SinogramBinning.

NOTE: The Michelogram histogrammer (_histogram_coincidences) is currently
a placeholder — direct segment only. See TODO in README.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..materials import MaterialRegistry
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

    These are things GATE cares about that MCGPU doesn't have an
    equivalent for. Binning (span, MRD, etc.) stays in Run.binning.
    """

    # ---- Physics -----------------------------------------------------
    physics_list: str = "G4EmStandardPhysics_option3"
    production_cut_mm: float = 1.0

    # ---- Source model ------------------------------------------------
    # back_to_back=True: emit two collinear 511 keV gammas directly.
    # Faster and matches MCGPU's model (no positron range).
    back_to_back: bool = True

    # ---- Output ------------------------------------------------------
    keep_root_output: bool = False
    root_output_filename: str = "gate_output.root"

    # ---- Runtime -----------------------------------------------------
    n_threads: int = 1
    verbose_level: int = 0


# =====================================================================
# Result
# =====================================================================


@dataclass
class GATERunResult:
    """What a GATE invocation produced."""

    workdir: Path
    wall_time_s: float
    root_output_path: Optional[Path] = None
    actor_stats: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Backend
# =====================================================================


class GATEBackend:
    """Driver for GATE 10 simulations via the opengate Python API."""

    def __init__(
        self,
        materials_db: Optional[str | Path] = None,
        materials_registry: Optional[MaterialRegistry] = None,
    ) -> None:
        """
        Parameters
        ----------
        materials_db : path to a GateMaterials.db file for custom materials
            (e.g. LYSO). If None, only NIST G4_* materials are available.
            A MaterialRegistry is built automatically from this path using
            the default material table in materials.py.
        materials_registry : pre-built MaterialRegistry. Pass this if you
            want to share one instance across both backends, or if you need
            custom materials beyond the defaults. Takes priority over
            materials_db if both are given.
        """
        self.materials_db = Path(materials_db).resolve() \
            if materials_db is not None else None

        if materials_registry is not None:
            self.materials_registry = materials_registry
        elif self.materials_db is not None:
            # Build a registry pointing at the db's parent directory.
            # GATE only needs gate_name() from the registry, so
            # mcgpu_materials_dir can point anywhere — it's unused here.
            self.materials_registry = MaterialRegistry(
                mcgpu_materials_dir=self.materials_db.parent,
            )
        else:
            self.materials_registry = MaterialRegistry(
                mcgpu_materials_dir=Path("."),
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        run: Run,
        workdir: str | Path,
        config: Optional[GATEConfig] = None,
    ) -> Any:
        """Build an opengate Simulation from a petsim Run."""
        try:
            import opengate as gate
            from opengate.geometry.utility import (
                get_grid_repetition,
                get_circular_repetition,
            )
        except ImportError as exc:
            raise ImportError(
                "The GATE backend requires opengate. Install with: uv add opengate"
            ) from exc

        if config is None:
            config = GATEConfig()

        if run.binning is None:
            raise ValueError(
                "Run.binning is None — GATE backend needs SinogramBinning. "
                "Use SinogramBinning.default_for(scanner) or a preset."
            )

        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        mm = gate.g4_units.mm
        keV = gate.g4_units.keV
        Bq = gate.g4_units.Bq
        s = gate.g4_units.s

        # ---- Simulation object ---------------------------------------
        sim = gate.Simulation()
        sim.output_dir = str(workdir)
        sim.random_seed = run.seed if run.seed is not None else "auto"
        sim.number_of_threads = config.n_threads
        sim.visu = False
        sim.check_volumes_overlap = False

        # ---- Materials -----------------------------------------------
        # Note: GateMaterials.db may emit warnings about zero-fraction
        # elements (e.g. "fraction 'f=' is 0"). These are benign — GATE
        # simply ignores zero-fraction components. They come from the
        # handover .db file and can be silenced by cleaning up those
        # entries, but they don't affect simulation correctness.
        if self.materials_db is not None:
            db_dest = workdir / self.materials_db.name
            if not db_dest.exists():
                shutil.copy2(self.materials_db, db_dest)
            sim.volume_manager.add_material_database(str(db_dest))

        # ---- World ---------------------------------------------------
        scanner = run.scanner
        world = sim.world
        world.size = [1000 * mm, 1000 * mm, 1000 * mm]  # 1m cube
        world.material = "G4_AIR"

        # ---- Scanner geometry ----------------------------------------
        if scanner.crystal_size_mm is not None:
            crystal = self._build_full_geometry(sim, scanner, mm)
        else:
            crystal = self._build_simple_geometry(sim, scanner, mm)

        # ---- Physics -------------------------------------------------
        sim.physics_manager.physics_list_name = config.physics_list
        sim.physics_manager.set_production_cut(
            "world", "all", config.production_cut_mm * mm
        )

        # ---- Source --------------------------------------------------
        source = sim.add_source("GenericSource", "pet_source")
        source.particle = "back_to_back" if config.back_to_back else "e+"
        source.activity = float(run.source.total_activity_Bq) * Bq
        source.position.type = "point"
        source.position.translation = [0, 0, 0]
        source.direction.type = "iso"
        # TODO: replace with voxelized source from run.source

        # ---- Digitizer chain -----------------------------------------
        output_file = config.root_output_filename

        hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
        hc.attached_to = crystal.name
        hc.authorize_repeated_volumes = True
        hc.output_filename = output_file
        hc.attributes = [
            "EventID",
            "PostPosition",
            "TotalEnergyDeposit",
            "PreStepUniqueVolumeID",
            "GlobalTime",
        ]

        sc = sim.add_actor("DigitizerAdderActor", "Singles")
        sc.attached_to = hc.attached_to
        sc.authorize_repeated_volumes = True
        sc.input_digi_collection = hc.name
        sc.policy = "EnergyWeightedCentroidPosition"
        sc.group_volume = crystal.name
        sc.output_filename = output_file

        ew = sim.add_actor("DigitizerEnergyWindowsActor", "EnergyWindow")
        ew.attached_to = hc.attached_to
        ew.authorize_repeated_volumes = True
        ew.input_digi_collection = sc.name
        ew.output_filename = output_file
        e_low, e_high = scanner.energy_window_keV
        ew.channels = [{"name": "peak511",
                        "min": e_low * keV, "max": e_high * keV}]

        cc = sim.add_actor("CoincidenceSorterActor", "Coincidences")
        cc.input_digi_collection = "peak511"
        cc.window = (scanner.coincidence_window_ns or 10.0) * 1e-9 * s
        cc.output_filename = output_file

        stats = sim.add_actor("SimulationStatisticsActor", "Stats")
        stats.output_filename = "stats.txt"

        sim.run_timing_intervals = [[0, scanner.acquisition_time_s * s]]

        return sim

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _build_full_geometry(self, sim, scanner, mm):
        """Full rsector → module → crystal hierarchy from Scanner fields.

        Mirrors bruker_pet_sim.py exactly. Key numbers for bruker_albira:
          - rsector at x=67mm, size 10.5 x 50.5 x 95mm
          - module size 10.5 x 50.5 x 50.5mm, 3 axially at 32mm pitch
          - crystal size 10 x 6mm x 6mm (6.3mm pitch), 8x8 per module
        """
        from opengate.geometry.utility import (
            get_grid_repetition,
            get_circular_repetition,
        )

        crystal_x, crystal_y, crystal_z = scanner.crystal_size_mm  # (10, 6.25, 6.25)
        cy, cz = scanner.crystals_per_module        # (8, 8) tang x axial
        _, _, mz = scanner.modules_per_rsector      # (1, 1, 3) axial modules
        n_rsectors = scanner.n_rsectors             # 8

        # Use slightly larger pitch than crystal size to avoid overlaps
        # (matches reference: 50.5mm module, 6.3mm crystal pitch)
        crystal_pitch_y = crystal_y * 1.008   # ~6.3mm for 6.25mm crystals
        crystal_pitch_z = crystal_z * 1.008
        module_y = cy * crystal_pitch_y         # ~50.4mm
        module_z = cz * crystal_pitch_z         # ~50.4mm
        module_pitch_z = 32.0                   # mm, matches reference exactly
        rsector_y = module_y + 0.5              # slight margin
        rsector_z = scanner.detector_axial_length_cm * 10.0  # 105mm

        # Rsector radial placement (matches reference: 67mm from axis)
        rsector_x_mm = scanner.detector_radius_cm * 10.0 + crystal_x / 2.0

        # Scanner envelope
        sc_vol = sim.add_volume("Tubs", "scanner")
        sc_vol.rmax = (rsector_x_mm + crystal_x / 2.0 + 2) * mm
        sc_vol.rmin = (rsector_x_mm - crystal_x / 2.0 - 2) * mm
        sc_vol.dz = (rsector_z / 2.0) * mm
        sc_vol.material = "G4_AIR"

        # rsector: 8 around the ring
        rsector = sim.add_volume("Box", "rsector")
        rsector.mother = sc_vol.name
        rsector.size = [(crystal_x + 0.5) * mm, rsector_y * mm, rsector_z * mm]
        rsector.material = "G4_AIR"
        t, r = get_circular_repetition(
            n_rsectors,
            [rsector_x_mm * mm, 0, 0],
            axis=[0, 0, 1],
        )
        rsector.translation = t
        rsector.rotation = r

        # module: mz axially per rsector at module_pitch_z spacing
        module = sim.add_volume("Box", "module")
        module.mother = rsector.name
        module.size = [(crystal_x + 0.5) * mm, rsector_y * mm, (module_z + 0.5) * mm]
        module.material = "G4_AIR"
        module.translation = get_grid_repetition(
            [1, 1, mz], [0, 0, module_pitch_z * mm]
        )

        # crystal: cy x cz per module
        crystal = sim.add_volume("Box", "crystal")
        crystal.mother = module.name
        crystal.size = [crystal_x * mm, crystal_y * mm, crystal_z * mm]

        # Resolve crystal material via MaterialRegistry
        raw_name = scanner.crystal_material or "lyso"
        crystal.material = self.materials_registry.gate_name(raw_name)

        crystal.translation = get_grid_repetition(
            [1, cy, cz], [0, crystal_pitch_y * mm, crystal_pitch_z * mm]
        )

        return crystal

    def _build_simple_geometry(self, sim, scanner, mm):
        """Fallback: a single flat ring of crystals."""
        from opengate.geometry.utility import get_circular_repetition

        n = scanner.n_crystals_per_ring
        r_mm = scanner.detector_radius_cm * 10.0
        crystal_tang_mm = 3.14159 * 2 * r_mm / n * 0.9

        cyl = sim.add_volume("Tubs", "scanner")
        cyl.rmax = (r_mm + 12) * mm
        cyl.rmin = (r_mm - 2) * mm
        cyl.dz = (scanner.detector_axial_length_cm * 10 / 2) * mm
        cyl.material = "G4_AIR"

        crystal = sim.add_volume("Box", "crystal")
        crystal.mother = cyl.name
        crystal.size = [10 * mm, crystal_tang_mm * mm, 10 * mm]
        crystal.material = "G4_AIR"
        t, r = get_circular_repetition(n, [r_mm * mm, 0, 0], axis=[0, 0, 1])
        crystal.translation = t
        crystal.rotation = r

        return crystal

    # ------------------------------------------------------------------
    # Parse sinogram from ROOT output
    # ------------------------------------------------------------------

    def parse_sinogram(self, run: Run, result: GATERunResult) -> Sinogram:
        """Read GATE's ROOT output and histogram into a Sinogram."""
        try:
            import uproot
        except ImportError as exc:
            raise ImportError(
                "Parsing GATE ROOT output requires uproot. "
                "Install with: uv add uproot"
            ) from exc

        if result.root_output_path is None or not result.root_output_path.exists():
            raise FileNotFoundError(
                f"GATE ROOT output not found at {result.root_output_path}."
            )

        binning = run.binning
        scanner = run.scanner

        with uproot.open(result.root_output_path) as f:
            tree = f["Coincidences"]
            arrays = tree.arrays(
                ["PostPosition1_X", "PostPosition1_Y", "PostPosition1_Z",
                 "PostPosition2_X", "PostPosition2_Y", "PostPosition2_Z"],
                library="np",
            )

        trues = self._histogram_coincidences(arrays, scanner, binning)

        return Sinogram(
            trues=trues,
            scatter=None,   # TODO: label scatter via EventID matching
            shape=trues.shape,
            metadata={
                "backend": "gate",
                "wall_time_s": result.wall_time_s,
                "scanner": scanner.name,
                "note": "Michelogram histogramming is placeholder — direct only",
            },
        )

    @staticmethod
    def _histogram_coincidences(arrays, scanner, binning) -> np.ndarray:
        """Bin list-mode coincidences into a sinogram.

        PLACEHOLDER: direct segment only (z-midpoint binning).
        Full Michelogram matching MCGPU's layout is deferred — see README TODO.
        """
        n_z = 2 * scanner.n_rings - 1
        n_ang = binning.n_angular_bins
        n_rad = binning.n_radial_bins

        sino = np.zeros((n_z, n_ang, n_rad), dtype=np.float32)

        x1, y1, z1 = arrays["PostPosition1_X"], arrays["PostPosition1_Y"], arrays["PostPosition1_Z"]
        x2, y2, z2 = arrays["PostPosition2_X"], arrays["PostPosition2_Y"], arrays["PostPosition2_Z"]

        phi = np.arctan2(y1 + y2, x1 + x2) % np.pi
        ang_idx = np.clip((phi / np.pi * n_ang).astype(np.int32), 0, n_ang - 1)

        r = np.sqrt(((x1 + x2) / 2) ** 2 + ((y1 + y2) / 2) ** 2)
        r_max = scanner.detector_radius_cm * 10.0
        rad_idx = np.clip(
            ((r / r_max + 1) / 2 * n_rad).astype(np.int32), 0, n_rad - 1
        )

        z_fov_half = scanner.detector_axial_length_cm * 10.0 / 2
        z_mid = (z1 + z2) / 2
        z_idx = np.clip(
            ((z_mid + z_fov_half) / (2 * z_fov_half) * n_z).astype(np.int32),
            0, n_z - 1,
        )

        np.add.at(sino, (z_idx, ang_idx, rad_idx), 1)
        return sino

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full(
        self,
        run: Run,
        run_dir: str | Path,
        config: Optional[GATEConfig] = None,
        workdir: Optional[str | Path] = None,
        keep_workdir: bool = False,
    ) -> tuple[Sinogram, GATERunResult]:
        """Build, run, histogram, save — the complete pipeline."""
        if config is None:
            config = GATEConfig()

        run_dir = Path(run_dir).resolve()
        if workdir is None:
            workdir = run_dir / "tmp"
        workdir = Path(workdir).resolve()

        sim = self.build(run, workdir, config=config)

        t0 = time.perf_counter()
        sim.run(start_new_process=True)
        wall_time = time.perf_counter() - t0

        # Locate the ROOT file. opengate prepends output_dir to the actor
        # filename, so the file should be at workdir/gate_output.root.
        # We glob as a fallback in case opengate creates a subdirectory
        # or changes the naming convention across versions.
        root_path = workdir / config.root_output_filename
        if not root_path.exists():
            candidates = list(workdir.rglob("*.root"))
            root_path = candidates[0] if candidates else None

        if root_path is None:
            raise FileNotFoundError(
                f"GATE did not produce a ROOT file in {workdir}.\nWorkdir contents: {list(workdir.rglob(chr(42))) if workdir.exists() else []}.\n"
                f"Check that the CoincidenceSorterActor ran successfully. "
                
            )

        result = GATERunResult(
            workdir=workdir,
            wall_time_s=wall_time,
            root_output_path=root_path,
        )

        sinogram = self.parse_sinogram(run, result)

        run.sinogram = sinogram
        run.metadata["backend"] = "gate"
        run.metadata["gate_config"] = asdict(config)
        run.metadata["wall_time_s"] = wall_time
        run.save(run_dir)

        if not keep_workdir and workdir.exists():
            shutil.rmtree(workdir)

        return sinogram, result