"""
GATE backend: drive GATE 10 simulations via the `opengate` Python API.

Pass 1 (physics) — DONE
-----------------------
- Voxelized source via VoxelSource + activity MHD.
- Voxelized phantom via ImageVolume + material-ID MHD.
- Scatter labelling via a separate HitsCollectionActor on the phantom:
  any EventID with a `compt`/`Rayl` process in the phantom is flagged
  as scattered. Coincidences split: same-event-no-scatter = true,
  same-event-with-scatter = scatter, different-event = random.

Pass 2 (histogrammer) — DONE
----------------------------
4D ring-pair sinogram: shape (n_rings, n_rings, n_angular, n_radial).

For each coincidence:
  1. Recover (ring, crystal_in_ring) from the energy-weighted centroid
     position via xy-angle and z-position.
  2. Compute view angle and *signed* perpendicular distance from origin
     to the LOR connecting the two crystal centers.
  3. Bin into (ring1, ring2, view, radial).

Output axes are recorded in Sinogram.axes for downstream introspection.

Note: each LOR appears in both (r1=A, r2=B) and (r1=B, r2=A) because the
simulator labels which photon is "1" vs "2" arbitrarily. Counts split
~50/50 between the two index orderings. Symmetrize at read time if you
need the canonical ring1 <= ring2 form.

Pass 3 (auxiliary outputs) — DONE
---------------------------------
- mu_map.npy at 511 keV.
- voxel_size_mm in run.metadata.

Known limitations
-----------------
- Bruker geometry has axial gaps between modules. The z->ring mapping
  uses a uniform linear projection over [-L/2, +L/2], which approximates
  this. Counts may bleed slightly between adjacent rings near gaps.
- LYSO density in GateMaterials.db (5.37 g/cm^3) is below real LYSO
  (~7.1 g/cm^3); affects detection efficiency.
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
# Mass attenuation coefficients at 511 keV (cm^2/g)
# =====================================================================

MU_OVER_RHO_511_KEV: dict[str, float] = {
    "air":                0.0869,
    "water":              0.0958,
    "lung":               0.0958,
    "adipose":            0.0937,
    "muscle":             0.0961,
    "soft_tissue":        0.0954,
    "blood":              0.0959,
    "brain":              0.0959,
    "liver":              0.0961,
    "kidneys":            0.0961,
    "skin":               0.0961,
    "cartilage":          0.0972,
    "spongiosa":          0.0866,
    "stomach_intestines": 0.0954,
    "glands":             0.0954,
    "eyes":               0.0958,
    "breast_glandular":   0.0941,
    "lyso":               0.0871,
    "lso":                0.0871,
    "bgo":                0.0961,
}


# =====================================================================
# Config
# =====================================================================


@dataclass
class GATEConfig:
    """GATE-specific runtime parameters."""

    physics_list: str = "G4EmStandardPhysics_option3"
    production_cut_mm: float = 1.0
    back_to_back: bool = True
    keep_root_output: bool = False
    root_output_filename: str = "gate_output.root"
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
        self.materials_db = (
            Path(materials_db).resolve() if materials_db is not None else None
        )

        if materials_registry is not None:
            self.materials_registry = materials_registry
        elif self.materials_db is not None:
            self.materials_registry = MaterialRegistry(
                mcgpu_materials_dir=self.materials_db.parent,
            )
        else:
            self.materials_registry = MaterialRegistry(
                mcgpu_materials_dir=Path("."),
            )

    # ------------------------------------------------------------------
    # MHD writers
    # ------------------------------------------------------------------

    @staticmethod
    def _mhd_offset(phantom: Phantom) -> tuple[float, float, float]:
        dx, dy, dz = [v * 10.0 for v in phantom.voxel_size]  # cm -> mm
        nx, ny, nz = phantom.shape
        return (-(nx - 1) * dx / 2.0,
                -(ny - 1) * dy / 2.0,
                -(nz - 1) * dz / 2.0)

    @staticmethod
    def _write_mhd(
        path: Path,
        raw_filename: str,
        shape: tuple[int, int, int],
        spacing_mm: tuple[float, float, float],
        offset_mm: tuple[float, float, float],
        element_type: str,
    ) -> None:
        nx, ny, nz = shape
        dx, dy, dz = spacing_mm
        ox, oy, oz = offset_mm
        path.write_text(
            "ObjectType = Image\n"
            "NDims = 3\n"
            "BinaryData = True\n"
            "BinaryDataByteOrderMSB = False\n"
            "CompressedData = False\n"
            "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
            f"Offset = {ox:.6f} {oy:.6f} {oz:.6f}\n"
            "CenterOfRotation = 0 0 0\n"
            f"ElementSpacing = {dx:.6f} {dy:.6f} {dz:.6f}\n"
            f"DimSize = {nx} {ny} {nz}\n"
            f"ElementType = {element_type}\n"
            f"ElementDataFile = {raw_filename}\n"
        )

    @classmethod
    def _write_activity_image(cls, source: Source, phantom: Phantom, workdir: Path) -> Path:
        mhd_path = workdir / "source_activity.mhd"
        raw_path = workdir / "source_activity.raw"
        source.activity_Bq.astype(np.float32).tofile(str(raw_path))
        spacing = tuple(v * 10.0 for v in phantom.voxel_size)
        cls._write_mhd(
            mhd_path, raw_path.name, phantom.shape, spacing,
            cls._mhd_offset(phantom), "MET_FLOAT",
        )
        return mhd_path

    @classmethod
    def _write_material_image(cls, phantom: Phantom, workdir: Path) -> Path:
        mhd_path = workdir / "phantom_materials.mhd"
        raw_path = workdir / "phantom_materials.raw"
        phantom.material_ids.astype(np.uint16).tofile(str(raw_path))
        spacing = tuple(v * 10.0 for v in phantom.voxel_size)
        cls._write_mhd(
            mhd_path, raw_path.name, phantom.shape, spacing,
            cls._mhd_offset(phantom), "MET_USHORT",
        )
        return mhd_path

    @staticmethod
    def write_mu_map(phantom: Phantom, output_path: Path) -> np.ndarray:
        """Compute and save linear attenuation map at 511 keV (cm^-1)."""
        mu = np.zeros(phantom.shape, dtype=np.float32)
        for i, name in enumerate(phantom.material_names, start=1):
            mu_over_rho = MU_OVER_RHO_511_KEV.get(name.lower(), 0.0958)
            mask = phantom.material_ids == i
            mu[mask] = phantom.densities[mask] * mu_over_rho
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mu)
        return mu

    # ------------------------------------------------------------------
    # Crystal-center radius (used by histogrammer)
    # ------------------------------------------------------------------

    @staticmethod
    def _crystal_center_radius_mm(scanner) -> float:
        """Effective radius at which crystal centers sit, in mm.

        For the full Bruker geometry, _build_full_geometry places crystal
        centers at (detector_radius_cm * 10 + crystal_x / 2). The simple
        fallback geometry uses detector_radius_cm * 10 directly.
        """
        if scanner.crystal_size_mm is not None:
            return scanner.detector_radius_cm * 10.0 + scanner.crystal_size_mm[0] / 2.0
        return scanner.detector_radius_cm * 10.0

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
        except ImportError as exc:
            raise ImportError(
                "The GATE backend requires opengate. Install with: uv add opengate"
            ) from exc

        if config is None:
            config = GATEConfig()
        if run.binning is None:
            raise ValueError("Run.binning is None — GATE backend needs SinogramBinning.")

        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        mm = gate.g4_units.mm
        keV = gate.g4_units.keV
        Bq = gate.g4_units.Bq
        s = gate.g4_units.s

        sim = gate.Simulation()
        sim.output_dir = str(workdir)
        sim.random_seed = run.seed if run.seed is not None else "auto"
        sim.number_of_threads = config.n_threads
        sim.visu = False
        sim.check_volumes_overlap = False

        if self.materials_db is not None:
            db_dest = workdir / self.materials_db.name
            if not db_dest.exists():
                shutil.copy2(self.materials_db, db_dest)
            sim.volume_manager.add_material_database(str(db_dest))

        scanner = run.scanner
        world = sim.world
        world.size = [1000 * mm, 1000 * mm, 1000 * mm]
        world.material = "G4_AIR"

        # ---- Voxelized phantom ---------------------------------------
        phantom_mhd = self._write_material_image(run.phantom, workdir)
        phantom_vol = sim.add_volume("ImageVolume", "phantom")
        phantom_vol.image = str(phantom_mhd)

        voxel_materials = []
        for i, name in enumerate(run.phantom.material_names, start=1):
            try:
                gate_name = self.materials_registry.gate_name(name)
            except KeyError:
                gate_name = "G4_AIR"
            voxel_materials.append([i - 0.5, i + 0.5, gate_name])
        phantom_vol.voxel_materials = voxel_materials

        # ---- Scanner geometry ----------------------------------------
        if scanner.crystal_size_mm is not None:
            crystal = self._build_full_geometry(sim, scanner, mm)
        else:
            crystal = self._build_simple_geometry(sim, scanner, mm)

        sim.physics_manager.physics_list_name = config.physics_list
        sim.physics_manager.set_production_cut(
            "world", "all", config.production_cut_mm * mm
        )

        # ---- Voxelized source ----------------------------------------
        source_mhd = self._write_activity_image(run.source, run.phantom, workdir)
        pet_source = sim.add_source("VoxelSource", "pet_source")
        pet_source.particle = "back_to_back" if config.back_to_back else "e+"
        pet_source.activity = float(run.source.total_activity_Bq) * Bq
        pet_source.image = str(source_mhd)
        print(f"[build] source MHD offset would put hot voxel at (+10, 0, 0) mm")
        print(f"[build] source position translation: {pet_source.position.translation}")
        print(f"[build] source position type: {pet_source.position.type}")

        # ---- Digitizer chain on crystals -----------------------------
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

        # ---- Phantom hits actor (scatter labelling) ------------------
        ph = sim.add_actor("DigitizerHitsCollectionActor", "PhantomHits")
        ph.attached_to = phantom_vol.name
        ph.authorize_repeated_volumes = True
        ph.output_filename = output_file
        ph.attributes = [
            "EventID",
            "ProcessDefinedStep",
            "TotalEnergyDeposit",
        ]

        stats = sim.add_actor("SimulationStatisticsActor", "Stats")
        stats.output_filename = "stats.txt"

        sim.run_timing_intervals = [[0, scanner.acquisition_time_s * s]]
        return sim

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _build_full_geometry(self, sim, scanner, mm):
        from opengate.geometry.utility import (
            get_grid_repetition,
            get_circular_repetition,
        )

        crystal_x, crystal_y, crystal_z = scanner.crystal_size_mm
        cy, cz = scanner.crystals_per_module
        _, _, mz = scanner.modules_per_rsector
        n_rsectors = scanner.n_rsectors

        crystal_pitch_y = crystal_y * 1.008
        crystal_pitch_z = crystal_z * 1.008
        module_y = cy * crystal_pitch_y
        module_z = cz * crystal_pitch_z
        module_pitch_z = 32.0
        rsector_y = module_y + 0.5
        rsector_z = scanner.detector_axial_length_cm * 10.0
        rsector_x_mm = scanner.detector_radius_cm * 10.0 + crystal_x / 2.0

        sc_vol = sim.add_volume("Tubs", "scanner")
        sc_vol.rmax = (rsector_x_mm + crystal_x / 2.0 + 2) * mm
        sc_vol.rmin = (rsector_x_mm - crystal_x / 2.0 - 2) * mm
        sc_vol.dz = (rsector_z / 2.0) * mm
        sc_vol.material = "G4_AIR"

        rsector = sim.add_volume("Box", "rsector")
        rsector.mother = sc_vol.name
        rsector.size = [(crystal_x + 0.5) * mm, rsector_y * mm, rsector_z * mm]
        rsector.material = "G4_AIR"
        t, r = get_circular_repetition(
            n_rsectors, [rsector_x_mm * mm, 0, 0], axis=[0, 0, 1]
        )
        rsector.translation = t
        rsector.rotation = r

        module = sim.add_volume("Box", "module")
        module.mother = rsector.name
        module.size = [(crystal_x + 0.5) * mm, rsector_y * mm, (module_z + 0.5) * mm]
        module.material = "G4_AIR"
        module.translation = get_grid_repetition([1, 1, mz], [0, 0, module_pitch_z * mm])

        crystal = sim.add_volume("Box", "crystal")
        crystal.mother = module.name
        crystal.size = [crystal_x * mm, crystal_y * mm, crystal_z * mm]
        crystal.material = self.materials_registry.gate_name(
            scanner.crystal_material or "lyso"
        )
        crystal.translation = get_grid_repetition(
            [1, cy, cz], [0, crystal_pitch_y * mm, crystal_pitch_z * mm]
        )
        return crystal

    def _build_simple_geometry(self, sim, scanner, mm):
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
    # Parse sinogram
    # ------------------------------------------------------------------

    @staticmethod
    def _scattered_event_ids(phantom_hits: dict) -> set[int]:
        proc = phantom_hits["ProcessDefinedStep"]
        if proc.dtype.kind == "O" or proc.dtype.kind == "S":
            proc_strs = np.array(
                [p.decode() if isinstance(p, (bytes, bytearray)) else str(p)
                 for p in proc],
                dtype=object,
            )
        else:
            proc_strs = proc.astype(str)
        is_scatter = np.array(
            ["compt" in p.lower() or "rayl" in p.lower() for p in proc_strs]
        )
        return set(phantom_hits["EventID"][is_scatter].tolist())

    def parse_sinogram(self, run: Run, result: GATERunResult) -> Sinogram:
        """Read GATE ROOT output, label scatter, histogram into 4D ring-pair sinogram."""
        try:
            import uproot
        except ImportError as exc:
            raise ImportError(
                "Parsing GATE ROOT output requires uproot. uv add uproot"
            ) from exc

        if result.root_output_path is None or not result.root_output_path.exists():
            raise FileNotFoundError(
                f"GATE ROOT output not found at {result.root_output_path}."
            )

        binning = run.binning
        scanner = run.scanner

        with uproot.open(result.root_output_path) as f:
            tree_keys = {k.split(";")[0] for k in f.keys()}
            print(f"[parse] available trees: {sorted(tree_keys)}")

            if "Coincidences" not in tree_keys:
                raise KeyError("No 'Coincidences' tree in ROOT output.")
            coinc = f["Coincidences"].arrays(
                [
                    "EventID1", "EventID2",
                    "PostPosition1_X", "PostPosition1_Y", "PostPosition1_Z",
                    "PostPosition2_X", "PostPosition2_Y", "PostPosition2_Z",
                ],
                library="np",
            )

            phantom_hits = None
            if "PhantomHits" in tree_keys:
                phantom_hits = f["PhantomHits"].arrays(
                    ["EventID", "ProcessDefinedStep"], library="np"
                )

        n_total = len(coinc["EventID1"])
        print(f"[parse] coincidences: {n_total}")

        if phantom_hits is None or len(phantom_hits["EventID"]) == 0:
            print("[parse] no phantom hits — assuming all events unscattered")
            scattered_ids: set[int] = set()
        else:
            scattered_ids = self._scattered_event_ids(phantom_hits)
            print(f"[parse] phantom hits: {len(phantom_hits['EventID'])}")
            print(f"[parse] events with scatter: {len(scattered_ids)}")

        same_event = coinc["EventID1"] == coinc["EventID2"]

        if scattered_ids:
            scattered_arr = np.fromiter(scattered_ids, dtype=np.int64,
                                        count=len(scattered_ids))
            event_was_scattered = np.isin(coinc["EventID1"], scattered_arr)
        else:
            event_was_scattered = np.zeros(n_total, dtype=bool)

        true_mask = same_event & ~event_was_scattered
        scatter_mask = same_event & event_was_scattered

        print(f"[parse] same_event: {same_event.sum()}")
        print(f"[parse] true:       {true_mask.sum()}")
        print(f"[parse] scatter:    {scatter_mask.sum()}")
        print(f"[parse] random:     {(~same_event).sum()}")

        true_arrays = {k: v[true_mask] for k, v in coinc.items()}
        scatter_arrays = {k: v[scatter_mask] for k, v in coinc.items()}

        trues = self._histogram_coincidences(true_arrays, scanner, binning)
        scatter = self._histogram_coincidences(scatter_arrays, scanner, binning)

        metadata = {
            "backend": "gate",
            "wall_time_s": result.wall_time_s,
            "scanner": scanner.name,
            "scatter_labelled": True,
            "n_total_coincidences": int(n_total),
            "n_trues": int(true_mask.sum()),
            "n_scatter": int(scatter_mask.sum()),
            "n_randoms": int((~same_event).sum()),
        }

        return Sinogram(
            trues=trues,
            scatter=scatter,
            shape=trues.shape,
            axes=("ring1", "ring2", "angular", "radial"),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Pass 2: LOR-based 4D histogrammer
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pass 2: LOR-based 4D histogrammer (FIXED)
    # ------------------------------------------------------------------

    def _histogram_coincidences(
        self, arrays: dict, scanner, binning,
    ) -> np.ndarray:
        """Bin coincidences into a 4D ring-pair sinogram.

        Output shape: (n_rings, n_rings, n_angular, n_radial).

        Uses the actual hit positions to compute LOR geometry, NOT
        reconstructed crystal centers. Snapping to crystal centers
        collapses opposite crystals to (c1, c1+N/2) for most events,
        which makes every LOR pass through the origin and destroys the
        radial offset signal — that bug took longer to find than I'd
        like to admit.

        Crystal indices (ring, crystal_in_ring) are still recovered for
        bookkeeping, but only the ring index is used (for the (r1, r2)
        bin); the in-ring index doesn't enter the LOR geometry.

        Steps:
          1. Recover ring index for each end from z position.
          2. Compute LOR midpoint and perpendicular from raw (x, y) hits.
          3. View angle phi = atan2(perp_y, perp_x) wrapped to [0, π).
             Signed radial s = midpoint · perp_normalized.
          4. Bin into (r1, r2, phi_bin, s_bin).
        """
        n_rings = scanner.n_rings
        R_eff = self._crystal_center_radius_mm(scanner)
        L_mm = scanner.detector_axial_length_cm * 10.0
        n_ang = binning.n_angular_bins
        n_rad = binning.n_radial_bins

        sino = np.zeros((n_rings, n_rings, n_ang, n_rad), dtype=np.float32)

        x1 = arrays["PostPosition1_X"]
        if len(x1) == 0:
            return sino

        y1 = arrays["PostPosition1_Y"]
        z1 = arrays["PostPosition1_Z"]
        x2 = arrays["PostPosition2_X"]
        y2 = arrays["PostPosition2_Y"]
        z2 = arrays["PostPosition2_Z"]

        # ---- Ring index from z, with rounding ------------------------
        # Use round() not int() to put z=0 events into ring n_rings/2,
        # not n_rings/2 - 1. The previous integer-truncation off-by-one
        # caused the (12, 12) direct plane to be empty — events landed
        # in (11, 12) and (12, 11) instead.
        r1 = np.round((z1 + L_mm / 2.0) / L_mm * (n_rings - 1)).astype(np.int32)
        r2 = np.round((z2 + L_mm / 2.0) / L_mm * (n_rings - 1)).astype(np.int32)
        r1 = np.clip(r1, 0, n_rings - 1)
        r2 = np.clip(r2, 0, n_rings - 1)

        # ---- LOR geometry from raw xy hits ---------------------------
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        dx = x2 - x1
        dy = y2 - y1
        d_mag = np.sqrt(dx * dx + dy * dy)

        valid = d_mag > 1e-6
        if not valid.all():
            kept = int(valid.sum())
            print(f"[hist] skipping {len(valid) - kept} degenerate LORs")

        # Avoid division warnings on the invalid rows
        d_mag_safe = np.where(valid, d_mag, 1.0)
        perp_x = -dy / d_mag_safe
        perp_y =  dx / d_mag_safe

        # View angle of the perpendicular, wrapped to [0, π)
        phi = np.arctan2(perp_y, perp_x)
        # Sign of perpendicular flips the radial sign; canonicalize by
        # forcing perp to point in the upper half-plane (phi in [0, π))
        flip = phi < 0
        perp_x = np.where(flip, -perp_x, perp_x)
        perp_y = np.where(flip, -perp_y, perp_y)
        phi = np.where(flip, phi + np.pi, phi)
        phi = phi % np.pi

        # Signed perpendicular distance from origin to LOR
        s = mx * perp_x + my * perp_y  # in [-R_eff, +R_eff]

        # Bin
        ang_idx = np.clip((phi / np.pi * n_ang).astype(np.int32), 0, n_ang - 1)
        rad_idx = np.clip(
            ((s / R_eff + 1.0) / 2.0 * n_rad).astype(np.int32),
            0, n_rad - 1,
        )

        if not valid.all():
            r1 = r1[valid]
            r2 = r2[valid]
            ang_idx = ang_idx[valid]
            rad_idx = rad_idx[valid]

        np.add.at(sino, (r1, r2, ang_idx, rad_idx), 1)
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
        keep_workdir: bool = True,
    ) -> tuple[Sinogram, GATERunResult]:
        """Build, run, parse, save."""
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

        root_path = workdir / config.root_output_filename
        if not root_path.exists():
            candidates = list(workdir.rglob("*.root"))
            root_path = candidates[0] if candidates else None

        if root_path is None:
            raise FileNotFoundError(
                f"GATE did not produce a ROOT file in {workdir}.\n"
                f"Workdir contents: "
                f"{list(workdir.rglob('*')) if workdir.exists() else []}.\n"
                "Check that the CoincidenceSorterActor ran successfully."
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
        run.metadata["voxel_size_mm"] = [v * 10.0 for v in run.phantom.voxel_size]
        run.save(run_dir)

        mu_path = run_dir / "mu_map.npy"
        self.write_mu_map(run.phantom, mu_path)
        print(f"[run_full] saved mu-map to {mu_path}")

        if not keep_workdir and workdir.exists():
            shutil.rmtree(workdir)

        return sinogram, result