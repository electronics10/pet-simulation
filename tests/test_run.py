"""
Unit tests for petsim.run.

This test module also covers the final integration validation criterion
for Phase 1: a full scene (phantom + source + scanner + sinogram)
round-trips through save/load with byte-exact match for arrays and value
equality for metadata.

Run with:
    uv run pytest tests/test_run.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from petsim.phantom import Phantom
from petsim.run import (
    MANIFEST_FILENAME,
    PHANTOM_FILENAME,
    SCANNER_FILENAME,
    SINOGRAM_FILENAME,
    SOURCE_FILENAME,
    Run,
)
from petsim.scanner import Scanner
from petsim.sinogram import Sinogram
from petsim.source import Source


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def phantom():
    """Reference 9x9x9 water-in-air phantom matching the MCGPU-PET sample."""
    return Phantom.cube(
        shape=(9, 9, 9),
        voxel_size=(1.0, 1.0, 1.0),
        inner_material="water",
        inner_density=1.0,
        outer_material="air",
        outer_density=0.0012,
        inner_size_vox=5,
    )


@pytest.fixture
def source(phantom):
    return Source.with_total_activity(
        phantom, material="water", total_activity_Bq=1000.0, isotope="F18",
    )


@pytest.fixture
def scanner():
    """Small scanner so test arrays stay tiny."""
    return Scanner(
        name="toy",
        detector_radius_cm=10.0,
        detector_axial_length_cm=10.0,
        energy_window_keV=(350.0, 650.0),
        energy_resolution=0.12,
        acquisition_time_s=1.0,
        n_radial_bins=8,
        n_angular_bins=6,
        n_z_slices=4,
        span=1,
        max_ring_difference=2,
    )


@pytest.fixture
def sinogram(scanner):
    rng = np.random.default_rng(42)
    trues = rng.integers(0, 5, size=scanner.sinogram_shape, dtype=np.int32)
    scatter = rng.integers(0, 2, size=scanner.sinogram_shape, dtype=np.int32)
    return Sinogram(
        scanner=scanner,
        trues=trues,
        scatter=scatter,
        metadata={"backend": "mcgpu", "simulated_histories": 23464},
    )


# =====================================================================
# Instantiation
# =====================================================================


class TestInstantiation:
    def test_construct_without_sinogram(self, phantom, source, scanner):
        """A Run can be instantiated before the simulation is run."""
        r = Run(phantom=phantom, source=source, scanner=scanner)
        assert r.sinogram is None

    def test_construct_with_all_parts(self, phantom, source, scanner, sinogram):
        r = Run(
            phantom=phantom, source=source, scanner=scanner, sinogram=sinogram,
        )
        assert r.sinogram is sinogram

    def test_mismatched_source_grid_rejected(self, phantom, scanner):
        """Source must share the phantom grid."""
        wrong_phantom = Phantom.uniform(
            shape=(4, 4, 4), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        mismatched_source = Source.zeros(wrong_phantom)
        with pytest.raises(ValueError, match="source grid"):
            Run(phantom=phantom, source=mismatched_source, scanner=scanner)

    def test_mismatched_sinogram_scanner_rejected(
        self, phantom, source, scanner, sinogram
    ):
        """Sinogram's scanner must match the run's scanner."""
        other_scanner = Scanner(
            name="other",  # different name
            detector_radius_cm=10.0,
            detector_axial_length_cm=10.0,
            energy_window_keV=(350.0, 650.0),
            energy_resolution=0.12,
            acquisition_time_s=1.0,
            n_radial_bins=8,
            n_angular_bins=6,
            n_z_slices=4,
            span=1,
            max_ring_difference=2,
        )
        with pytest.raises(ValueError, match="sinogram.scanner"):
            Run(
                phantom=phantom, source=source, scanner=other_scanner,
                sinogram=sinogram,
            )


# =====================================================================
# Persistence — the big Phase 1 integration test
# =====================================================================


class TestPersistence:
    def test_round_trip_with_sinogram(
        self, phantom, source, scanner, sinogram, tmp_path: Path
    ):
        """Phase 1 final validation criterion: a complete scene round-trips
        through save/load with arrays and metadata preserved exactly.
        """
        original = Run(
            phantom=phantom,
            source=source,
            scanner=scanner,
            sinogram=sinogram,
            seed=266280817,
            metadata={
                "backend": "mcgpu",
                "wall_time_seconds": 4.8,
                "git_hash": "fde2470",
            },
        )

        run_dir = tmp_path / "runs" / "001_water_cube"
        original.save(run_dir)
        loaded = Run.load(run_dir)

        assert loaded == original
        # Double-check arrays are byte-exact
        assert np.array_equal(loaded.phantom.material_ids, phantom.material_ids)
        assert np.array_equal(loaded.phantom.densities, phantom.densities)
        assert np.array_equal(loaded.source.activity_Bq, source.activity_Bq)
        assert np.array_equal(loaded.sinogram.trues, sinogram.trues)
        assert np.array_equal(loaded.sinogram.scatter, sinogram.scatter)
        # Seed is a first-class field
        assert loaded.seed == 266280817
        # Metadata values are preserved
        assert loaded.metadata["backend"] == "mcgpu"

    def test_round_trip_without_sinogram(
        self, phantom, source, scanner, tmp_path: Path
    ):
        """Prepared-but-unrun bundles must round-trip too."""
        original = Run(phantom=phantom, source=source, scanner=scanner)
        run_dir = tmp_path / "unrun"
        original.save(run_dir)
        loaded = Run.load(run_dir)
        assert loaded == original
        assert loaded.sinogram is None

    def test_save_creates_expected_files(
        self, phantom, source, scanner, sinogram, tmp_path: Path
    ):
        r = Run(
            phantom=phantom, source=source, scanner=scanner, sinogram=sinogram,
        )
        run_dir = tmp_path / "run"
        r.save(run_dir)
        assert (run_dir / PHANTOM_FILENAME).exists()
        assert (run_dir / SOURCE_FILENAME).exists()
        assert (run_dir / SCANNER_FILENAME).exists()
        assert (run_dir / SINOGRAM_FILENAME).exists()
        assert (run_dir / MANIFEST_FILENAME).exists()

    def test_save_without_sinogram_skips_sinogram_file(
        self, phantom, source, scanner, tmp_path: Path
    ):
        r = Run(phantom=phantom, source=source, scanner=scanner)
        run_dir = tmp_path / "run"
        r.save(run_dir)
        assert not (run_dir / SINOGRAM_FILENAME).exists()

    def test_manifest_records_has_sinogram(
        self, phantom, source, scanner, sinogram, tmp_path: Path
    ):
        r_with = Run(
            phantom=phantom, source=source, scanner=scanner, sinogram=sinogram,
        )
        r_without = Run(phantom=phantom, source=source, scanner=scanner)

        dir_with = tmp_path / "with"
        dir_without = tmp_path / "without"
        r_with.save(dir_with)
        r_without.save(dir_without)

        with open(dir_with / MANIFEST_FILENAME) as f:
            manifest_with = yaml.safe_load(f)
        with open(dir_without / MANIFEST_FILENAME) as f:
            manifest_without = yaml.safe_load(f)

        assert manifest_with["has_sinogram"] is True
        assert manifest_without["has_sinogram"] is False

    def test_manifest_populates_created_at(
        self, phantom, source, scanner, tmp_path: Path
    ):
        r = Run(phantom=phantom, source=source, scanner=scanner)
        run_dir = tmp_path / "run"
        r.save(run_dir)
        with open(run_dir / MANIFEST_FILENAME) as f:
            manifest = yaml.safe_load(f)
        assert "created_at" in manifest

    def test_manifest_preserves_user_metadata(
        self, phantom, source, scanner, tmp_path: Path
    ):
        r = Run(
            phantom=phantom, source=source, scanner=scanner,
            metadata={"backend": "gate", "my_custom_field": "hello"},
        )
        run_dir = tmp_path / "run"
        r.save(run_dir)
        with open(run_dir / MANIFEST_FILENAME) as f:
            manifest = yaml.safe_load(f)
        assert manifest["backend"] == "gate"
        assert manifest["my_custom_field"] == "hello"

    def test_load_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
            Run.load(tmp_path / "does_not_exist")

    def test_load_fails_when_sinogram_promised_but_missing(
        self, phantom, source, scanner, sinogram, tmp_path: Path
    ):
        """If the manifest says has_sinogram=true but the file is gone,
        load should fail loudly instead of silently returning sinogram=None.
        """
        r = Run(
            phantom=phantom, source=source, scanner=scanner, sinogram=sinogram,
        )
        run_dir = tmp_path / "run"
        r.save(run_dir)
        (run_dir / SINOGRAM_FILENAME).unlink()
        with pytest.raises(FileNotFoundError, match="claims sinogram present"):
            Run.load(run_dir)

    def test_save_creates_parent_dirs(
        self, phantom, source, scanner, tmp_path: Path
    ):
        r = Run(phantom=phantom, source=source, scanner=scanner)
        nested = tmp_path / "a" / "b" / "c" / "run"
        r.save(nested)
        assert (nested / MANIFEST_FILENAME).exists()

    def test_seed_round_trips(
        self, phantom, source, scanner, tmp_path: Path
    ):
        """Seed is a first-class field and must survive save/load."""
        r = Run(phantom=phantom, source=source, scanner=scanner, seed=42)
        run_dir = tmp_path / "run"
        r.save(run_dir)
        loaded = Run.load(run_dir)
        assert loaded.seed == 42

    def test_seed_none_round_trips(
        self, phantom, source, scanner, tmp_path: Path
    ):
        """Missing seed (None) round-trips as None, not KeyError."""
        r = Run(phantom=phantom, source=source, scanner=scanner, seed=None)
        run_dir = tmp_path / "run"
        r.save(run_dir)
        loaded = Run.load(run_dir)
        assert loaded.seed is None


# =====================================================================
# Equality
# =====================================================================


class TestEquality:
    def test_identical_runs_equal(self, phantom, source, scanner, sinogram):
        r1 = Run(phantom=phantom, source=source, scanner=scanner, sinogram=sinogram)
        r2 = Run(phantom=phantom, source=source, scanner=scanner, sinogram=sinogram)
        assert r1 == r2

    def test_different_metadata_not_equal(self, phantom, source, scanner):
        r1 = Run(phantom=phantom, source=source, scanner=scanner,
                 metadata={"note": "a"})
        r2 = Run(phantom=phantom, source=source, scanner=scanner,
                 metadata={"note": "b"})
        assert r1 != r2

    def test_different_seeds_not_equal(self, phantom, source, scanner):
        r1 = Run(phantom=phantom, source=source, scanner=scanner, seed=1)
        r2 = Run(phantom=phantom, source=source, scanner=scanner, seed=2)
        assert r1 != r2

    def test_created_at_ignored_in_equality(
        self, phantom, source, scanner, tmp_path: Path
    ):
        """Two runs with the same content but saved at different times
        (different created_at) should still compare equal.
        """
        r = Run(phantom=phantom, source=source, scanner=scanner,
                metadata={"seed": 42})
        r.save(tmp_path / "a")
        r.save(tmp_path / "b")
        loaded_a = Run.load(tmp_path / "a")
        loaded_b = Run.load(tmp_path / "b")
        assert loaded_a == loaded_b

    def test_different_types_not_equal(self, phantom, source, scanner):
        r = Run(phantom=phantom, source=source, scanner=scanner)
        assert r != "not a run"
        assert r != 42


# =====================================================================
# Reference scene integration test
# =====================================================================


class TestReferenceScene:
    """The reference scene from PLAN.md — a water cylinder with a hot spot
    insert. Every phase validates against this same scene. For Phase 1,
    we just confirm we can construct and round-trip it without touching
    any simulator.
    """

    def test_water_cylinder_with_hot_spot_round_trip(self, tmp_path: Path):
        # Geometry: 64x64x64 grid, 2 cm radius cylinder, 4 cm tall
        phantom = Phantom.cylinder(
            shape=(64, 64, 64), voxel_size=(0.1, 0.1, 0.1),
            radius_cm=2.0, height_cm=4.0,
            inner_material="water", inner_density=1.0,
            outer_material="air", outer_density=0.0012,
            axis="z",
        )
        # Source: 1 MBq of F18 in the water + a hot spot at the center
        source = (
            Source.with_total_activity(
                phantom, material="water", total_activity_Bq=1_000_000.0,
                isotope="F18",
            )
            .add_hot_spot(
                position_cm=(3.2, 3.2, 3.2),  # roughly the center
                activity_Bq=500_000.0, radius_cm=0.3,
            )
        )
        scanner = Scanner(
            name="reference",
            detector_radius_cm=10.0,
            detector_axial_length_cm=10.0,
            energy_window_keV=(350.0, 650.0),
            energy_resolution=0.12,
            acquisition_time_s=1.0,
            n_radial_bins=16,
            n_angular_bins=16,
            n_z_slices=8,
            span=1,
            max_ring_difference=4,
        )
        r = Run(
            phantom=phantom, source=source, scanner=scanner,
            metadata={"scene": "reference_water_cylinder"},
        )
        run_dir = tmp_path / "reference_scene"
        r.save(run_dir)
        loaded = Run.load(run_dir)
        assert loaded == r

        # Bonus: verify the constructed scene has sensible physical properties
        assert phantom.mass_of("water") > 0
        assert source.total_activity_Bq == pytest.approx(1_500_000.0, rel=1e-3)