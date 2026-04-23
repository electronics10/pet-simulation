"""
Unit tests for petsim.phantom.

Covers the Phase 1 validation criteria relevant to Phantom:
  - Instantiation with plausible inputs does not raise.
  - Phantom.cylinder voxel count matches analytical expectation within 5%.
  - All factory methods (uniform, cube, cylinder, sphere, from_numpy) work.
  - save() / load() round-trip preserves all data exactly.
  - Invalid inputs raise clear exceptions.

Run with:
    uv run pytest tests/test_phantom.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from petsim.phantom import Phantom


# =====================================================================
# Instantiation
# =====================================================================


class TestInstantiation:
    def test_from_numpy_basic(self):
        mid = np.ones((4, 4, 4), dtype=np.int32)
        dens = np.full((4, 4, 4), 1.0, dtype=np.float32)
        p = Phantom.from_numpy(mid, dens, (1.0, 1.0, 1.0), ("water",))
        assert p.shape == (4, 4, 4)
        assert p.voxel_size == (1.0, 1.0, 1.0)
        assert p.material_names == ("water",)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Phantom.from_numpy(
                np.ones((4, 4, 4), dtype=np.int32),
                np.ones((4, 4, 5), dtype=np.float32),
                (1.0, 1.0, 1.0),
                ("water",),
            )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="3D"):
            Phantom.from_numpy(
                np.ones((4, 4), dtype=np.int32),
                np.ones((4, 4), dtype=np.float32),
                (1.0, 1.0, 1.0),
                ("water",),
            )

    def test_zero_voxel_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Phantom.from_numpy(
                np.ones((2, 2, 2), dtype=np.int32),
                np.ones((2, 2, 2), dtype=np.float32),
                (0.0, 1.0, 1.0),
                ("water",),
            )

    def test_material_id_zero_raises(self):
        """material_ids must be 1-indexed; 0 is reserved."""
        mid = np.zeros((2, 2, 2), dtype=np.int32)
        dens = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="1-indexed"):
            Phantom.from_numpy(mid, dens, (1.0, 1.0, 1.0), ("water",))

    def test_material_id_out_of_range_raises(self):
        mid = np.full((2, 2, 2), 5, dtype=np.int32)
        dens = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="only"):
            Phantom.from_numpy(mid, dens, (1.0, 1.0, 1.0), ("water",))

    def test_empty_material_names_raises(self):
        mid = np.ones((2, 2, 2), dtype=np.int32)
        dens = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="material_names"):
            Phantom.from_numpy(mid, dens, (1.0, 1.0, 1.0), ())


# =====================================================================
# Factory methods
# =====================================================================


class TestUniform:
    def test_all_voxels_are_inner(self):
        p = Phantom.uniform(
            shape=(8, 8, 8), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        assert np.all(p.material_ids == 1)
        assert np.all(p.densities == 1.0)
        assert p.material_names == ("water",)


class TestCube:
    def test_sample_geometry_matches_reference(self):
        """The MCGPU-PET sample uses a 5x5x5 water cube in a 9x9x9 grid
        with 2-voxel-thick air walls. Reproduce that pattern exactly.
        """
        p = Phantom.cube(
            shape=(9, 9, 9),
            voxel_size=(1.0, 1.0, 1.0),
            inner_material="water",
            inner_density=1.0,
            outer_material="air",
            outer_density=0.0012,
            inner_size_vox=5,
        )
        # Inner 5x5x5 should be water (id=2)
        inner = p.material_ids[2:7, 2:7, 2:7]
        assert np.all(inner == 2)
        # Everything outside the inner cube should be air (id=1)
        outer_mask = np.ones((9, 9, 9), dtype=bool)
        outer_mask[2:7, 2:7, 2:7] = False
        assert np.all(p.material_ids[outer_mask] == 1)


class TestCylinder:
    def test_voxel_count_matches_analytical_within_5_percent(self):
        """Validation criterion from PLAN.md Phase 1:
        Phantom.cylinder voxel count inside the cylinder must match the
        analytical expectation within 5%.
        """
        # High-resolution grid so discretization error is small
        nx = ny = nz = 64
        voxel_size = (0.1, 0.1, 0.1)  # cm, so phantom is 6.4 cm on a side
        radius_cm = 2.0
        height_cm = 4.0

        p = Phantom.cylinder(
            shape=(nx, ny, nz),
            voxel_size=voxel_size,
            radius_cm=radius_cm,
            height_cm=height_cm,
            inner_material="water",
            inner_density=1.0,
            axis="z",
        )

        # Count voxels marked as inner (water, id=2)
        inner_voxel_count = int(np.sum(p.material_ids == 2))
        inner_volume_cm3 = inner_voxel_count * p.voxel_volume_cm3

        expected_volume_cm3 = math.pi * radius_cm**2 * height_cm
        rel_error = abs(inner_volume_cm3 - expected_volume_cm3) / expected_volume_cm3

        assert rel_error < 0.05, (
            f"cylinder volume off by {rel_error:.1%}: "
            f"got {inner_volume_cm3:.3f} cm^3, expected {expected_volume_cm3:.3f} cm^3"
        )

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_cylinder_axis_orientations(self, axis):
        p = Phantom.cylinder(
            shape=(32, 32, 32),
            voxel_size=(0.1, 0.1, 0.1),
            radius_cm=1.0,
            height_cm=2.0,
            inner_material="water",
            inner_density=1.0,
            axis=axis,
        )
        # Just check it produces a non-trivial amount of inner material
        inner_count = int(np.sum(p.material_ids == 2))
        assert inner_count > 0
        assert inner_count < 32**3  # not filling the whole thing

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis"):
            Phantom.cylinder(
                shape=(16, 16, 16), voxel_size=(0.1, 0.1, 0.1),
                radius_cm=1.0, height_cm=1.0,
                inner_material="water", inner_density=1.0,
                axis="w",
            )


class TestSphere:
    def test_voxel_count_matches_analytical_within_5_percent(self):
        nx = ny = nz = 64
        voxel_size = (0.1, 0.1, 0.1)
        radius_cm = 2.0

        p = Phantom.sphere(
            shape=(nx, ny, nz),
            voxel_size=voxel_size,
            radius_cm=radius_cm,
            inner_material="water",
            inner_density=1.0,
        )

        inner_voxel_count = int(np.sum(p.material_ids == 2))
        inner_volume_cm3 = inner_voxel_count * p.voxel_volume_cm3
        expected_volume_cm3 = (4 / 3) * math.pi * radius_cm**3
        rel_error = abs(inner_volume_cm3 - expected_volume_cm3) / expected_volume_cm3

        assert rel_error < 0.05, (
            f"sphere volume off by {rel_error:.1%}: "
            f"got {inner_volume_cm3:.3f}, expected {expected_volume_cm3:.3f}"
        )


# =====================================================================
# Properties
# =====================================================================


class TestProperties:
    def test_shape_and_extent(self):
        p = Phantom.uniform(
            shape=(4, 6, 8), voxel_size=(0.5, 1.0, 2.0),
            material="water", density=1.0,
        )
        assert p.shape == (4, 6, 8)
        assert p.extent_cm == (2.0, 6.0, 16.0)
        assert p.voxel_volume_cm3 == pytest.approx(1.0)

    def test_mass_of_water_in_uniform_phantom(self):
        p = Phantom.uniform(
            shape=(10, 10, 10), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        # 1000 voxels * 1 cm^3 each * 1 g/cm^3 = 1000 g
        assert p.mass_of("water") == pytest.approx(1000.0)

    def test_mass_of_unknown_material_raises(self):
        p = Phantom.uniform(
            shape=(4, 4, 4), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        with pytest.raises(KeyError, match="brain"):
            p.mass_of("brain")


# =====================================================================
# Persistence (save/load round-trip)
# =====================================================================


class TestPersistence:
    def test_round_trip_preserves_everything(self, tmp_path: Path):
        original = Phantom.cylinder(
            shape=(16, 16, 16),
            voxel_size=(0.5, 0.5, 0.5),
            radius_cm=3.0,
            height_cm=6.0,
            inner_material="water",
            inner_density=1.0,
            outer_material="air",
            outer_density=0.0012,
        )
        out = tmp_path / "phantom.npz"
        original.save(out)
        loaded = Phantom.load(out)

        assert loaded == original
        assert np.array_equal(loaded.material_ids, original.material_ids)
        assert np.array_equal(loaded.densities, original.densities)
        assert loaded.voxel_size == original.voxel_size
        assert loaded.material_names == original.material_names

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        p = Phantom.uniform(
            shape=(2, 2, 2), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        nested = tmp_path / "a" / "b" / "c" / "phantom.npz"
        p.save(nested)
        assert nested.exists()


# =====================================================================
# Equality
# =====================================================================


class TestEquality:
    def test_identical_phantoms_equal(self):
        p1 = Phantom.uniform((4, 4, 4), (1.0, 1.0, 1.0), "water", 1.0)
        p2 = Phantom.uniform((4, 4, 4), (1.0, 1.0, 1.0), "water", 1.0)
        assert p1 == p2

    def test_different_materials_not_equal(self):
        p1 = Phantom.uniform((4, 4, 4), (1.0, 1.0, 1.0), "water", 1.0)
        p2 = Phantom.uniform((4, 4, 4), (1.0, 1.0, 1.0), "brain", 1.0)
        assert p1 != p2

    def test_different_types_not_equal(self):
        p1 = Phantom.uniform((4, 4, 4), (1.0, 1.0, 1.0), "water", 1.0)
        assert p1 != "not a phantom"
        assert p1 != 42