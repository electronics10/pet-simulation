"""
Unit tests for petsim.source.

Phase 1 validation criteria covered here:
  - Instantiation with plausible inputs does not raise.
  - Factory methods (zeros, from_numpy, uniform_in_material,
    with_total_activity) work as expected.
  - `with_total_activity` correctly scales to hit the target total Bq.
  - add_hot_spot adds the right amount at the right location.
  - save() / load() round-trip preserves all data exactly.
  - Invalid inputs raise clear exceptions.
  - Source matches() Phantom grids correctly.

Run with:
    uv run pytest tests/test_source.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from petsim.phantom import Phantom
from petsim.source import ISOTOPE_HALF_LIFE_S, Source


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def water_cube_phantom():
    """A 9x9x9 phantom with a 5x5x5 water cube in air - matches the
    MCGPU-PET sample geometry used as our reference scene.
    """
    return Phantom.cube(
        shape=(9, 9, 9),
        voxel_size=(1.0, 1.0, 1.0),
        inner_material="water",
        inner_density=1.0,
        outer_material="air",
        outer_density=0.0012,
        inner_size_vox=5,
    )


# =====================================================================
# Instantiation
# =====================================================================


class TestInstantiation:
    def test_direct_construction_works(self, water_cube_phantom):
        s = Source(
            activity_Bq=np.zeros(water_cube_phantom.shape, dtype=np.float32),
            voxel_size=water_cube_phantom.voxel_size,
            isotope="F18",
        )
        assert s.shape == (9, 9, 9)
        assert s.isotope == "F18"

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="3D"):
            Source(
                activity_Bq=np.zeros((4, 4), dtype=np.float32),
                voxel_size=(1.0, 1.0, 1.0),
                isotope="F18",
            )

    def test_negative_activity_raises(self):
        act = np.zeros((4, 4, 4), dtype=np.float32)
        act[0, 0, 0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            Source(activity_Bq=act, voxel_size=(1.0, 1.0, 1.0), isotope="F18")

    def test_unknown_isotope_raises(self):
        with pytest.raises(ValueError, match="unknown isotope"):
            Source(
                activity_Bq=np.zeros((2, 2, 2), dtype=np.float32),
                voxel_size=(1.0, 1.0, 1.0),
                isotope="Pu239",
            )

    def test_zero_voxel_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Source(
                activity_Bq=np.zeros((2, 2, 2), dtype=np.float32),
                voxel_size=(0.0, 1.0, 1.0),
                isotope="F18",
            )


# =====================================================================
# Factory methods
# =====================================================================


class TestZeros:
    def test_matches_phantom(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom)
        assert s.shape == water_cube_phantom.shape
        assert s.voxel_size == water_cube_phantom.voxel_size
        assert s.total_activity_Bq == 0.0
        assert s.matches(water_cube_phantom)


class TestFromNumpy:
    def test_works_with_matching_shape(self, water_cube_phantom):
        act = np.full(water_cube_phantom.shape, 10.0, dtype=np.float32)
        s = Source.from_numpy(water_cube_phantom, act, isotope="C11")
        assert s.isotope == "C11"
        assert s.total_activity_Bq == pytest.approx(10.0 * 9 * 9 * 9)

    def test_shape_mismatch_raises(self, water_cube_phantom):
        bad = np.zeros((8, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            Source.from_numpy(water_cube_phantom, bad)


class TestUniformInMaterial:
    def test_activity_only_inside_material(self, water_cube_phantom):
        s = Source.uniform_in_material(
            water_cube_phantom,
            material="water",
            activity_per_voxel_Bq=50.0,
        )
        # Water voxels = 5x5x5 = 125, each with 50 Bq
        assert s.total_activity_Bq == pytest.approx(50.0 * 125)
        # Air voxels should all be zero
        air_mask = water_cube_phantom.material_ids == 1
        assert np.all(s.activity_Bq[air_mask] == 0.0)

    def test_unknown_material_raises(self, water_cube_phantom):
        with pytest.raises(KeyError, match="brain"):
            Source.uniform_in_material(
                water_cube_phantom, material="brain", activity_per_voxel_Bq=1.0
            )

    def test_negative_activity_raises(self, water_cube_phantom):
        with pytest.raises(ValueError, match="non-negative"):
            Source.uniform_in_material(
                water_cube_phantom, material="water",
                activity_per_voxel_Bq=-1.0,
            )


class TestWithTotalActivity:
    def test_total_activity_hits_target(self, water_cube_phantom):
        """Phase 1 validation criterion: Source correctly scales a constant
        activity map to hit a target total Bq.
        """
        target_Bq = 1_000_000.0  # 1 MBq
        s = Source.with_total_activity(
            water_cube_phantom,
            material="water",
            total_activity_Bq=target_Bq,
        )
        assert s.total_activity_Bq == pytest.approx(target_Bq, rel=1e-5)

    def test_activity_is_uniform_within_material(self, water_cube_phantom):
        s = Source.with_total_activity(
            water_cube_phantom, material="water", total_activity_Bq=1000.0,
        )
        water_mask = water_cube_phantom.material_ids == 2
        water_activities = s.activity_Bq[water_mask]
        # All water voxels should have the same (non-zero) activity
        assert np.allclose(water_activities, water_activities[0])
        assert water_activities[0] > 0

    def test_raises_when_material_absent(self, water_cube_phantom):
        with pytest.raises(ValueError, match="zero voxels"):
            # The phantom has no 'brain' material, so this first fails with
            # KeyError. Build a custom phantom with brain declared but
            # never used.
            brain_phantom = Phantom.from_numpy(
                material_ids=np.ones((4, 4, 4), dtype=np.int32),
                densities=np.ones((4, 4, 4), dtype=np.float32),
                voxel_size=(1.0, 1.0, 1.0),
                material_names=("air", "brain"),  # brain declared but unused
            )
            Source.with_total_activity(
                brain_phantom, material="brain", total_activity_Bq=1.0,
            )


class TestAddHotSpot:
    def test_single_voxel_hot_spot(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom)
        # Place a hot spot at voxel (4, 4, 4) — center of the 9x9x9 grid
        s2 = s.add_hot_spot(position_cm=(4.5, 4.5, 4.5), activity_Bq=1000.0)
        assert s2.total_activity_Bq == pytest.approx(1000.0)
        assert s2.activity_Bq[4, 4, 4] == pytest.approx(1000.0)
        # Original unchanged
        assert s.total_activity_Bq == 0.0

    def test_hot_spot_accumulates(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom)
        s = s.add_hot_spot((4.5, 4.5, 4.5), 1000.0)
        s = s.add_hot_spot((4.5, 4.5, 4.5), 2000.0)
        assert s.activity_Bq[4, 4, 4] == pytest.approx(3000.0)

    def test_spherical_hot_spot(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom).add_hot_spot(
            position_cm=(4.5, 4.5, 4.5), activity_Bq=1000.0, radius_cm=1.5
        )
        # Total activity should be preserved (split across voxels)
        assert s.total_activity_Bq == pytest.approx(1000.0)
        # Several voxels should be hot
        n_hot = int((s.activity_Bq > 0).sum())
        assert n_hot > 1

    def test_out_of_bounds_raises(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom)
        with pytest.raises(ValueError, match="outside phantom extent"):
            s.add_hot_spot(position_cm=(100.0, 0.0, 0.0), activity_Bq=1.0)


# =====================================================================
# Properties
# =====================================================================


class TestProperties:
    def test_half_life_lookup(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom, isotope="F18")
        assert s.half_life_s == pytest.approx(ISOTOPE_HALF_LIFE_S["F18"])

    def test_mean_lifetime_relation(self, water_cube_phantom):
        """mean lifetime = half_life / ln(2)"""
        s = Source.zeros(water_cube_phantom, isotope="F18")
        assert s.mean_lifetime_s == pytest.approx(
            s.half_life_s / np.log(2)
        )

    def test_decay_factor_at_zero(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom, isotope="F18")
        assert s.decay_factor(0.0) == pytest.approx(1.0)

    def test_decay_factor_at_one_half_life(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom, isotope="F18")
        assert s.decay_factor(s.half_life_s) == pytest.approx(0.5, rel=1e-4)

    def test_decay_factor_negative_time_raises(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom, isotope="F18")
        with pytest.raises(ValueError, match="non-negative"):
            s.decay_factor(-1.0)

    def test_matches_phantom(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom)
        assert s.matches(water_cube_phantom)

    def test_does_not_match_different_shape(self, water_cube_phantom):
        other = Phantom.uniform(
            shape=(8, 8, 8), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        s = Source.zeros(water_cube_phantom)
        assert not s.matches(other)


# =====================================================================
# Diagnostics
# =====================================================================


class TestCheckActivityInAir:
    def test_no_air_activity_when_using_uniform_in_material(
        self, water_cube_phantom
    ):
        s = Source.uniform_in_material(
            water_cube_phantom, material="water", activity_per_voxel_Bq=50.0
        )
        n, total = s.check_activity_in_air(water_cube_phantom)
        assert n == 0
        assert total == 0.0

    def test_detects_activity_in_air(self, water_cube_phantom):
        s = Source.zeros(water_cube_phantom)
        # (0.5, 0.5, 0.5) is in the outer air layer for this phantom
        s = s.add_hot_spot(position_cm=(0.5, 0.5, 0.5), activity_Bq=100.0)
        n, total = s.check_activity_in_air(water_cube_phantom)
        assert n == 1
        assert total == pytest.approx(100.0)

    def test_returns_zero_when_no_air_material(self, water_cube_phantom):
        uniform = Phantom.uniform(
            shape=(4, 4, 4), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        s = Source.uniform_in_material(
            uniform, material="water", activity_per_voxel_Bq=10.0
        )
        n, total = s.check_activity_in_air(uniform)
        assert n == 0
        assert total == 0.0


# =====================================================================
# Persistence
# =====================================================================


class TestPersistence:
    def test_round_trip_preserves_everything(
        self, water_cube_phantom, tmp_path: Path
    ):
        original = Source.with_total_activity(
            water_cube_phantom, material="water",
            total_activity_Bq=1_000_000.0, isotope="F18",
        ).add_hot_spot(position_cm=(4.5, 4.5, 4.5), activity_Bq=5000.0)

        out = tmp_path / "source.npz"
        original.save(out)
        loaded = Source.load(out)

        assert loaded == original
        assert np.array_equal(loaded.activity_Bq, original.activity_Bq)
        assert loaded.voxel_size == original.voxel_size
        assert loaded.isotope == original.isotope
        assert loaded.total_activity_Bq == pytest.approx(
            original.total_activity_Bq
        )

    def test_save_creates_parent_dirs(
        self, water_cube_phantom, tmp_path: Path
    ):
        s = Source.zeros(water_cube_phantom)
        nested = tmp_path / "a" / "b" / "source.npz"
        s.save(nested)
        assert nested.exists()


# =====================================================================
# Equality
# =====================================================================


class TestEquality:
    def test_identical_sources_equal(self, water_cube_phantom):
        s1 = Source.zeros(water_cube_phantom, isotope="F18")
        s2 = Source.zeros(water_cube_phantom, isotope="F18")
        assert s1 == s2

    def test_different_isotopes_not_equal(self, water_cube_phantom):
        s1 = Source.zeros(water_cube_phantom, isotope="F18")
        s2 = Source.zeros(water_cube_phantom, isotope="C11")
        assert s1 != s2

    def test_different_types_not_equal(self, water_cube_phantom):
        s1 = Source.zeros(water_cube_phantom)
        assert s1 != "not a source"
        assert s1 != 42