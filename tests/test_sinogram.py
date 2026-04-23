"""
Unit tests for petsim.sinogram.

Phase 1 validation criteria covered here:
  - Instantiation with plausible inputs does not raise.
  - Shape validation against the Scanner's sinogram_shape works.
  - Negative counts are rejected.
  - At least one component is required.
  - scatter_fraction matches expected formula.
  - save() / load() round-trip preserves arrays and metadata exactly.
  - `measured` property sums components correctly.

Run with:
    uv run pytest tests/test_sinogram.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from petsim.scanner import Scanner
from petsim.sinogram import Sinogram


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def toy_scanner():
    """A small scanner so test arrays stay tiny."""
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
def sample_trues(toy_scanner):
    """A small deterministic array in the scanner's sinogram shape."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 5, size=toy_scanner.sinogram_shape, dtype=np.int32)


@pytest.fixture
def sample_scatter(toy_scanner):
    rng = np.random.default_rng(1)
    return rng.integers(0, 3, size=toy_scanner.sinogram_shape, dtype=np.int32)


# =====================================================================
# Instantiation
# =====================================================================


class TestInstantiation:
    def test_trues_only(self, toy_scanner, sample_trues):
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        assert sino.shape == toy_scanner.sinogram_shape
        assert sino.scatter is None
        assert sino.randoms is None

    def test_trues_and_scatter(self, toy_scanner, sample_trues, sample_scatter):
        sino = Sinogram(
            scanner=toy_scanner,
            trues=sample_trues,
            scatter=sample_scatter,
        )
        assert sino.total_trues == int(sample_trues.sum())
        assert sino.total_scatter == int(sample_scatter.sum())

    def test_all_three_components(self, toy_scanner, sample_trues, sample_scatter):
        rng = np.random.default_rng(2)
        randoms = rng.integers(0, 2, size=toy_scanner.sinogram_shape, dtype=np.int32)
        sino = Sinogram(
            scanner=toy_scanner,
            trues=sample_trues,
            scatter=sample_scatter,
            randoms=randoms,
        )
        assert sino.total_randoms == int(randoms.sum())

    def test_shape_mismatch_raises(self, toy_scanner):
        wrong = np.zeros((2, 2, 2), dtype=np.int32)
        with pytest.raises(ValueError, match="shape"):
            Sinogram(scanner=toy_scanner, trues=wrong)

    def test_negative_counts_rejected(self, toy_scanner):
        bad = np.zeros(toy_scanner.sinogram_shape, dtype=np.int32)
        bad[0, 0, 0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            Sinogram(scanner=toy_scanner, trues=bad)

    def test_no_components_raises(self, toy_scanner):
        with pytest.raises(ValueError, match="at least one"):
            Sinogram(scanner=toy_scanner)

    def test_only_scatter_is_valid(self, toy_scanner, sample_scatter):
        """Scatter-only sinogram should be allowed — useful as a
        ground-truth target in training.
        """
        sino = Sinogram(scanner=toy_scanner, scatter=sample_scatter)
        assert sino.trues is None
        assert sino.scatter is not None


# =====================================================================
# Properties
# =====================================================================


class TestProperties:
    def test_shape_matches_scanner(self, toy_scanner, sample_trues):
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        assert sino.shape == toy_scanner.sinogram_shape

    def test_measured_sums_components(
        self, toy_scanner, sample_trues, sample_scatter
    ):
        sino = Sinogram(
            scanner=toy_scanner, trues=sample_trues, scatter=sample_scatter,
        )
        expected = sample_trues.astype(np.float64) + sample_scatter.astype(np.float64)
        assert np.allclose(sino.measured, expected)

    def test_measured_treats_missing_as_zero(self, toy_scanner, sample_trues):
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        assert np.allclose(sino.measured, sample_trues)

    def test_scatter_fraction_formula(
        self, toy_scanner, sample_trues, sample_scatter
    ):
        """scatter / (trues + scatter) matches SF standard definition."""
        sino = Sinogram(
            scanner=toy_scanner, trues=sample_trues, scatter=sample_scatter,
        )
        expected = sample_scatter.sum() / (sample_trues.sum() + sample_scatter.sum())
        assert sino.scatter_fraction == pytest.approx(expected)

    def test_scatter_fraction_none_when_missing(
        self, toy_scanner, sample_trues
    ):
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        assert sino.scatter_fraction is None

    def test_scatter_fraction_zero_when_empty(self, toy_scanner):
        """SF should be 0, not NaN, when both components are zero arrays."""
        zeros = np.zeros(toy_scanner.sinogram_shape, dtype=np.int32)
        sino = Sinogram(scanner=toy_scanner, trues=zeros, scatter=zeros)
        assert sino.scatter_fraction == 0.0

    def test_totals_zero_when_missing(self, toy_scanner, sample_trues):
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        assert sino.total_scatter == 0
        assert sino.total_randoms == 0


# =====================================================================
# Persistence
# =====================================================================


class TestPersistence:
    def test_round_trip_preserves_arrays_and_metadata(
        self, toy_scanner, sample_trues, sample_scatter, tmp_path: Path
    ):
        original = Sinogram(
            scanner=toy_scanner,
            trues=sample_trues,
            scatter=sample_scatter,
            metadata={
                "backend": "mcgpu",
                "simulated_histories": 23464,
                "wall_time_seconds": 4.8,
                "seed": 266280817,
            },
        )
        out = tmp_path / "sinogram.npz"
        original.save(out)
        loaded = Sinogram.load(out, scanner=toy_scanner)

        assert loaded == original
        assert np.array_equal(loaded.trues, original.trues)
        assert np.array_equal(loaded.scatter, original.scatter)
        assert loaded.metadata == original.metadata

    def test_round_trip_with_only_trues(
        self, toy_scanner, sample_trues, tmp_path: Path
    ):
        original = Sinogram(scanner=toy_scanner, trues=sample_trues)
        out = tmp_path / "sinogram.npz"
        original.save(out)
        loaded = Sinogram.load(out, scanner=toy_scanner)

        assert loaded == original
        assert loaded.scatter is None
        assert loaded.randoms is None

    def test_save_creates_parent_dirs(
        self, toy_scanner, sample_trues, tmp_path: Path
    ):
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        nested = tmp_path / "a" / "b" / "sinogram.npz"
        sino.save(nested)
        assert nested.exists()

    def test_load_uses_provided_scanner(
        self, toy_scanner, sample_trues, tmp_path: Path
    ):
        """The scanner is stored separately (in scanner.yaml); load()
        takes it as an argument and that's what the loaded Sinogram uses.
        """
        sino = Sinogram(scanner=toy_scanner, trues=sample_trues)
        out = tmp_path / "sinogram.npz"
        sino.save(out)

        # Load with a freshly re-instantiated (but equal) scanner
        fresh_scanner = Scanner(
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
        loaded = Sinogram.load(out, scanner=fresh_scanner)
        assert loaded.scanner == toy_scanner
        assert loaded.scanner is fresh_scanner  # identity, not just equality


# =====================================================================
# Equality
# =====================================================================


class TestEquality:
    def test_identical_sinograms_equal(self, toy_scanner, sample_trues):
        s1 = Sinogram(scanner=toy_scanner, trues=sample_trues)
        s2 = Sinogram(scanner=toy_scanner, trues=sample_trues.copy())
        assert s1 == s2

    def test_different_arrays_not_equal(self, toy_scanner):
        zeros = np.zeros(toy_scanner.sinogram_shape, dtype=np.int32)
        ones = np.ones(toy_scanner.sinogram_shape, dtype=np.int32)
        s1 = Sinogram(scanner=toy_scanner, trues=zeros)
        s2 = Sinogram(scanner=toy_scanner, trues=ones)
        assert s1 != s2

    def test_different_metadata_not_equal(self, toy_scanner, sample_trues):
        s1 = Sinogram(
            scanner=toy_scanner, trues=sample_trues, metadata={"seed": 1}
        )
        s2 = Sinogram(
            scanner=toy_scanner, trues=sample_trues, metadata={"seed": 2}
        )
        assert s1 != s2

    def test_different_types_not_equal(self, toy_scanner, sample_trues):
        s1 = Sinogram(scanner=toy_scanner, trues=sample_trues)
        assert s1 != "not a sinogram"
        assert s1 != 42