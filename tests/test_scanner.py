"""
Unit tests for petsim.scanner.

Phase 1 validation criteria covered here:
  - Instantiation with plausible inputs does not raise.
  - Presets load correctly and parameters match reference scanners.
  - from_preset() with overrides works.
  - save() / load() YAML round-trip preserves everything exactly.
  - Invalid inputs raise clear exceptions.
  - sinogram_shape property returns the correct ordering.

Run with:
    uv run pytest tests/test_scanner.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from petsim.scanner import SCANNER_PRESETS, Scanner


# =====================================================================
# Instantiation
# =====================================================================


class TestInstantiation:
    def test_minimum_valid_scanner(self):
        s = Scanner(
            name="toy",
            detector_radius_cm=10.0,
            detector_axial_length_cm=15.0,
            energy_window_keV=(350.0, 650.0),
            energy_resolution=0.12,
            acquisition_time_s=1.0,
            n_radial_bins=128,
            n_angular_bins=128,
            n_z_slices=63,
            span=3,
            max_ring_difference=31,
        )
        assert s.name == "toy"
        assert s.energy_window_eV == (350000.0, 650000.0)
        assert s.sinogram_shape == (63, 128, 128)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="detector_radius_cm"):
            Scanner(
                name="bad", detector_radius_cm=-1.0,
                detector_axial_length_cm=10.0,
                energy_window_keV=(350.0, 650.0), energy_resolution=0.12,
                acquisition_time_s=1.0, n_radial_bins=128, n_angular_bins=128,
                n_z_slices=63, span=3, max_ring_difference=31,
            )

    def test_inverted_energy_window_raises(self):
        with pytest.raises(ValueError, match="energy_window_keV"):
            Scanner(
                name="bad", detector_radius_cm=10.0,
                detector_axial_length_cm=10.0,
                energy_window_keV=(600.0, 350.0),  # inverted
                energy_resolution=0.12,
                acquisition_time_s=1.0, n_radial_bins=128, n_angular_bins=128,
                n_z_slices=63, span=3, max_ring_difference=31,
            )

    def test_energy_resolution_out_of_range_raises(self):
        with pytest.raises(ValueError, match="energy_resolution"):
            Scanner(
                name="bad", detector_radius_cm=10.0,
                detector_axial_length_cm=10.0,
                energy_window_keV=(350.0, 650.0),
                energy_resolution=1.5,  # > 1
                acquisition_time_s=1.0, n_radial_bins=128, n_angular_bins=128,
                n_z_slices=63, span=3, max_ring_difference=31,
            )

    def test_zero_bins_raises(self):
        with pytest.raises(ValueError, match="n_radial_bins"):
            Scanner(
                name="bad", detector_radius_cm=10.0,
                detector_axial_length_cm=10.0,
                energy_window_keV=(350.0, 650.0), energy_resolution=0.12,
                acquisition_time_s=1.0, n_radial_bins=0, n_angular_bins=128,
                n_z_slices=63, span=3, max_ring_difference=31,
            )

    def test_zero_acquisition_time_raises(self):
        with pytest.raises(ValueError, match="acquisition_time_s"):
            Scanner(
                name="bad", detector_radius_cm=10.0,
                detector_axial_length_cm=10.0,
                energy_window_keV=(350.0, 650.0), energy_resolution=0.12,
                acquisition_time_s=0.0, n_radial_bins=128, n_angular_bins=128,
                n_z_slices=63, span=3, max_ring_difference=31,
            )

    def test_negative_coincidence_window_raises(self):
        with pytest.raises(ValueError, match="coincidence_window_ns"):
            Scanner(
                name="bad", detector_radius_cm=10.0,
                detector_axial_length_cm=10.0,
                energy_window_keV=(350.0, 650.0), energy_resolution=0.12,
                acquisition_time_s=1.0, n_radial_bins=128, n_angular_bins=128,
                n_z_slices=63, span=3, max_ring_difference=31,
                coincidence_window_ns=-1.0,
            )


# =====================================================================
# Presets
# =====================================================================


class TestPresets:
    def test_mcgpu_sample_matches_reference(self):
        """The mcgpu_sample preset must reproduce the parameters we
        actually observed in the MCGPU-PET.in sample simulation output:
        sinogram 147 radial x 168 angular x 1293 slices, span 11, MRD 79.
        """
        s = Scanner.from_preset("mcgpu_sample")
        assert s.n_radial_bins == 147
        assert s.n_angular_bins == 168
        assert s.n_z_slices == 1293
        assert s.span == 11
        assert s.max_ring_difference == 79
        assert s.energy_window_keV == (350.0, 600.0)
        assert s.sinogram_shape == (1293, 168, 147)

    def test_bruker_albira_preset_matches_gate_script(self):
        s = Scanner.from_preset("bruker_albira")
        assert s.detector_radius_cm == 5.8  # 58 mm inner radius
        assert s.energy_window_keV == (350.0, 650.0)
        assert s.coincidence_window_ns == 10.0
        assert s.crystal_material == "LYSO"
        assert s.n_rsectors == 8

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="unknown preset"):
            Scanner.from_preset("nonexistent_scanner")

    def test_from_preset_override(self):
        s = Scanner.from_preset(
            "mcgpu_sample", acquisition_time_s=60.0, name="custom",
        )
        assert s.acquisition_time_s == 60.0
        assert s.name == "custom"
        # Other fields still come from the preset
        assert s.n_radial_bins == 147

    def test_all_presets_instantiate(self):
        """Every preset in SCANNER_PRESETS must produce a valid Scanner."""
        for preset_name in SCANNER_PRESETS:
            s = Scanner.from_preset(preset_name)
            assert isinstance(s, Scanner)
            assert s.name


# =====================================================================
# Persistence
# =====================================================================


class TestPersistence:
    def test_round_trip_mcgpu_sample(self, tmp_path: Path):
        original = Scanner.from_preset("mcgpu_sample")
        out = tmp_path / "scanner.yaml"
        original.save(out)
        loaded = Scanner.load(out)
        assert loaded == original

    def test_round_trip_bruker(self, tmp_path: Path):
        """Bruker preset exercises the optional tuple fields
        (crystal_size_mm, crystals_per_module, modules_per_rsector).
        """
        original = Scanner.from_preset("bruker_albira")
        out = tmp_path / "scanner.yaml"
        original.save(out)
        loaded = Scanner.load(out)
        assert loaded == original
        # Tuples must still be tuples after round-trip
        assert isinstance(loaded.crystal_size_mm, tuple)
        assert isinstance(loaded.crystals_per_module, tuple)
        assert isinstance(loaded.modules_per_rsector, tuple)

    def test_round_trip_with_extra_dict(self, tmp_path: Path):
        s = Scanner.from_preset(
            "mcgpu_sample",
            extra={"note": "custom run", "nbins": 513, "nsegs": 15},
        )
        out = tmp_path / "scanner.yaml"
        s.save(out)
        loaded = Scanner.load(out)
        assert loaded == s
        assert loaded.extra["nbins"] == 513

    def test_yaml_is_human_readable(self, tmp_path: Path):
        s = Scanner.from_preset("mcgpu_sample")
        out = tmp_path / "scanner.yaml"
        s.save(out)
        content = out.read_text()
        # Some named fields should appear verbatim in the YAML
        assert "name: mcgpu_sample" in content
        assert "detector_radius_cm:" in content
        assert "energy_window_keV:" in content

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        s = Scanner.from_preset("mcgpu_sample")
        nested = tmp_path / "a" / "b" / "c" / "scanner.yaml"
        s.save(nested)
        assert nested.exists()


# =====================================================================
# Equality and properties
# =====================================================================


class TestEquality:
    def test_identical_scanners_equal(self):
        s1 = Scanner.from_preset("mcgpu_sample")
        s2 = Scanner.from_preset("mcgpu_sample")
        assert s1 == s2

    def test_different_presets_not_equal(self):
        s1 = Scanner.from_preset("mcgpu_sample")
        s2 = Scanner.from_preset("bruker_albira")
        assert s1 != s2

    def test_different_types_not_equal(self):
        s1 = Scanner.from_preset("mcgpu_sample")
        assert s1 != "not a scanner"
        assert s1 != 42


class TestProperties:
    def test_sinogram_shape_ordering(self):
        """sinogram_shape is (n_z_slices, n_angular_bins, n_radial_bins)."""
        s = Scanner.from_preset("mcgpu_sample")
        assert s.sinogram_shape == (1293, 168, 147)

    def test_energy_window_unit_conversion(self):
        s = Scanner.from_preset("mcgpu_sample")
        assert s.energy_window_keV == (350.0, 600.0)
        assert s.energy_window_eV == (350000.0, 600000.0)