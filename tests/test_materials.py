"""
Unit tests for petsim.materials.

Phase 1 validation criteria covered here:
  - All 17 ICRP 110 tissues are present in the default registry.
  - Name lookup is case-insensitive.
  - Unknown material lookup raises a clear exception.
  - register() adds new materials and warns on conflicting overwrites.
  - mcgpu_path() returns the correct filesystem path.
  - verify_mcgpu_files() identifies missing files.

Run with:
    uv run pytest tests/test_materials.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from petsim.materials import Material, MaterialRegistry


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def registry(tmp_path: Path) -> MaterialRegistry:
    """A registry with the default materials and a fake materials dir."""
    return MaterialRegistry(mcgpu_materials_dir=tmp_path)


# =====================================================================
# Defaults
# =====================================================================


class TestDefaults:
    def test_basics_present(self, registry):
        """air and water must exist — these are needed for every phantom."""
        assert "air" in registry
        assert "water" in registry

    def test_icrp110_tissues_present(self, registry):
        """Phase 1 validation criterion: all 17 ICRP 110 tissues from the
        MCGPU material zip must be in the registry. 'water' and 'air' are
        counted here too since they're in the zip.
        """
        expected = {
            "adipose", "air", "blood", "brain", "breast_glandular",
            "cartilage", "eyes", "glands", "kidneys", "liver",
            "lung", "muscle", "skin", "soft_tissue", "spongiosa",
            "stomach_intestines", "water",
        }
        # All 17 expected tissues must be registered
        assert expected.issubset(set(registry.names()))

    def test_brain_has_icrp_definition(self, registry):
        """Sanity check the brain entry points at ICRP 110 files."""
        m = registry["brain"]
        assert m.nominal_density == pytest.approx(1.04)
        assert "ICRP110" in m.mcgpu_file
        assert m.gate_name == "G4_BRAIN_ICRP"

    def test_water_definition(self, registry):
        m = registry["water"]
        assert m.nominal_density == 1.0
        assert m.gate_name == "G4_WATER"


# =====================================================================
# Lookup
# =====================================================================


class TestLookup:
    def test_case_insensitive(self, registry):
        assert registry["BRAIN"] == registry["brain"]
        assert registry["Brain"] == registry["brain"]

    def test_in_is_case_insensitive(self, registry):
        assert "WATER" in registry
        assert "Water" in registry
        assert "water" in registry

    def test_unknown_material_raises(self, registry):
        with pytest.raises(KeyError, match="plutonium"):
            registry["plutonium"]

    def test_unknown_material_lists_available(self, registry):
        """Error message should help users find the right name."""
        with pytest.raises(KeyError, match="available"):
            registry["brian"]  # typo

    def test_names_returns_sorted_list(self, registry):
        names = registry.names()
        assert names == sorted(names)


# =====================================================================
# MCGPU path resolution
# =====================================================================


class TestMCGPUPath:
    def test_mcgpu_path_is_under_materials_dir(self, registry, tmp_path):
        path = registry.mcgpu_path("water")
        assert path.parent == tmp_path
        assert path.name == "water_5-515keV.mcgpu.gz"

    def test_mcgpu_path_for_icrp_tissue(self, registry, tmp_path):
        path = registry.mcgpu_path("brain")
        assert path.name == "brain_ICRP110_5-515keV.mcgpu.gz"

    def test_mcgpu_path_case_insensitive(self, registry):
        assert registry.mcgpu_path("BRAIN") == registry.mcgpu_path("brain")

    def test_verify_mcgpu_files_detects_missing(self, tmp_path):
        # Empty materials dir - all files are missing
        reg = MaterialRegistry(mcgpu_materials_dir=tmp_path)
        missing = reg.verify_mcgpu_files(["water", "brain"])
        assert set(missing) == {"water", "brain"}

    def test_verify_mcgpu_files_accepts_present(self, tmp_path):
        # Create fake material files so they pass the existence check
        (tmp_path / "water_5-515keV.mcgpu.gz").write_bytes(b"fake")
        (tmp_path / "brain_ICRP110_5-515keV.mcgpu.gz").write_bytes(b"fake")
        reg = MaterialRegistry(mcgpu_materials_dir=tmp_path)
        missing = reg.verify_mcgpu_files(["water", "brain"])
        assert missing == []


# =====================================================================
# GATE helpers
# =====================================================================


class TestGATE:
    def test_gate_name_lookup(self, registry):
        assert registry.gate_name("water") == "G4_WATER"
        assert registry.gate_name("brain") == "G4_BRAIN_ICRP"

    def test_gate_name_case_insensitive(self, registry):
        assert registry.gate_name("WATER") == "G4_WATER"


# =====================================================================
# Registration
# =====================================================================


class TestRegistration:
    def test_register_new_material(self, registry):
        lyso = Material(
            name="LYSO",
            nominal_density=7.1,
            mcgpu_file="lyso_5-515keV.mcgpu.gz",
            gate_name="LYSO",
        )
        registry.register(lyso)
        assert "lyso" in registry
        assert registry["LYSO"] == lyso

    def test_reregister_same_definition_is_silent(self, registry, recwarn):
        """Re-registering an identical Material should not warn."""
        original = registry["water"]
        registry.register(original)
        assert len(recwarn) == 0

    def test_register_conflicting_definition_warns(self, registry):
        """Overwriting an existing material with a different definition
        should produce a warning so users notice shadowing.
        """
        different_water = Material(
            name="water",
            nominal_density=0.99,  # changed
            mcgpu_file="water_5-515keV.mcgpu.gz",
            gate_name="G4_WATER",
        )
        with pytest.warns(UserWarning, match="overwriting"):
            registry.register(different_water)
        assert registry["water"].nominal_density == 0.99


# =====================================================================
# Material dataclass
# =====================================================================


class TestMaterialDataclass:
    def test_material_is_hashable(self):
        """frozen=True on Material means it's hashable — required if we
        ever want to put Materials in a set or use as a dict key."""
        m = Material("x", 1.0, "x.gz", "G4_X")
        assert hash(m) == hash(m)
        assert {m, m} == {m}

    def test_material_equality(self):
        m1 = Material("x", 1.0, "x.gz", "G4_X")
        m2 = Material("x", 1.0, "x.gz", "G4_X")
        m3 = Material("x", 2.0, "x.gz", "G4_X")
        assert m1 == m2
        assert m1 != m3