"""
Tests for the GATE backend (backends/gate.py).

The `opengate` package is not installed in this test environment, so
tests focus on the parts of the backend that don't require GATE:

  - .mhd/.raw export of phantom and source
  - material name mapping
  - ImageVolume LUT construction
  - the ImportError path when opengate is missing
  - the minimal .mhd reader used on the output side

The full build/run path must be exercised on a machine with GATE 10
and the `opengate` Python package installed. A manual validation
recipe is documented at the bottom of this file.

Run with:
    uv run pytest tests/test_gate.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from petsim.backends.gate import (
    GATEBackend,
    GATEConfig,
    GATERunResult,
    _petsim_to_gate_material,
    _PETSIM_TO_GATE_MATERIAL_MAP,
)
from petsim.phantom import Phantom
from petsim.run import Run
from petsim.scanner import Scanner
from petsim.source import Source


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def basic_phantom():
    return Phantom.cube(
        shape=(5, 5, 5),
        voxel_size=(1.0, 1.0, 1.0),
        inner_material="water",
        inner_density=1.0,
        outer_material="air",
        outer_density=0.0012,
        inner_size_vox=3,
    )


@pytest.fixture
def basic_source(basic_phantom):
    return Source.with_total_activity(
        basic_phantom, material="water",
        total_activity_Bq=1e6, isotope="F18",
    )


@pytest.fixture
def basic_run(basic_phantom, basic_source):
    scanner = Scanner.from_preset("mcgpu_sample")
    return Run(phantom=basic_phantom, source=basic_source,
               scanner=scanner, seed=7)


# =====================================================================
# Material mapping
# =====================================================================


class TestMaterialMapping:
    def test_known_petsim_materials_map_to_geant4(self):
        assert _petsim_to_gate_material("air") == "G4_AIR"
        assert _petsim_to_gate_material("water") == "G4_WATER"
        assert _petsim_to_gate_material("bone") == "G4_BONE_COMPACT_ICRU"

    def test_case_insensitive(self):
        assert _petsim_to_gate_material("Water") == "G4_WATER"
        assert _petsim_to_gate_material("AIR") == "G4_AIR"

    def test_unknown_material_passes_through(self):
        """Unmapped names fall through unchanged — the user is expected
        to have them in a custom materials.db.
        """
        assert _petsim_to_gate_material("unobtanium") == "unobtanium"

    def test_mapping_covers_common_icrp_tissues(self):
        """Sanity: the usual PET-relevant tissues are all in the map."""
        required = {"air", "water", "bone", "soft_tissue", "lung",
                    "adipose", "muscle", "brain", "blood"}
        assert required.issubset(set(_PETSIM_TO_GATE_MATERIAL_MAP.keys()))


# =====================================================================
# ImageVolume LUT
# =====================================================================


class TestMaterialLUT:
    def test_lut_preserves_material_order(self, basic_phantom):
        lut = GATEBackend._build_material_lut(basic_phantom)
        # basic_phantom has ("air", "water") in that order → ids 1, 2
        assert lut[0] == (1, 1, "G4_AIR")
        assert lut[1] == (2, 2, "G4_WATER")

    def test_lut_length_matches_material_count(self, basic_phantom):
        lut = GATEBackend._build_material_lut(basic_phantom)
        assert len(lut) == len(basic_phantom.material_names)


# =====================================================================
# .mhd/.raw export
# =====================================================================


class TestMHDExport:
    def test_material_image_writes_mhd_and_raw(
        self, basic_phantom, tmp_path: Path
    ):
        path = tmp_path / "phantom.mhd"
        GATEBackend._write_mhd_material_image(basic_phantom, path)
        assert path.exists()
        assert (tmp_path / "phantom.raw").exists()

    def test_material_image_has_correct_dimensions(
        self, basic_phantom, tmp_path: Path
    ):
        path = tmp_path / "phantom.mhd"
        GATEBackend._write_mhd_material_image(basic_phantom, path)
        header = path.read_text()
        assert "DimSize = 5 5 5" in header
        # Voxel size 1.0 cm = 10.0 mm
        assert "ElementSpacing = 10.0 10.0 10.0" in header
        assert "ElementType = MET_USHORT" in header

    def test_material_image_roundtrip(
        self, basic_phantom, tmp_path: Path
    ):
        """Write out, read back with the internal reader, verify the
        values are preserved up to the z-y-x reordering.
        """
        path = tmp_path / "phantom.mhd"
        GATEBackend._write_mhd_material_image(basic_phantom, path)
        raw = GATEBackend._read_mhd(path)
        # Reader returns flat array; reshape to (z, y, x) and transpose
        # back to (x, y, z) to compare with phantom.material_ids.
        nx, ny, nz = basic_phantom.shape
        arr = raw.reshape((nz, ny, nx)).transpose(2, 1, 0)
        np.testing.assert_array_equal(
            arr.astype(basic_phantom.material_ids.dtype),
            basic_phantom.material_ids,
        )

    def test_activity_image_writes_float(
        self, basic_source, tmp_path: Path
    ):
        path = tmp_path / "source.mhd"
        GATEBackend._write_mhd_activity_image(basic_source, path)
        header = path.read_text()
        assert "ElementType = MET_FLOAT" in header

    def test_activity_image_roundtrip(
        self, basic_source, tmp_path: Path
    ):
        path = tmp_path / "source.mhd"
        GATEBackend._write_mhd_activity_image(basic_source, path)
        raw = GATEBackend._read_mhd(path)
        nx, ny, nz = basic_source.shape
        arr = raw.reshape((nz, ny, nx)).transpose(2, 1, 0)
        np.testing.assert_allclose(
            arr,
            basic_source.activity_Bq.astype(np.float32),
            rtol=0, atol=0,
        )


# =====================================================================
# Minimal .mhd reader
# =====================================================================


class TestMHDReader:
    def test_reads_float32_raw(self, tmp_path: Path):
        raw = np.arange(24, dtype=np.float32)
        (tmp_path / "img.raw").write_bytes(raw.tobytes())
        (tmp_path / "img.mhd").write_text(
            "ObjectType = Image\n"
            "NDims = 3\n"
            "DimSize = 2 3 4\n"
            "ElementType = MET_FLOAT\n"
            "ElementDataFile = img.raw\n"
        )
        out = GATEBackend._read_mhd(tmp_path / "img.mhd")
        np.testing.assert_array_equal(out, raw)

    def test_reads_ushort_raw(self, tmp_path: Path):
        raw = np.arange(10, dtype=np.uint16)
        (tmp_path / "labels.raw").write_bytes(raw.tobytes())
        (tmp_path / "labels.mhd").write_text(
            "ObjectType = Image\n"
            "DimSize = 2 5 1\n"
            "ElementType = MET_USHORT\n"
            "ElementDataFile = labels.raw\n"
        )
        out = GATEBackend._read_mhd(tmp_path / "labels.mhd")
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, raw)

    def test_defaults_to_float_for_unknown_type(self, tmp_path: Path):
        (tmp_path / "x.raw").write_bytes(b"\x00" * 16)
        (tmp_path / "x.mhd").write_text(
            "ObjectType = Image\n"
            "ElementType = MET_MYSTERY\n"
            "ElementDataFile = x.raw\n"
        )
        out = GATEBackend._read_mhd(tmp_path / "x.mhd")
        # Falls back to float32 (4 bytes each, 16 bytes → 4 elements)
        assert out.dtype == np.float32
        assert out.size == 4


# =====================================================================
# Config
# =====================================================================


class TestGATEConfig:
    def test_defaults(self):
        cfg = GATEConfig()
        assert cfg.physics_list == "G4EmStandardPhysics_option4"
        assert cfg.back_to_back_annihilation is True
        assert cfg.crystal_material == "LSO"

    def test_construction_with_overrides(self):
        cfg = GATEConfig(
            back_to_back_annihilation=False,
            crystal_material="BGO",
            n_threads=8,
        )
        assert cfg.back_to_back_annihilation is False
        assert cfg.crystal_material == "BGO"
        assert cfg.n_threads == 8


# =====================================================================
# Build: ImportError path
# =====================================================================


class TestBuildWithoutOpenGate:
    """Until opengate is installed, build() must fail with a clear error."""

    def test_build_raises_importerror_without_opengate(
        self, basic_run, tmp_path: Path, monkeypatch
    ):
        # Simulate opengate being unavailable
        import sys
        monkeypatch.setitem(sys.modules, "opengate", None)
        backend = GATEBackend()
        with pytest.raises(ImportError, match="opengate"):
            backend.build(basic_run, tmp_path / "run1")


# =====================================================================
# parse_sinogram
# =====================================================================


class TestParseSinogram:
    def test_parse_sinogram_reads_written_mhd(
        self, basic_run, tmp_path: Path
    ):
        """End-to-end parse: write a synthetic sinogram .mhd, verify the
        backend reads it back with the right shape.
        """
        shape = basic_run.scanner.sinogram_shape
        fake_sino = np.arange(int(np.prod(shape)),
                              dtype=np.float32).reshape(shape)
        # GATE writes in (z, y, x) order internally
        (tmp_path / "sinogram.raw").write_bytes(
            np.ascontiguousarray(fake_sino).tobytes()
        )
        nz, ny, nx = shape
        (tmp_path / "sinogram.mhd").write_text(
            f"ObjectType = Image\n"
            f"DimSize = {nx} {ny} {nz}\n"
            f"ElementType = MET_FLOAT\n"
            f"ElementDataFile = sinogram.raw\n"
        )
        backend = GATEBackend()
        result = GATERunResult(
            workdir=tmp_path,
            wall_time_s=1.0,
            sinogram_output_path=tmp_path / "sinogram.mhd",
        )
        sinogram = backend.parse_sinogram(basic_run, result)
        assert sinogram.trues.shape == shape
        assert sinogram.metadata["backend"] == "gate"

    def test_parse_sinogram_missing_file_raises(
        self, basic_run, tmp_path: Path
    ):
        backend = GATEBackend()
        result = GATERunResult(
            workdir=tmp_path,
            wall_time_s=0.0,
            sinogram_output_path=tmp_path / "does_not_exist.mhd",
        )
        with pytest.raises(FileNotFoundError, match="sinogram"):
            backend.parse_sinogram(basic_run, result)

    def test_parse_sinogram_wrong_size_raises(
        self, basic_run, tmp_path: Path
    ):
        """If the raw file has the wrong number of elements, fail with
        a clear error."""
        wrong = np.zeros(42, dtype=np.float32)
        (tmp_path / "sinogram.raw").write_bytes(wrong.tobytes())
        (tmp_path / "sinogram.mhd").write_text(
            "ObjectType = Image\n"
            "ElementType = MET_FLOAT\n"
            "ElementDataFile = sinogram.raw\n"
        )
        backend = GATEBackend()
        result = GATERunResult(
            workdir=tmp_path, wall_time_s=0.0,
            sinogram_output_path=tmp_path / "sinogram.mhd",
        )
        with pytest.raises(ValueError, match="expected"):
            backend.parse_sinogram(basic_run, result)


# =====================================================================
# Manual validation recipe
# =====================================================================
# To validate the full GATE pipeline, run the following on a machine
# with GATE 10 and opengate installed:
#
#     from petsim.phantom import Phantom
#     from petsim.source import Source
#     from petsim.scanner import Scanner
#     from petsim.run import Run
#     from petsim.backends.gate import GATEBackend, GATEConfig
#
#     phantom = Phantom.cube(
#         shape=(9,9,9), voxel_size=(1.0,1.0,1.0),
#         inner_material='water', inner_density=1.0,
#         outer_material='air', outer_density=0.0012,
#         inner_size_vox=5,
#     )
#     source = Source.with_total_activity(
#         phantom, material='water',
#         total_activity_Bq=1e6, isotope='F18',
#     )
#     scanner = Scanner.from_preset('mcgpu_sample')
#     run = Run(phantom=phantom, source=source, scanner=scanner, seed=1)
#
#     backend = GATEBackend()
#     sinogram, result = backend.run_full(run, '/tmp/gate_run')
#     print(f"GATE produced {sinogram.total_trues} trues in "
#           f"{result.wall_time_s:.1f} s")
#
# Expected: a Sinogram with non-zero trues, on a similar order of
# magnitude to the MCGPU-PET result (within ~10% per the MCGPU-PET
# paper's GATE validation).