"""
Unit tests for the MCGPU-PET backend's input-file writers.

`write_vox` is tested byte-exactly against hand-constructed expected
strings that follow MCGPU-PET's reference format rules. The actual
sample phantom_9x9x9cm.vox (which lives on the user's simulation server,
not in this test environment) is reproduced by
TestSampleSceneReproduction and can be byte-diffed against the real
sample file with a manual step documented at the bottom of this file.

`write_in` is tested structurally (section headers present, numeric
values correctly substituted). Byte-exact comparison to the distributed
sample is infeasible because the sample contains a placeholder isotope
mean life and leftover dose-ROI values from an unrelated phantom.

Run with:
    uv run pytest tests/test_mcgpu.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from petsim.backends.mcgpu import (
    VOX_HEADER_TEMPLATE,
    MCGPUConfig,
    write_in,
    write_vox,
)
from petsim.materials import MaterialRegistry
from petsim.phantom import Phantom
from petsim.run import Run
from petsim.scanner import Scanner
from petsim.source import Source


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def all_air_phantom():
    """9x9x9 phantom of pure air — matches the top and bottom Z-slices
    of the MCGPU-PET sample exactly.
    """
    return Phantom.uniform(
        shape=(9, 9, 9),
        voxel_size=(1.0, 1.0, 1.0),
        material="air",
        density=0.0012,
    )


@pytest.fixture
def zero_source(all_air_phantom):
    """A Source with zero activity everywhere, matching an all-air phantom."""
    return Source.zeros(all_air_phantom, isotope="F18")


@pytest.fixture
def sample_scene_phantom():
    """Reproduces the geometry of the MCGPU-PET sample:
    a 5×5×5 cm water cube centered in a 9×9×9 cm block of air.
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
# Byte-exact .vox: the all-air case
# =====================================================================


class TestAllAirVoxByteExact:
    """Byte-exact reproduction for a 9×9×9 all-air phantom. We can
    construct the expected string by hand from the format rules.
    """

    def _build_expected(
        self,
        nx: int = 9, ny: int = 9, nz: int = 9,
        dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
    ) -> str:
        header = VOX_HEADER_TEMPLATE.format(
            nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
        )
        parts = [header]
        voxel_line = "1 0.0012 0.0\n"
        for _ in range(nz):
            for _ in range(ny):
                for _ in range(nx):
                    parts.append(voxel_line)
                parts.append("\n")  # end of X
            parts.append("\n")      # end of Y
        parts.append("\n")          # trailing blank at EOF (see writer)
        return "".join(parts)

    def test_matches_hand_constructed_expected(
        self, all_air_phantom, zero_source, tmp_path: Path
    ):
        path = tmp_path / "phantom.vox"
        write_vox(all_air_phantom, zero_source, path)

        expected = self._build_expected()
        actual = path.read_text()
        assert actual == expected

    def test_header_section_markers(
        self, all_air_phantom, zero_source, tmp_path: Path
    ):
        path = tmp_path / "phantom.vox"
        write_vox(all_air_phantom, zero_source, path)
        content = path.read_text()
        assert content.startswith("[SECTION VOXELS HEADER v.2008-04-13]\n")
        assert "[END OF VXH SECTION]" in content

    def test_header_dimensions_line(
        self, all_air_phantom, zero_source, tmp_path: Path
    ):
        path = tmp_path / "phantom.vox"
        write_vox(all_air_phantom, zero_source, path)
        content = path.read_text()
        assert "9 9 9   No. OF VOXELS IN X,Y,Z" in content
        assert "1.0 1.0 1.0   VOXEL SIZE (cm) ALONG X,Y,Z" in content

    def test_total_voxel_count(
        self, all_air_phantom, zero_source, tmp_path: Path
    ):
        path = tmp_path / "phantom.vox"
        write_vox(all_air_phantom, zero_source, path)
        content = path.read_text()
        # 9*9*9 = 729 voxel lines, each "1 0.0012 0.0"
        assert content.count("1 0.0012 0.0\n") == 729


# =====================================================================
# Byte-exact .vox: the sample scene
# =====================================================================


class TestSampleSceneReproduction:
    """Reproduces the format structure of the MCGPU-PET sample. Hot-spot
    positions in the actual sample are unknown from the documentation,
    so this test checks that (1) the water cube is correctly placed
    and (2) the format rules are followed byte-exactly for the regions
    we can fully specify.
    """

    def test_water_cube_voxel_present_with_background_activity(
        self, sample_scene_phantom, tmp_path: Path
    ):
        """The center voxel of the 5×5×5 cube must be water (id=2),
        density 1.0, with whatever activity we set.
        """
        source = Source.uniform_in_material(
            sample_scene_phantom,
            material="water",
            activity_per_voxel_Bq=50.0,
            isotope="F18",
        )
        path = tmp_path / "sample.vox"
        write_vox(sample_scene_phantom, source, path)
        content = path.read_text()
        # Water voxels should appear: "2 1.0 50.0"
        assert "2 1.0 50.0\n" in content
        # Air voxels should still appear: "1 0.0012 0.0"
        assert "1 0.0012 0.0\n" in content

    def test_line_counts_match_format_rules(
        self, sample_scene_phantom, zero_source_for_sample, tmp_path: Path
    ):
        """Total newlines = header (7) + body (nz × (ny × (nx + 1) + 1))
        + 1 trailing blank at EOF.
        """
        path = tmp_path / "sample.vox"
        write_vox(sample_scene_phantom, zero_source_for_sample, path)
        content = path.read_text()
        nx, ny, nz = 9, 9, 9
        expected_body_lines = nz * (ny * (nx + 1) + 1)  # 9 * (9*10 + 1) = 819
        expected_total_lines = 7 + expected_body_lines + 1  # +1 trailing blank
        actual_lines = content.count("\n")
        assert actual_lines == expected_total_lines

    def test_no_source_phantom_grid_mismatch(
        self, sample_scene_phantom, tmp_path: Path
    ):
        """write_vox must refuse mismatched grids — this is the safety
        net that prevents silent corruption.
        """
        other_phantom = Phantom.uniform(
            shape=(4, 4, 4), voxel_size=(1.0, 1.0, 1.0),
            material="water", density=1.0,
        )
        other_source = Source.zeros(other_phantom)
        with pytest.raises(ValueError, match="does not"):
            write_vox(sample_scene_phantom, other_source, tmp_path / "bad.vox")


@pytest.fixture
def zero_source_for_sample(sample_scene_phantom):
    return Source.zeros(sample_scene_phantom, isotope="F18")


# =====================================================================
# Format rules: structural checks
# =====================================================================


class TestVoxFormatRules:
    def test_x_fastest_iteration_order(self, tmp_path: Path):
        """Iteration must be X-fastest, then Y, then Z — this is what
        MCGPU-PET's reader expects. Use a 3×2×2 phantom with unique
        material IDs per voxel to check the traversal order directly.
        """
        nx, ny, nz = 3, 2, 2
        mat = np.zeros((nx, ny, nz), dtype=np.int32)
        # Label each voxel with an id encoding its (x, y, z) position.
        # Use ids 1..12 so "material_names" stays a tuple of length 12.
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    mat[x, y, z] = 1 + x + y * nx + z * nx * ny
        densities = np.ones((nx, ny, nz), dtype=np.float64)
        phantom = Phantom.from_numpy(
            material_ids=mat,
            densities=densities,
            voxel_size=(1.0, 1.0, 1.0),
            material_names=tuple(f"m{i}" for i in range(1, nx * ny * nz + 1)),
        )
        source = Source.zeros(phantom)
        path = tmp_path / "ordering.vox"
        write_vox(phantom, source, path)

        # Everything after "[END OF VXH SECTION]" up to EOF is the body.
        body = path.read_text().split("[END OF VXH SECTION]", 1)[1]
        # Voxel lines are those with exactly 3 whitespace-separated fields
        # where the first token is a small positive integer.
        voxel_lines = []
        for line in body.splitlines():
            toks = line.split()
            if len(toks) == 3 and toks[0].isdigit() and int(toks[0]) >= 1:
                voxel_lines.append(line)

        # Expected order: id=1 (x=0,y=0,z=0), id=2 (x=1,...), ..., id=12
        expected_ids = list(range(1, nx * ny * nz + 1))
        actual_ids = [int(l.split()[0]) for l in voxel_lines]
        assert actual_ids == expected_ids

    def test_blank_line_after_each_x_row(
        self, all_air_phantom, zero_source, tmp_path: Path
    ):
        """After each row of 9 voxels, there must be exactly one blank line."""
        path = tmp_path / "vox.vox"
        write_vox(all_air_phantom, zero_source, path)
        content = path.read_text()
        # Find one "end-of-X-row" pattern: 9 consecutive voxel lines followed
        # by a blank line.
        chunk = ("1 0.0012 0.0\n" * 9) + "\n"
        assert chunk in content

    def test_double_blank_between_z_slices(
        self, all_air_phantom, zero_source, tmp_path: Path
    ):
        """Between consecutive Z-slices there are two blank lines:
        end-of-X-row + end-of-Y-cycle.
        """
        path = tmp_path / "vox.vox"
        write_vox(all_air_phantom, zero_source, path)
        content = path.read_text()
        # A complete Z-slice ends with a voxel line, then "\n" (end of X),
        # then "\n" (end of Y), then the next slice begins with a voxel line.
        # Search for: ...0.0\n\n\n1 0.0012 0.0
        assert "0.0\n\n\n1 0.0012 0.0" in content


# =====================================================================
# write_in: structural tests
# =====================================================================


class TestWriteInStructural:
    @pytest.fixture
    def mcgpu_run(self, sample_scene_phantom):
        """A Run configured for the MCGPU sample simulation."""
        source = Source.with_total_activity(
            sample_scene_phantom, material="water",
            total_activity_Bq=1_000_000.0, isotope="F18",
        )
        scanner = Scanner.from_preset("mcgpu_sample")
        return Run(
            phantom=sample_scene_phantom,
            source=source,
            scanner=scanner,
            seed=12345,
        )

    @pytest.fixture
    def materials(self, tmp_path: Path):
        return MaterialRegistry(mcgpu_materials_dir=tmp_path / "materials")

    def test_all_sections_present(self, mcgpu_run, materials, tmp_path: Path):
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        for section in [
            "[SECTION SIMULATION CONFIG v.2016-07-05]",
            "[SECTION SOURCE PET SCAN v.2017-03-14]",
            "[SECTION PHASE SPACE FILE v.2016-07-05]",
            "[SECTION DOSE DEPOSITION v.2012-12-12]",
            "[SECTION ENERGY PARAMETERS v.2019-04-25]",
            "[SECTION SINOGRAM PARAMETERS v.2019-04-25]",
            "[SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]",
            "[SECTION MATERIAL FILE LIST v.2009-11-30]",
        ]:
            assert section in content

    def test_seed_from_run_is_written(self, mcgpu_run, materials, tmp_path: Path):
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        # Seed 12345 should appear as the first numeric value under SIMULATION CONFIG
        assert "12345" in content

    def test_none_seed_becomes_zero(self, sample_scene_phantom, materials,
                                     tmp_path: Path):
        """When Run.seed is None, MCGPU convention says write 0 so the
        simulator picks a time-based seed.
        """
        source = Source.zeros(sample_scene_phantom, isotope="F18")
        scanner = Scanner.from_preset("mcgpu_sample")
        run = Run(phantom=sample_scene_phantom, source=source, scanner=scanner)
        path = tmp_path / "MCGPU-PET.in"
        write_in(run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        # The first numeric line in SIMULATION CONFIG section should be "0"
        section_idx = content.index("[SECTION SIMULATION CONFIG v.2016-07-05]")
        first_nonblank = next(
            line.strip() for line in content[section_idx:].splitlines()[1:]
            if line.strip() and not line.strip().startswith("#")
        )
        assert first_nonblank.startswith("0 ") or first_nonblank.startswith("0\t") \
            or first_nonblank.split()[0] == "0"

    def test_sinogram_params_from_scanner(
        self, mcgpu_run, materials, tmp_path: Path
    ):
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        # mcgpu_sample preset values
        assert "147" in content   # radial bins
        assert "168" in content   # angular bins
        assert "159" in content   # z slices
        assert "80" in content    # n_rings
        assert "336" in content   # n_crystals_per_ring
        assert "79" in content    # MRD
        assert "11" in content    # span

    def test_energy_window_in_eV(
        self, mcgpu_run, materials, tmp_path: Path
    ):
        """Scanner stores keV but MCGPU-PET wants eV. Verify conversion
        happens at write time.
        """
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        # 350 keV → 350000 eV, 600 keV → 600000 eV
        assert "350000" in content
        assert "600000" in content

    def test_detector_radius_is_negated(
        self, mcgpu_run, materials, tmp_path: Path
    ):
        """Negated radius tells MCGPU-PET to auto-center on voxel geometry."""
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        assert "-9.05" in content

    def test_material_file_list_reflects_phantom_materials(
        self, mcgpu_run, materials, tmp_path: Path
    ):
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox", path=path)
        content = path.read_text()
        # Phantom's materials are ("air", "water") in that order (id 1, 2)
        assert "air_5-515keV.mcgpu.gz" in content
        assert "water_5-515keV.mcgpu.gz" in content

    def test_missing_n_rings_raises(
        self, sample_scene_phantom, materials, tmp_path: Path
    ):
        """A Scanner without n_rings cannot drive MCGPU-PET."""
        source = Source.zeros(sample_scene_phantom, isotope="F18")
        scanner = Scanner(
            name="minimal",
            detector_radius_cm=10.0, detector_axial_length_cm=10.0,
            energy_window_keV=(350.0, 650.0), energy_resolution=0.12,
            acquisition_time_s=1.0,
            n_radial_bins=128, n_angular_bins=128, n_z_slices=63,
            span=3, max_ring_difference=31,
            # n_rings and n_crystals_per_ring intentionally not set
        )
        run = Run(phantom=sample_scene_phantom, source=source, scanner=scanner)
        with pytest.raises(ValueError, match="n_rings"):
            write_in(run, materials, vox_filename="phantom.vox",
                     path=tmp_path / "bad.in")

    def test_unknown_material_in_phantom_raises(
        self, tmp_path: Path
    ):
        """A phantom referencing a material not in the registry must fail
        at write time with a clear error.
        """
        # Make a phantom with a fake material name
        mat = np.ones((2, 2, 2), dtype=np.int32)
        dens = np.full((2, 2, 2), 1.0, dtype=np.float64)
        phantom = Phantom.from_numpy(
            material_ids=mat,
            densities=dens,
            voxel_size=(1.0, 1.0, 1.0),
            material_names=("unobtanium",),
        )
        source = Source.zeros(phantom, isotope="F18")
        scanner = Scanner.from_preset("mcgpu_sample")
        run = Run(phantom=phantom, source=source, scanner=scanner)
        registry = MaterialRegistry(mcgpu_materials_dir=tmp_path / "materials")
        with pytest.raises(KeyError, match="unobtanium"):
            write_in(run, registry, vox_filename="phantom.vox",
                     path=tmp_path / "bad.in")

    def test_mcgpu_config_default_reproduces_sample_defaults(
        self, mcgpu_run, materials, tmp_path: Path
    ):
        """With a default MCGPUConfig, key reporting and tally defaults
        match the sample simulation's configuration.
        """
        path = tmp_path / "MCGPU-PET.in"
        write_in(mcgpu_run, materials, vox_filename="phantom.vox",
                 path=path, config=MCGPUConfig())
        content = path.read_text()
        # Default PSF size 150000000, image_resolution 128, n_energy_bins 700
        assert "150000000" in content
        assert "YES" in content  # material dose tally on by default
        assert "NO" in content   # voxel dose tally off by default


# =====================================================================
# Manual validation notes
# =====================================================================
# To verify byte-exact .vox reproduction against the real MCGPU-PET sample
# (which is not available in this test environment), run on a machine
# that has the MCGPU-PET repository:
#
#     python -c "
#     from petsim.phantom import Phantom
#     from petsim.source import Source
#     from petsim.backends.mcgpu import write_vox
#
#     phantom = Phantom.cube(
#         shape=(9,9,9), voxel_size=(1.0,1.0,1.0),
#         inner_material='water', inner_density=1.0,
#         outer_material='air', outer_density=0.0012,
#         inner_size_vox=5,
#     )
#     source = Source.uniform_in_material(
#         phantom, material='water', activity_per_voxel_Bq=50.0,
#     )
#     # Note: the sample also has 3 hot spots at unknown positions;
#     # exact reproduction requires locating them from the sample file.
#     write_vox(phantom, source, '/tmp/generated.vox')
#     "
#     diff /tmp/generated.vox path/to/MCGPU-PET/sample_simulation/phantom_9x9x9cm.vox
#
# Expected differences: only the voxel lines corresponding to the 3 hot
# spots (activity 1000, 2000, 3000 Bq). Once their positions are known,
# they can be added via Source.add_hot_spot() and the diff should be empty.