"""
Tests for the MCGPU-PET execution layer (backends/mcgpu_runtime.py).

These tests do NOT invoke the real MCGPU-PET.x binary — that requires
CUDA and is covered by a manual end-to-end validation step on the
simulation server. Instead, they use a stub "executable" (a shell
script) that produces plausible output files so we can exercise the
staging, invocation, and parsing machinery in isolation.

The stub approach gives us confidence that:
  1. stage_inputs writes a valid .vox and .in into the workdir.
  2. The subprocess plumbing runs the right command in the right cwd.
  3. parse_sinogram correctly reads the binary output files.

It does NOT validate that MCGPU-PET itself produces correct results —
that's what byte-exact .vox reproduction + real-run diffs are for.

Run with:
    uv run pytest tests/test_mcgpu_runtime.py -v
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from petsim.backends.mcgpu import MCGPUConfig
from petsim.backends.mcgpu_runtime import MCGPUBackend, MCGPURunResult
from petsim.materials import MaterialRegistry
from petsim.phantom import Phantom
from petsim.run import Run
from petsim.scanner import Scanner
from petsim.sinogram import Sinogram
from petsim.source import Source


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def basic_run():
    """A small but fully-specified Run suitable for driving the backend."""
    phantom = Phantom.cube(
        shape=(9, 9, 9),
        voxel_size=(1.0, 1.0, 1.0),
        inner_material="water",
        inner_density=1.0,
        outer_material="air",
        outer_density=0.0012,
        inner_size_vox=5,
    )
    source = Source.with_total_activity(
        phantom, material="water",
        total_activity_Bq=1e6, isotope="F18",
    )
    scanner = Scanner.from_preset("mcgpu_sample")
    return Run(phantom=phantom, source=source, scanner=scanner, seed=42)


@pytest.fixture
def fake_materials_dir(tmp_path: Path) -> Path:
    """A materials/ directory with empty placeholder .mcgpu.gz files.

    Just enough for the stager to find files by name; the stub executable
    doesn't actually read them.
    """
    mdir = tmp_path / "materials_root"
    mdir.mkdir()
    for filename in ["air_5-515keV.mcgpu.gz", "water_5-515keV.mcgpu.gz"]:
        (mdir / filename).write_bytes(b"")
    return mdir


@pytest.fixture
def stub_executable(tmp_path: Path, basic_run) -> Path:
    """A shell script that pretends to be MCGPU-PET.x.

    It writes a sinogram_Trues.raw file with the expected shape and
    content, plus a sinogram_Scatter.raw at 10% of trues. This lets the
    parsing path be exercised without CUDA.
    """
    shape = basic_run.scanner.sinogram_shape
    n = int(np.prod(shape))
    # Deterministic counts: each bin gets its flat index (mod 7) so the
    # total is reproducible and nonzero.
    stub_path = tmp_path / "stub_mcgpu.sh"
    script = f"""#!/usr/bin/env bash
# Stub MCGPU-PET replacement for tests.
set -e
python3 - <<'PY'
import numpy as np
trues = (np.arange({n}, dtype=np.int32) % 7).reshape{shape}
scatter = (np.arange({n}, dtype=np.int32) % 3).reshape{shape}
trues.tofile("sinogram_Trues.raw")
scatter.tofile("sinogram_Scatter.raw")
print("Stub MCGPU-PET done")
PY
"""
    stub_path.write_text(script)
    stub_path.chmod(stub_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    return stub_path


@pytest.fixture
def backend(stub_executable, fake_materials_dir) -> MCGPUBackend:
    return MCGPUBackend(
        executable=stub_executable,
        materials_dir=fake_materials_dir,
    )


# =====================================================================
# stage_inputs
# =====================================================================


class TestStageInputs:
    def test_writes_vox_and_in(self, backend, basic_run, tmp_path: Path):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        assert (workdir / "phantom.vox").exists()
        assert (workdir / "MCGPU-PET.in").exists()

    def test_creates_workdir_if_missing(self, backend, basic_run, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        workdir = backend.stage_inputs(basic_run, nested)
        assert workdir.is_dir()
        assert (workdir / "MCGPU-PET.in").exists()

    def test_links_materials_directory(self, backend, basic_run, tmp_path: Path):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        materials_link = workdir / "materials"
        assert materials_link.exists()
        # Either symlink or copy, but must contain the expected files
        assert (materials_link / "air_5-515keV.mcgpu.gz").exists()
        assert (materials_link / "water_5-515keV.mcgpu.gz").exists()

    def test_seed_propagates_to_in_file(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        content = (workdir / "MCGPU-PET.in").read_text()
        assert "42" in content

    def test_config_is_applied(self, backend, basic_run, tmp_path: Path):
        """A non-default MCGPUConfig must be reflected in the .in file."""
        config = MCGPUConfig(gpu_number=3, threads_per_block=64)
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1",
                                       config=config)
        content = (workdir / "MCGPU-PET.in").read_text()
        # gpu_number appears as "3" on the second numeric line of the
        # SIMULATION CONFIG section; threads_per_block as "64" on the third.
        section = content.split("[SECTION SIMULATION CONFIG")[1]
        section = section.split("[SECTION SOURCE")[0]
        # Extract numeric lines
        numeric_lines = [
            line.split()[0] for line in section.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # The first "numeric" line is actually a leftover from the header
        # split (e.g. "v.2016-07-05]"). Values follow: seed, gpu, threads, density.
        assert numeric_lines[1] == "42"  # seed
        assert numeric_lines[2] == "3"   # gpu_number
        assert numeric_lines[3] == "64"  # threads_per_block

    def test_registry_required(self, basic_run, tmp_path: Path, stub_executable):
        """Without materials_dir or materials_registry, staging must fail
        with a clear error before any files are written.
        """
        bare = MCGPUBackend(executable=stub_executable)
        with pytest.raises(ValueError, match="MaterialRegistry"):
            bare.stage_inputs(basic_run, tmp_path / "run1")


# =====================================================================
# invoke
# =====================================================================


class TestInvoke:
    def test_returncode_zero_on_success(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        assert result.returncode == 0

    def test_captures_stdout_to_file(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        assert result.stdout_path.exists()
        assert "Stub MCGPU-PET done" in result.stdout_path.read_text()

    def test_wall_time_recorded(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        assert result.wall_time_s > 0
        assert result.wall_time_s < 60  # stub is fast

    def test_enumerates_output_files(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        # Stub writes sinogram_Trues.raw and sinogram_Scatter.raw
        assert "sinogram_Trues.raw" in result.output_files
        assert "sinogram_Scatter.raw" in result.output_files

    def test_nonzero_returncode_does_not_raise(
        self, basic_run, tmp_path: Path, fake_materials_dir
    ):
        """invoke() must tolerate failures — caller decides."""
        failing = tmp_path / "failing.sh"
        failing.write_text("#!/usr/bin/env bash\nexit 7\n")
        failing.chmod(failing.stat().st_mode | stat.S_IEXEC)
        backend = MCGPUBackend(
            executable=failing,
            materials_dir=fake_materials_dir,
        )
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        assert result.returncode == 7


# =====================================================================
# parse_sinogram
# =====================================================================


class TestParseSinogram:
    def test_parses_stub_output(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        sinogram = backend.parse_sinogram(basic_run, result)

        assert isinstance(sinogram, Sinogram)
        assert sinogram.trues is not None
        assert sinogram.scatter is not None
        assert sinogram.trues.shape == basic_run.scanner.sinogram_shape

    def test_deterministic_counts(
        self, backend, basic_run, tmp_path: Path
    ):
        """The stub writes counts = (flat_index % 7). Verify that's what
        we read back.
        """
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        sinogram = backend.parse_sinogram(basic_run, result)

        n = int(np.prod(basic_run.scanner.sinogram_shape))
        expected = (np.arange(n, dtype=np.int32) % 7).reshape(
            basic_run.scanner.sinogram_shape
        )
        np.testing.assert_array_equal(sinogram.trues.astype(np.int32),
                                      expected)

    def test_metadata_populated(
        self, backend, basic_run, tmp_path: Path
    ):
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        sinogram = backend.parse_sinogram(basic_run, result)
        assert sinogram.metadata["backend"] == "mcgpu"
        assert sinogram.metadata["returncode"] == 0
        assert "wall_time_s" in sinogram.metadata

    def test_missing_trues_raises(
        self, backend, basic_run, tmp_path: Path
    ):
        """If the stub didn't write sinogram_Trues, parsing must fail."""
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        # Delete the trues file
        (workdir / "sinogram_Trues.raw").unlink()
        result.output_files.pop("sinogram_Trues.raw", None)
        with pytest.raises(FileNotFoundError, match="sinogram"):
            backend.parse_sinogram(basic_run, result)

    def test_missing_scatter_is_ok(
        self, backend, basic_run, tmp_path: Path
    ):
        """Scatter is optional — some configs don't tally it."""
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        (workdir / "sinogram_Scatter.raw").unlink()
        sinogram = backend.parse_sinogram(basic_run, result)
        assert sinogram.scatter is None
        assert sinogram.trues is not None

    def test_shape_mismatch_raises_with_hint(
        self, backend, basic_run, tmp_path: Path
    ):
        """If the binary file has a different size than Scanner expects,
        the error message must suggest what the actual z-dimension is.
        """
        workdir = backend.stage_inputs(basic_run, tmp_path / "run1")
        result = backend.invoke(workdir)
        # Overwrite with wrong size: half the expected z-extent
        nx = basic_run.scanner.n_radial_bins
        ny = basic_run.scanner.n_angular_bins
        half_z = basic_run.scanner.n_z_slices // 2
        wrong = np.ones((half_z, ny, nx), dtype=np.int32)
        wrong.tofile(workdir / "sinogram_Trues.raw")
        with pytest.raises(ValueError, match="n_z_slices"):
            backend.parse_sinogram(basic_run, result)


# =====================================================================
# run_full (end-to-end)
# =====================================================================


class TestRunFull:
    def test_full_pipeline(self, backend, basic_run, tmp_path: Path):
        sinogram, result = backend.run_full(basic_run, tmp_path / "run1")
        assert sinogram.trues is not None
        assert result.returncode == 0

    def test_full_pipeline_raises_on_failure(
        self, basic_run, tmp_path: Path, fake_materials_dir
    ):
        failing = tmp_path / "failing.sh"
        failing.write_text("#!/usr/bin/env bash\nexit 1\n")
        failing.chmod(failing.stat().st_mode | stat.S_IEXEC)
        backend = MCGPUBackend(
            executable=failing,
            materials_dir=fake_materials_dir,
        )
        with pytest.raises(RuntimeError, match="exit"):
            backend.run_full(basic_run, tmp_path / "run1")