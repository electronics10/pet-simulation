"""
MCGPU-PET backend runtime: invoke the executable and parse outputs.

This module is the execution half of the MCGPU-PET backend. The input
half (write_vox, write_in) lives in mcgpu.py. Together they provide a
complete path: Run → input files → subprocess → output files → Sinogram.

Typical usage:

    backend = MCGPUBackend(executable="./MCGPU-PET.x")
    sinogram = backend.run(run, workdir="/tmp/my_sim")

The backend handles four things:

  1. Write .vox and .in into the workdir.
  2. Link/copy the materials/ directory the .in file references.
  3. Invoke MCGPU-PET.x and capture stdout/stderr + timing.
  4. Parse the binary output files into a Sinogram.

Counting convention: the number of detected coincidences is computed
by summing the parsed binary arrays, NEVER by parsing MCGPU-PET's stdout
log. The log format is fragile and has changed across versions; the
binary arrays are the ground truth.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..materials import MaterialRegistry
from ..run import Run
from ..sinogram import Sinogram
from .mcgpu import MCGPUConfig, write_in, write_vox


# =====================================================================
# Run result
# =====================================================================


@dataclass
class MCGPURunResult:
    """Everything an MCGPU-PET invocation produced, beyond the sinogram.

    The Sinogram is saved separately through the Run machinery. This
    dataclass captures what was needed to produce it, for provenance.
    """

    workdir: Path
    returncode: int
    wall_time_s: float
    stdout_path: Path
    stderr_path: Path
    # Absolute paths to the raw output files MCGPU-PET emitted
    output_files: dict[str, Path] = field(default_factory=dict)


# =====================================================================
# Backend
# =====================================================================


class MCGPUBackend:
    """Driver for the MCGPU-PET command-line simulator.

    The backend is stateless except for the path to the executable and
    the materials directory; it's safe to reuse one instance across
    many runs in the same process.
    """

    def __init__(
        self,
        executable: str | Path = "./MCGPU-PET.x",
        materials_dir: Optional[str | Path] = None,
        materials_registry: Optional[MaterialRegistry] = None,
    ) -> None:
        """
        Parameters
        ----------
        executable : path to the MCGPU-PET.x binary. Used as-is (you
            can pass an absolute path, or a path relative to the workdir
            — MCGPU-PET is typically invoked as ./MCGPU-PET.x from its
            own build directory).
        materials_dir : directory containing the .mcgpu.gz material files.
            This directory will be symlinked as "materials/" inside each
            workdir, since the .in file uses the relative path
            "./materials/...".
        materials_registry : a MaterialRegistry used to resolve material
            names in the Phantom to the .mcgpu.gz filenames. If not
            provided, a default registry is used that points to
            materials_dir.
        """
        self.executable = Path(executable).resolve() \
            if Path(executable).is_absolute() or Path(executable).exists() \
            else Path(executable)
        self.materials_dir = Path(materials_dir).resolve() \
            if materials_dir is not None else None
        if materials_registry is not None:
            self.materials_registry = materials_registry
        elif self.materials_dir is not None:
            self.materials_registry = MaterialRegistry(
                mcgpu_materials_dir=self.materials_dir,
            )
        else:
            # Caller must provide one or the other before calling
            # stage_inputs. Defer the error until that point so __init__
            # stays cheap.
            self.materials_registry = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Input staging
    # ------------------------------------------------------------------

    def stage_inputs(
        self,
        run: Run,
        workdir: str | Path,
        config: Optional[MCGPUConfig] = None,
        petsim_tag: str = "petsim",
    ) -> Path:
        """Write .vox, .in, and link materials/ into workdir.

        Returns the absolute workdir path.
        """
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        if self.materials_registry is None:
            raise ValueError(
                "MCGPUBackend has no MaterialRegistry; pass either "
                "`materials_dir` or `materials_registry` to the constructor."
            )

        vox_filename = "phantom.vox"
        write_vox(run.phantom, run.source, workdir / vox_filename)
        write_in(
            run=run,
            materials=self.materials_registry,
            vox_filename=vox_filename,
            config=config,
            path=workdir / "MCGPU-PET.in",
            petsim_tag=petsim_tag,
        )

        if self.materials_dir is not None:
            self._link_materials(workdir)

        return workdir

    def _link_materials(self, workdir: Path) -> None:
        """Make the materials/ directory available inside workdir.

        MCGPU-PET's .in file references materials as "./materials/foo.mcgpu.gz",
        so the workdir needs a materials/ entry. Symlink is preferred
        (cheap, no copy); fall back to copying if symlinks aren't
        supported.
        """
        target = workdir / "materials"
        if target.exists() or target.is_symlink():
            return
        try:
            target.symlink_to(self.materials_dir)
        except (OSError, NotImplementedError):
            shutil.copytree(self.materials_dir, target)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def invoke(
        self,
        workdir: str | Path,
        timeout_s: Optional[float] = None,
    ) -> MCGPURunResult:
        """Run MCGPU-PET.x in workdir and capture stdout/stderr.

        Returns a MCGPURunResult. Does NOT raise on nonzero exit — the
        caller decides how to handle failure (e.g. a sanity-check run
        might tolerate errors while an ML-training run would not).
        """
        workdir = Path(workdir).resolve()
        stdout_path = workdir / "MCGPU-PET.out"
        stderr_path = workdir / "MCGPU-PET.err"

        t0 = time.perf_counter()
        with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
            proc = subprocess.run(
                [str(self.executable), "MCGPU-PET.in"],
                cwd=str(workdir),
                stdout=out_f,
                stderr=err_f,
                timeout=timeout_s,
                check=False,
            )
        wall_time = time.perf_counter() - t0

        # Enumerate output files MCGPU-PET produced (anything matching
        # its known output patterns)
        output_files: dict[str, Path] = {}
        for pattern in ["*.raw", "*.raw.gz", "*.psf", "*.psf.gz",
                        "*.dat", "image_*", "sinogram_*"]:
            for p in workdir.glob(pattern):
                output_files[p.name] = p

        return MCGPURunResult(
            workdir=workdir,
            returncode=proc.returncode,
            wall_time_s=wall_time,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output_files=output_files,
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def parse_sinogram(
        self,
        run: Run,
        invoke_result: MCGPURunResult,
    ) -> Sinogram:
        """Parse MCGPU-PET's sinogram output files into a Sinogram object.

        MCGPU-PET writes binary int32 arrays for trues and scatter.
        Auto-detects the z-dimension from file size since span compression
        can produce different output sizes than the input parameter.
        """
        workdir = invoke_result.workdir
        n_angular = run.scanner.n_angular_bins
        n_radial = run.scanner.n_radial_bins

        trues = self._read_sinogram_file(
            candidate_names=[
                "sinogram_Trues.raw", "sinogram_Trues.raw.gz",
                "sino_Trues.raw", "sino_Trues.raw.gz",
            ],
            workdir=workdir,
            n_angular=n_angular,
            n_radial=n_radial,
        )
        scatter = self._read_sinogram_file(
            candidate_names=[
                "sinogram_Scatter.raw", "sinogram_Scatter.raw.gz",
                "sino_Scatter.raw", "sino_Scatter.raw.gz",
            ],
            workdir=workdir,
            n_angular=n_angular,
            n_radial=n_radial,
            optional=True,
        )

        sinogram_shape = trues.shape
        return Sinogram(
            trues=trues,
            scatter=scatter,
            shape=sinogram_shape,
            metadata={
                "backend": "mcgpu",
                "wall_time_s": invoke_result.wall_time_s,
                "returncode": invoke_result.returncode,
            },
        )

    @staticmethod
    def _read_sinogram_file(
        candidate_names: list[str],
        workdir: Path,
        n_angular: int,
        n_radial: int,
        optional: bool = False,
    ) -> Optional[np.ndarray]:
        """Read a sinogram binary file, auto-detecting .gz compression.

        Auto-detects the z-dimension from file size. Returns an int32
        ndarray of shape (n_z, n_angular, n_radial) where n_z is inferred
        from the file size.
        """
        path = None
        for name in candidate_names:
            candidate = workdir / name
            if candidate.exists():
                path = candidate
                break
        if path is None:
            if optional:
                return None
            raise FileNotFoundError(
                f"No sinogram file found in {workdir}; "
                f"tried {candidate_names}"
            )

        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                raw = f.read()
        else:
            raw = path.read_bytes()

        arr = np.frombuffer(raw, dtype=np.int32)

        # Auto-detect n_z from file size
        if arr.size % (n_angular * n_radial) != 0:
            raise ValueError(
                f"Sinogram file {path.name} has {arr.size} int32 elements, "
                f"which is not divisible by n_angular*n_radial "
                f"({n_angular}*{n_radial}={n_angular*n_radial}). "
                f"The file may be corrupt or the scanner binning parameters are wrong."
            )

        n_z = arr.size // (n_angular * n_radial)
        return arr.reshape((n_z, n_angular, n_radial)).copy()

    # ------------------------------------------------------------------
    # Convenience: full pipeline in one call
    # ------------------------------------------------------------------

    def run_full(
        self,
        run: Run,
        workdir: str | Path,
        config: Optional[MCGPUConfig] = None,
        timeout_s: Optional[float] = None,
        petsim_tag: str = "petsim",
    ) -> tuple[Sinogram, MCGPURunResult]:
        """Stage inputs, invoke the binary, parse outputs.

        This is the one-shot convenience method. The three-step API
        (stage_inputs, invoke, parse_sinogram) remains available for
        callers who want to inspect intermediate state.
        """
        workdir = self.stage_inputs(run, workdir, config=config,
                                    petsim_tag=petsim_tag)
        invoke_result = self.invoke(workdir, timeout_s=timeout_s)
        if invoke_result.returncode != 0:
            raise RuntimeError(
                f"MCGPU-PET exited with code {invoke_result.returncode}; "
                f"see {invoke_result.stdout_path} and {invoke_result.stderr_path}"
            )
        sinogram = self.parse_sinogram(run, invoke_result)
        return sinogram, invoke_result