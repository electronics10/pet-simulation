"""
MCGPU-PET backend runtime: invoke the executable and parse outputs.

Typical usage:

    backend = MCGPUBackend(executable="./bin/MCGPU-PET.x",
                           materials_dir="./materials")
    sinogram, result = backend.run_full(run, run_dir="./runs/0001")

The Run must carry both a Scanner (hardware) and a SinogramBinning
(layout choice). Use SinogramBinning.default_for(scanner) if unsure.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..materials import MaterialRegistry
from ..run import Run
from ..sinogram import Sinogram
from .mcgpu import MCGPUConfig, write_in, write_vox


@dataclass
class MCGPURunResult:
    """Everything an MCGPU-PET invocation produced."""

    workdir: Path
    returncode: int
    wall_time_s: float
    stdout_path: Path
    stderr_path: Path
    output_files: dict[str, Path] = field(default_factory=dict)


class MCGPUBackend:
    """Driver for the MCGPU-PET command-line simulator."""

    def __init__(
        self,
        executable: str | Path = "./bin/MCGPU-PET.x",
        materials_dir: Optional[str | Path] = None,
        materials_registry: Optional[MaterialRegistry] = None,
    ) -> None:
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
        """Write .vox, .in, and link materials/ into workdir."""
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        if self.materials_registry is None:
            raise ValueError(
                "MCGPUBackend has no MaterialRegistry; pass either "
                "`materials_dir` or `materials_registry` to the constructor."
            )

        # Clean stale outputs
        for pattern in ["*.raw", "*.raw.gz", "*.psf", "*.psf.gz",
                        "*.dat", "image_*", "sinogram_*",
                        "MCGPU-PET.in", "phantom.vox",
                        "MCGPU-PET.out", "MCGPU-PET.err"]:
            for stale in workdir.glob(pattern):
                if stale.is_file() or stale.is_symlink():
                    stale.unlink()

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
        """Parse MCGPU-PET's sinogram output into a Sinogram object.

        Reads n_angular_bins and n_radial_bins from run.binning.
        Auto-detects n_z from file size.
        """
        if run.binning is None:
            raise ValueError(
                "Run.binning is None — cannot parse sinogram without "
                "knowing the bin counts."
            )

        workdir = invoke_result.workdir
        n_angular = run.binning.n_angular_bins
        n_radial = run.binning.n_radial_bins

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

        return Sinogram(
            trues=trues,
            scatter=scatter,
            shape=trues.shape,
            metadata={
                "backend": "mcgpu",
                "wall_time_s": invoke_result.wall_time_s,
                "returncode": invoke_result.returncode,
                "scanner": run.scanner.name,
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

        if arr.size % (n_angular * n_radial) != 0:
            raise ValueError(
                f"Sinogram file {path.name} has {arr.size} int32 elements, "
                f"which is not divisible by n_angular*n_radial "
                f"({n_angular}*{n_radial}={n_angular*n_radial})."
            )

        n_z = arr.size // (n_angular * n_radial)
        return arr.reshape((n_z, n_angular, n_radial)).copy()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full(
        self,
        run: Run,
        run_dir: str | Path,
        config: Optional[MCGPUConfig] = None,
        workdir: Optional[str | Path] = None,
        timeout_s: Optional[float] = None,
        petsim_tag: str = "petsim",
        keep_workdir: bool = False,
    ) -> tuple[Sinogram, MCGPURunResult]:
        """Stage, run, parse, save — the complete pipeline.

        Saves the 5 essential files to run_dir on success and cleans
        up the temp workdir. On crash, the workdir is preserved for
        debugging.
        """
        if config is None:
            config = MCGPUConfig()

        run_dir = Path(run_dir).resolve()
        if workdir is None:
            workdir = run_dir / "tmp"
        workdir = Path(workdir).resolve()

        # Stage inputs
        self.stage_inputs(run, workdir, config=config, petsim_tag=petsim_tag)

        # Run
        invoke_result = self.invoke(workdir, timeout_s=timeout_s)

        # On crash, leave workdir for debugging
        if invoke_result.returncode != 0:
            raise RuntimeError(
                f"MCGPU-PET exited with code {invoke_result.returncode}; "
                f"inspect {workdir} for input files and logs"
            )

        # Parse
        sinogram = self.parse_sinogram(run, invoke_result)

        # Attach to run, record metadata, save
        run.sinogram = sinogram
        run.metadata["backend"] = "mcgpu"
        run.metadata["mcgpu_config"] = asdict(config)
        run.metadata["wall_time_s"] = invoke_result.wall_time_s
        run.save(run_dir)

        # Clean up on success
        if not keep_workdir and workdir.exists():
            shutil.rmtree(workdir)

        return sinogram, invoke_result