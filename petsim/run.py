"""
Run: a complete simulation on disk.

A Run represents one end-to-end simulation bundle:

    x (ground truth) = Phantom + Source
    A (forward model) = Scanner + SinogramBinning
    y (measurement)   = Sinogram

Six files per run directory:

    runs/0001/
    ├── run.yaml          ← manifest: seed, backend, runtime config, timestamps
    ├── phantom.npz       ← x (geometry + materials)
    ├── source.npz        ← x (activity distribution)
    ├── scanner.yaml      ← A (hardware spec)
    ├── binning.yaml      ← A (sinogram layout choice)
    └── sinogram.npz      ← y (measurement)

The manifest preserves backend-specific runtime config (MCGPUConfig etc.)
so runs are exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .phantom import Phantom
from .scanner import Scanner
from .sinogram import Sinogram
from .sinogram_binning import SinogramBinning
from .source import Source


MANIFEST_FILENAME = "run.yaml"
PHANTOM_FILENAME = "phantom.npz"
SOURCE_FILENAME = "source.npz"
SCANNER_FILENAME = "scanner.yaml"
BINNING_FILENAME = "binning.yaml"
SINOGRAM_FILENAME = "sinogram.npz"


@dataclass
class Run:
    """A complete PET simulation bundle.

    Attributes
    ----------
    phantom : Phantom
        Ground truth geometry and materials.
    source : Source
        Ground truth radioactivity distribution.
    scanner : Scanner
        Hardware spec — backend-agnostic.
    binning : SinogramBinning | None
        Sinogram layout choice — backend-agnostic. Required to run
        any backend; can be None only for partially-built bundles.
        Use SinogramBinning.default_for(scanner) for sensible defaults.
    sinogram : Sinogram | None
        Measurement. None if not yet simulated.
    seed : int | None
        RNG seed for reproducibility.
    metadata : dict
        Backend-specific runtime config and bookkeeping. Backends
        populate keys like 'backend', 'mcgpu_config', 'gate_config',
        'wall_time_s', 'created_at'.
    """

    phantom: Phantom
    source: Source
    scanner: Scanner
    binning: SinogramBinning | None = None
    sinogram: Sinogram | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.matches(self.phantom):
            raise ValueError(
                f"source grid {self.source.shape} {self.source.voxel_size} cm "
                f"does not match phantom grid {self.phantom.shape} "
                f"{self.phantom.voxel_size} cm"
            )
        if self.binning is not None:
            self.binning.validate(self.scanner)

    # ---- persistence --------------------------------------------------

    def save(self, run_dir: str | Path) -> None:
        """Write the entire run to a directory.

        Saves up to six files. binning.yaml is written if binning is set;
        sinogram.npz is written if a sinogram exists.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self.phantom.save(run_dir / PHANTOM_FILENAME)
        self.source.save(run_dir / SOURCE_FILENAME)
        self.scanner.save(run_dir / SCANNER_FILENAME)
        if self.binning is not None:
            self.binning.save(run_dir / BINNING_FILENAME)
        if self.sinogram is not None:
            self.sinogram.save(run_dir / SINOGRAM_FILENAME)

        manifest = dict(self.metadata)
        manifest.setdefault("created_at", datetime.now().isoformat())
        manifest["seed"] = self.seed
        manifest["has_binning"] = self.binning is not None
        manifest["has_sinogram"] = self.sinogram is not None
        manifest["files"] = {
            "phantom": PHANTOM_FILENAME,
            "source": SOURCE_FILENAME,
            "scanner": SCANNER_FILENAME,
            "binning": BINNING_FILENAME if self.binning is not None else None,
            "sinogram": SINOGRAM_FILENAME if self.sinogram is not None else None,
        }

        with open(run_dir / MANIFEST_FILENAME, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load(cls, run_dir: str | Path) -> "Run":
        """Load a complete run from a directory produced by save()."""
        run_dir = Path(run_dir)
        manifest_path = run_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_FILENAME} in {run_dir}; is this a valid run directory?"
            )

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        phantom = Phantom.load(run_dir / PHANTOM_FILENAME)
        source = Source.load(run_dir / SOURCE_FILENAME)
        scanner = Scanner.load(run_dir / SCANNER_FILENAME)

        binning: SinogramBinning | None = None
        if manifest.get("has_binning", False):
            binning = SinogramBinning.load(run_dir / BINNING_FILENAME)

        sinogram: Sinogram | None = None
        if manifest.get("has_sinogram", False):
            sino_path = run_dir / SINOGRAM_FILENAME
            if not sino_path.exists():
                raise FileNotFoundError(
                    f"manifest claims sinogram present but {SINOGRAM_FILENAME} "
                    f"is missing in {run_dir}"
                )
            sinogram = Sinogram.load(sino_path)

        seed = manifest.get("seed", None)
        metadata = {
            k: v for k, v in manifest.items()
            if k not in ("has_binning", "has_sinogram", "files")
        }

        return cls(
            phantom=phantom,
            source=source,
            scanner=scanner,
            binning=binning,
            sinogram=sinogram,
            seed=seed,
            metadata=metadata,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Run):
            return NotImplemented

        def strip(d: dict) -> dict:
            return {
                k: v for k, v in d.items()
                if k not in ("created_at",)
            }

        return (
            self.phantom == other.phantom
            and self.source == other.source
            and self.scanner == other.scanner
            and self.binning == other.binning
            and self.seed == other.seed
            and strip(self.metadata) == strip(other.metadata)
        )

    def __repr__(self) -> str:
        sino_part = (
            f"sinogram={self.sinogram!r}" if self.sinogram is not None
            else "sinogram=None"
        )
        return (
            f"Run(phantom={self.phantom.shape}, "
            f"source={self.source.isotope} {self.source.total_activity_Bq:.3g} Bq, "
            f"scanner={self.scanner.name!r}, "
            f"binning={self.binning!r}, "
            f"{sino_part})"
        )