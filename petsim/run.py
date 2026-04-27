"""
Run: a complete simulation on disk.

A Run represents one end-to-end simulation bundle:

    x (ground truth) = Phantom + Source
    A (forward model) = Scanner
    y (measurement)   = Sinogram

All four components plus a manifest live in a single directory:

    runs/001_water_cylinder/
    ├── run.yaml          ← manifest: backend, seed, git hash, wall time, counts
    ├── phantom.npz       ← x (geometry + materials)
    ├── source.npz        ← x (activity distribution)
    ├── scanner.yaml      ← A (forward model — hardware spec)
    ├── sinogram.npz      ← y (measurement)
    └── simulator_output/ ← raw backend-specific outputs (not loaded by Run)
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
from .source import Source


MANIFEST_FILENAME = "run.yaml"
PHANTOM_FILENAME = "phantom.npz"
SOURCE_FILENAME = "source.npz"
SCANNER_FILENAME = "scanner.yaml"
SINOGRAM_FILENAME = "sinogram.npz"
SIMULATOR_OUTPUT_DIRNAME = "simulator_output"


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
        Hardware spec — scanner geometry, energy window, etc.
    sinogram : Sinogram | None
        Measurement. None if not yet simulated.
    seed : int | None
        RNG seed for reproducibility.
    metadata : dict
        Free-form manifest fields (backend, wall_time_seconds, git_hash, ...).
    """

    phantom: Phantom
    source: Source
    scanner: Scanner
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

    # ---- persistence --------------------------------------------------

    def save(self, run_dir: str | Path) -> None:
        """Write the entire run to a directory."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self.phantom.save(run_dir / PHANTOM_FILENAME)
        self.source.save(run_dir / SOURCE_FILENAME)
        self.scanner.save(run_dir / SCANNER_FILENAME)
        if self.sinogram is not None:
            self.sinogram.save(run_dir / SINOGRAM_FILENAME)

        manifest = dict(self.metadata)
        manifest.setdefault("created_at", datetime.now().isoformat())
        manifest["seed"] = self.seed
        manifest["has_sinogram"] = self.sinogram is not None
        manifest["files"] = {
            "phantom": PHANTOM_FILENAME,
            "source": SOURCE_FILENAME,
            "scanner": SCANNER_FILENAME,
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

        sinogram: Sinogram | None = None
        has_sinogram = manifest.get("has_sinogram", False)
        if has_sinogram:
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
            if k not in ("has_sinogram", "files", "seed")
        }

        return cls(
            phantom=phantom,
            source=source,
            scanner=scanner,
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
                if k not in ("created_at", "has_sinogram", "files")
            }

        return (
            self.phantom == other.phantom
            and self.source == other.source
            and self.scanner == other.scanner
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
            f"{sino_part})"
        )