"""
Run: a complete simulation on disk.

A Run represents one end-to-end simulation bundle:

    x (ground truth) = Phantom + Source
    A (forward model) = Scanner
    y (measurement)   = Sinogram

All four components plus a manifest live in a single directory following
the storage format from PLAN.md:

    runs/001_water_cylinder/
    ├── run.yaml          ← manifest: backend, seed, git hash, wall time, counts
    ├── phantom.npz       ← x (geometry + materials)
    ├── source.npz        ← x (activity distribution)
    ├── scanner.yaml      ← A (forward model)
    ├── sinogram.npz      ← y (measurement, backend-agnostic format)
    └── simulator_output/ ← raw backend-specific outputs (not loaded by Run)

The `Run` class is the top-level API for saving / loading an entire
simulation. Backends (Phase 2+) produce `Run` objects; downstream code
(analysis, DL training) consumes them.

This module has no backend-specific logic — backends call `Run.save` after
they finish running the simulator.
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


# File names inside a run directory. Kept as module-level constants so
# backends and tests can reference them consistently.
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
        Forward model — scanner geometry and binning.
    sinogram : Sinogram
        Measurement — coincidence histograms. None if the simulation has
        not been run yet (a prepared-but-unrun bundle).
    seed : int | None
        RNG seed used for this run. First-class field because
        reproducibility is critical for ML: the same seed must produce
        statistically identical output, and changing the seed is the
        standard way to generate noise-pair datasets. Backends are
        responsible for actually feeding this seed to the simulator.
    metadata : dict
        Free-form manifest fields. Backends populate:
          - backend: "mcgpu" | "gate"
          - wall_time_seconds: float
          - git_hash: str
          - created_at: ISO 8601 string
          - simulator_version: str
          - plus anything the backend wants to record
    """

    phantom: Phantom
    source: Source
    scanner: Scanner
    sinogram: Sinogram | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        if not self.source.matches(self.phantom):
            raise ValueError(
                f"source grid {self.source.shape} {self.source.voxel_size} cm "
                f"does not match phantom grid {self.phantom.shape} "
                f"{self.phantom.voxel_size} cm"
            )
        if self.sinogram is not None and self.sinogram.scanner != self.scanner:
            raise ValueError(
                "sinogram.scanner does not match run.scanner; "
                "did you accidentally pass a mismatched scanner?"
            )

    # ---- persistence --------------------------------------------------

    def save(self, run_dir: str | Path) -> None:
        """Write the entire run to a directory following the storage format.

        Creates `run_dir` if it doesn't exist. Does not touch
        `simulator_output/` — backends write that themselves during or
        before calling save().

        If `sinogram` is None, no sinogram.npz file is written, and the
        manifest records `has_sinogram: false`.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self.phantom.save(run_dir / PHANTOM_FILENAME)
        self.source.save(run_dir / SOURCE_FILENAME)
        self.scanner.save(run_dir / SCANNER_FILENAME)
        if self.sinogram is not None:
            self.sinogram.save(run_dir / SINOGRAM_FILENAME)

        # Build the manifest. Always populate created_at if not provided.
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
        """Load a complete run from a directory produced by save().

        The manifest's has_sinogram flag determines whether sinogram.npz
        is read. If the file is missing but has_sinogram is true, an
        error is raised.
        """
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
            sinogram = Sinogram.load(sino_path, scanner=scanner)

        # Strip bookkeeping fields from the user-facing metadata.
        # Seed is promoted to a first-class field on Run.
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

    # ---- equality / repr ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Run):
            return NotImplemented

        # Compare metadata without the auto-populated bookkeeping fields,
        # so two Runs that were saved at different times can still be
        # considered equal if their actual content matches.
        def strip(d: dict) -> dict:
            return {
                k: v for k, v in d.items()
                if k not in ("created_at", "has_sinogram", "files")
            }

        return (
            self.phantom == other.phantom
            and self.source == other.source
            and self.scanner == other.scanner
            and self.sinogram == other.sinogram
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