"""
Run: a complete simulation on disk.

A `Run` represents one executed simulation in the standard storage layout:

    runs/001_water_cylinder/
    ├── run.yaml                ← manifest: backend, seed, git hash, wall time
    ├── phantom.npz             ← x: material_ids, densities, voxel_size
    ├── source.npz              ← x: activity_map, isotope
    ├── scanner.yaml            ← A: geometry, energy window, binning
    ├── sinogram.npz            ← y: trues, scatter, randoms
    └── simulator_output/       ← raw backend files (gitignored)

The Run class bundles the four petsim objects (Phantom, Source, Scanner,
Sinogram) together with a manifest dict and provides save() / load()
methods that operate on a directory rather than a single file.

This is the class that Phase 2 and Phase 3 backends will ultimately
return from their .run() methods.

The Run class itself is simulator-agnostic — it contains no code for
running simulations. It's a container for the *result* of a simulation,
regardless of which backend produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .phantom import Phantom
from .scanner import Scanner
from .sinogram import Sinogram
from .source import Source


# File names used in the storage format. These are intentionally fixed —
# the layout is part of the project's public contract, so tooling and
# humans alike can predict where things live.
MANIFEST_FILENAME = "run.yaml"
PHANTOM_FILENAME = "phantom.npz"
SOURCE_FILENAME = "source.npz"
SCANNER_FILENAME = "scanner.yaml"
SINOGRAM_FILENAME = "sinogram.npz"
SIMULATOR_OUTPUT_DIRNAME = "simulator_output"


@dataclass
class Run:
    """A complete simulation stored on disk.

    Attributes
    ----------
    path : Path
        Directory where this run lives (or will live).
    phantom : Phantom
        The phantom geometry and materials (x_geometry).
    source : Source
        The activity distribution (x_activity).
    scanner : Scanner
        The scanner that produced the sinogram (A).
    sinogram : Sinogram | None
        The measurement (y). May be None if the run is in the "inputs
        written but simulator not yet invoked" state.
    manifest : dict
        Free-form run metadata. Common keys:
          - backend: "mcgpu" | "gate"
          - seed: int
          - git_hash: str
          - wall_time_seconds: float
          - created: ISO-8601 timestamp
          - petsim_version: str
        Additional keys may be added by backends.
    """

    path: Path
    phantom: Phantom
    source: Source
    scanner: Scanner
    sinogram: Sinogram | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    # ---- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        self.path = Path(self.path)

        # The phantom and source must live on the same voxel grid —
        # this is the "x" side of the inverse problem.
        if not self.source.matches(self.phantom):
            raise ValueError(
                f"source shape {self.source.shape} or voxel size "
                f"{self.source.voxel_size} does not match phantom "
                f"({self.phantom.shape}, {self.phantom.voxel_size})"
            )

        # If a sinogram is provided, its scanner must match this run's scanner.
        if self.sinogram is not None and self.sinogram.scanner != self.scanner:
            raise ValueError(
                "sinogram.scanner does not match the run's scanner; "
                "the sinogram was produced by a different scanner"
            )

    # ---- file path helpers -------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_FILENAME

    @property
    def phantom_path(self) -> Path:
        return self.path / PHANTOM_FILENAME

    @property
    def source_path(self) -> Path:
        return self.path / SOURCE_FILENAME

    @property
    def scanner_path(self) -> Path:
        return self.path / SCANNER_FILENAME

    @property
    def sinogram_path(self) -> Path:
        return self.path / SINOGRAM_FILENAME

    @property
    def simulator_output_dir(self) -> Path:
        return self.path / SIMULATOR_OUTPUT_DIRNAME

    # ---- persistence --------------------------------------------------

    def save(self) -> None:
        """Write all four object files and the manifest into self.path.

        Creates the directory (and any needed parents) if it doesn't
        exist. Does NOT touch simulator_output/ — that's managed by
        backends and is not part of the Run's responsibility.
        """
        self.path.mkdir(parents=True, exist_ok=True)

        self.phantom.save(self.phantom_path)
        self.source.save(self.source_path)
        self.scanner.save(self.scanner_path)
        if self.sinogram is not None:
            self.sinogram.save(self.sinogram_path)

        with open(self.manifest_path, "w") as f:
            yaml.safe_dump(
                self.manifest, f, sort_keys=False, default_flow_style=False
            )

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        """Load a Run from its directory.

        The phantom, source, and scanner files are all required. The
        sinogram is optional (a run with inputs prepared but not yet
        executed won't have one). The manifest is optional and defaults
        to an empty dict if missing.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"run directory does not exist: {path}")

        phantom_path = path / PHANTOM_FILENAME
        source_path = path / SOURCE_FILENAME
        scanner_path = path / SCANNER_FILENAME
        sinogram_path = path / SINOGRAM_FILENAME
        manifest_path = path / MANIFEST_FILENAME

        for p in (phantom_path, source_path, scanner_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"missing required file for run at {path}: {p.name}"
                )

        phantom = Phantom.load(phantom_path)
        source = Source.load(source_path)
        scanner = Scanner.load(scanner_path)

        sinogram: Sinogram | None = None
        if sinogram_path.exists():
            sinogram = Sinogram.load(sinogram_path, scanner=scanner)

        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    manifest = loaded

        return cls(
            path=path,
            phantom=phantom,
            source=source,
            scanner=scanner,
            sinogram=sinogram,
            manifest=manifest,
        )

    # ---- introspection ------------------------------------------------

    def has_sinogram(self) -> bool:
        """True if this run has a sinogram (i.e. the simulator ran
        successfully and produced output).
        """
        return self.sinogram is not None

    def __repr__(self) -> str:
        sino_str = "with sinogram" if self.has_sinogram() else "no sinogram"
        backend = self.manifest.get("backend", "?")
        return (
            f"Run(path={str(self.path)!r}, "
            f"phantom={self.phantom.shape}, "
            f"scanner={self.scanner.name!r}, "
            f"backend={backend}, {sino_str})"
        )

    # ---- equality -----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Run):
            return NotImplemented
        return (
            self.phantom == other.phantom
            and self.source == other.source
            and self.scanner == other.scanner
            and self.sinogram == other.sinogram
            and self.manifest == other.manifest
        )