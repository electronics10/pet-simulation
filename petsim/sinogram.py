"""
Sinogram: a container for PET coincidence histograms.

A Sinogram holds the measurement side of the inverse problem:

    y_measured = y_trues + y_scatter + y_randoms

Each component is a 3D numpy array with shape
(n_z_slices, n_angular_bins, n_radial_bins). Each bin counts how many
coincidence events were detected along a specific Line of Response.

A Sinogram carries a reference to the Scanner that produced it, because
the physical meaning of each bin is only defined relative to the scanner
geometry and binning convention. The Scanner is saved alongside as a
separate YAML file to keep the storage format modular.

Any of trues / scatter / randoms may be None. For MCGPU-PET we populate
trues and scatter. For a real scanner we'd only have the total. Downstream
code must check for None before using a component.

This module is a dumb container. It contains no physics and does no
simulation — those live in the backend modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .scanner import Scanner


@dataclass
class Sinogram:
    """A PET sinogram: histogram of coincidence events.

    Attributes
    ----------
    trues : np.ndarray of int or float, or None
        Counts of true coincidences per LOR, shape matches scanner.sinogram_shape.
    scatter : np.ndarray or None
        Counts of scattered coincidences per LOR.
    randoms : np.ndarray or None
        Counts of random (accidental) coincidences per LOR.
    scanner : Scanner
        The scanner that produced this sinogram. Required — without it,
        the bin indices have no physical meaning.
    metadata : dict
        Free-form metadata populated by backends. Common keys:
          - backend: "mcgpu" | "gate"
          - simulated_histories: int
          - wall_time_seconds: float
          - scatter_fraction: float
          - seed: int
          - git_hash: str
    """

    scanner: Scanner
    trues: np.ndarray | None = None
    scatter: np.ndarray | None = None
    randoms: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        expected_shape = self.scanner.sinogram_shape

        for name in ("trues", "scatter", "randoms"):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected "
                    f"{expected_shape} from scanner binning"
                )
            if np.any(arr < 0):
                raise ValueError(
                    f"{name} contains negative values; counts must be "
                    f"non-negative. Min value: {float(arr.min())}"
                )
            setattr(self, name, arr)

        if self.trues is None and self.scatter is None and self.randoms is None:
            raise ValueError(
                "at least one of trues / scatter / randoms must be provided"
            )

    # ---- convenience properties --------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Sinogram shape, which always matches scanner.sinogram_shape."""
        return self.scanner.sinogram_shape

    @property
    def measured(self) -> np.ndarray:
        """Sum of all available components: trues + scatter + randoms.

        This is the closest analogue to what a real scanner would report:
        a histogram of coincidences with no component separation.
        Components that are None are treated as zero.
        """
        parts = [c for c in (self.trues, self.scatter, self.randoms) if c is not None]
        result = np.zeros(self.shape, dtype=np.float64)
        for p in parts:
            result += p
        return result

    @property
    def total_trues(self) -> int:
        return int(self.trues.sum()) if self.trues is not None else 0

    @property
    def total_scatter(self) -> int:
        return int(self.scatter.sum()) if self.scatter is not None else 0

    @property
    def total_randoms(self) -> int:
        return int(self.randoms.sum()) if self.randoms is not None else 0

    @property
    def scatter_fraction(self) -> float | None:
        """Scatter fraction = scatter / (trues + scatter).

        Standard metric reported by most PET literature. Returns None if
        either component is missing.
        """
        if self.trues is None or self.scatter is None:
            return None
        denom = self.total_trues + self.total_scatter
        if denom == 0:
            return 0.0
        return self.total_scatter / denom

    # ---- persistence --------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the sinogram arrays and metadata to an npz file.

        The Scanner is NOT saved here — it belongs next to this file as
        `scanner.yaml` in the run directory. This keeps the storage format
        modular: you can load a Scanner without pulling in a 100 MB
        sinogram, and vice versa.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        if self.trues is not None:
            arrays["trues"] = self.trues
        if self.scatter is not None:
            arrays["scatter"] = self.scatter
        if self.randoms is not None:
            arrays["randoms"] = self.randoms

        # Metadata goes in as a 0-d object array holding a dict. npz
        # supports this natively via pickle. We allow_pickle=True when
        # loading only for this key.
        arrays["metadata"] = np.array(self.metadata, dtype=object)

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path, scanner: Scanner) -> "Sinogram":
        """Load from an npz file. The Scanner must be provided explicitly
        since it lives in a separate YAML file in the run directory.

        Typical usage:
            scanner  = Scanner.load("runs/001/scanner.yaml")
            sinogram = Sinogram.load("runs/001/sinogram.npz", scanner)
        """
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            trues   = data["trues"]   if "trues"   in data.files else None
            scatter = data["scatter"] if "scatter" in data.files else None
            randoms = data["randoms"] if "randoms" in data.files else None
            metadata = (
                data["metadata"].item() if "metadata" in data.files else {}
            )

        return cls(
            scanner=scanner,
            trues=trues,
            scatter=scatter,
            randoms=randoms,
            metadata=metadata,
        )

    # ---- equality / repr ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sinogram):
            return NotImplemented

        def arr_eq(a: np.ndarray | None, b: np.ndarray | None) -> bool:
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return np.array_equal(a, b)

        return (
            self.scanner == other.scanner
            and arr_eq(self.trues, other.trues)
            and arr_eq(self.scatter, other.scatter)
            and arr_eq(self.randoms, other.randoms)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        components = []
        if self.trues is not None:
            components.append(f"trues={self.total_trues}")
        if self.scatter is not None:
            components.append(f"scatter={self.total_scatter}")
        if self.randoms is not None:
            components.append(f"randoms={self.total_randoms}")
        sf = self.scatter_fraction
        sf_str = f", SF={sf:.1%}" if sf is not None else ""
        return (
            f"Sinogram(shape={self.shape}, "
            f"{', '.join(components)}{sf_str}, "
            f"scanner={self.scanner.name!r})"
        )