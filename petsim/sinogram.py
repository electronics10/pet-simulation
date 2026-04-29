"""
Sinogram: a minimal container for PET coincidence histograms.

A Sinogram wraps a pair of arrays (trues and scatter) of arbitrary rank
plus metadata. It's a data container, nothing more.

Backends choose the layout. Two formats are currently in use:

    3D legacy:        (n_z_planes, n_angular, n_radial)
                      Used by MCGPU's native Michelogram output.
    4D ring-pair:     (n_rings, n_rings, n_angular, n_radial)
                      Used by GATE backend's LOR-based histogrammer.

The `axes` field optionally records what each axis means
(e.g. ("ring1", "ring2", "angular", "radial")) so downstream code can
introspect the layout. Empty for legacy 3D sinograms.

For ML training, you mostly just need the arrays. The save/load methods
keep them organized in `.npz` files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sinogram:
    """A PET sinogram: histogram of coincidence events.

    Attributes
    ----------
    trues : np.ndarray
        Counts of true coincidences. Arbitrary rank.
    scatter : np.ndarray or None
        Counts of scattered coincidences, same shape as `trues`.
    shape : tuple[int, ...]
        Sinogram shape, must match `trues.shape`.
    axes : tuple[str, ...]
        Optional names for each axis (e.g. ("ring1", "ring2", "angular",
        "radial")). Empty for legacy 3D sinograms.
    metadata : dict
        Free-form dict populated by backends. Common keys:
          - backend: "mcgpu" | "gate"
          - wall_time_s: float
          - returncode: int
    """

    trues: np.ndarray
    shape: tuple[int, ...]
    scatter: np.ndarray | None = None
    axes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trues = np.asarray(self.trues, dtype=np.float32)
        if self.scatter is not None:
            self.scatter = np.asarray(self.scatter, dtype=np.float32)

        self.shape = tuple(int(s) for s in self.shape)

        if self.trues.shape != self.shape:
            raise ValueError(
                f"trues shape {self.trues.shape} doesn't match declared shape {self.shape}"
            )
        if self.scatter is not None and self.scatter.shape != self.shape:
            raise ValueError(
                f"scatter shape {self.scatter.shape} doesn't match declared shape {self.shape}"
            )

        if self.axes:
            self.axes = tuple(str(a) for a in self.axes)
            if len(self.axes) != len(self.shape):
                raise ValueError(
                    f"axes length {len(self.axes)} doesn't match shape rank "
                    f"{len(self.shape)}"
                )

    @property
    def total_trues(self) -> int:
        return int(self.trues.sum())

    @property
    def total_scatter(self) -> int:
        return int(self.scatter.sum()) if self.scatter is not None else 0

    @property
    def scatter_fraction(self) -> float | None:
        """Scatter fraction = scatter / (trues + scatter)."""
        if self.scatter is None:
            return None
        denom = self.total_trues + self.total_scatter
        return self.total_scatter / denom if denom > 0 else 0.0

    def save(self, path: str | Path) -> None:
        """Save arrays and metadata to an npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "trues": self.trues,
            "shape": np.array(self.shape, dtype=np.int32),
        }
        if self.scatter is not None:
            arrays["scatter"] = self.scatter
        if self.axes:
            arrays["axes"] = np.array(self.axes)
        arrays["metadata"] = np.array(self.metadata, dtype=object)

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Sinogram":
        """Load from an npz file."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            return cls(
                trues=data["trues"],
                shape=tuple(int(x) for x in data["shape"]),
                scatter=data["scatter"] if "scatter" in data.files else None,
                axes=tuple(str(a) for a in data["axes"]) if "axes" in data.files else (),
                metadata=data["metadata"].item() if "metadata" in data.files else {},
            )

    def __repr__(self) -> str:
        components = [f"trues={self.total_trues}"]
        if self.scatter is not None:
            components.append(f"scatter={self.total_scatter}")
            sf = self.scatter_fraction
            if sf is not None:
                components.append(f"SF={sf:.1%}")
        axes_part = f", axes={self.axes}" if self.axes else ""
        return f"Sinogram(shape={self.shape}{axes_part}, {', '.join(components)})"