"""
Sinogram: a minimal container for PET coincidence histograms.

A Sinogram is just a pair of 3D arrays (trues and scatter) with shape
(n_z, n_angular, n_radial), plus metadata. It's a data container, nothing more.

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
        Counts of true coincidences, shape (n_z, n_angular, n_radial).
    scatter : np.ndarray or None
        Counts of scattered coincidences, same shape. Can be None.
    shape : tuple[int, int, int]
        The sinogram shape (n_z, n_angular, n_radial) for reference.
    metadata : dict
        Free-form dict populated by backends. Common keys:
          - backend: "mcgpu" | "gate"
          - wall_time_s: float
          - returncode: int
    """

    trues: np.ndarray
    shape: tuple[int, int, int]
    scatter: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trues = np.asarray(self.trues, dtype=np.float32)
        if self.scatter is not None:
            self.scatter = np.asarray(self.scatter, dtype=np.float32)
        
        if self.trues.shape != self.shape:
            raise ValueError(
                f"trues shape {self.trues.shape} doesn't match declared shape {self.shape}"
            )
        if self.scatter is not None and self.scatter.shape != self.shape:
            raise ValueError(
                f"scatter shape {self.scatter.shape} doesn't match declared shape {self.shape}"
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

        arrays = {"trues": self.trues, "shape": np.array(self.shape, dtype=np.int32)}
        if self.scatter is not None:
            arrays["scatter"] = self.scatter
        arrays["metadata"] = np.array(self.metadata, dtype=object)

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Sinogram":
        """Load from an npz file."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            return cls(
                trues=data["trues"],
                shape=tuple(data["shape"]),
                scatter=data["scatter"] if "scatter" in data.files else None,
                metadata=data["metadata"].item() if "metadata" in data.files else {},
            )

    def __repr__(self) -> str:
        components = [f"trues={self.total_trues}"]
        if self.scatter is not None:
            components.append(f"scatter={self.total_scatter}")
            sf = self.scatter_fraction
            if sf is not None:
                components.append(f"SF={sf:.1%}")
        return f"Sinogram(shape={self.shape}, {', '.join(components)})"