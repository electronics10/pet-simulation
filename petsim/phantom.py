"""
Phantom: a voxelized anatomical model.

A Phantom describes the *geometry and materials* of the object being imaged.
It does NOT carry activity information — that belongs to the Source class.
This separation mirrors GATE's convention and reflects physical reality:
the same patient anatomy can be imaged with different tracers.

A Phantom is fundamentally three aligned 3D numpy arrays:
  - material_ids: int array, each voxel holds an index into a material list
  - densities:    float array, density in g/cm^3 per voxel
  - voxel_size:   (dx, dy, dz) tuple in cm

Only material_ids and densities vary per voxel. voxel_size is grid-wide.

Backend-specific serialization (e.g. writing .vox files for MCGPU-PET)
lives in the backend modules, not here. This module only handles the
simulator-agnostic representation and npz persistence for the storage format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Phantom:
    """A voxelized phantom: material IDs + densities on a regular grid.

    Attributes
    ----------
    material_ids : np.ndarray of int, shape (nx, ny, nz)
        Material index per voxel. Values are 1-indexed to match MCGPU-PET
        convention (material 1 = first entry in the material list).
        Value 0 is reserved for "undefined" and should not appear in a
        valid phantom.
    densities : np.ndarray of float, shape (nx, ny, nz)
        Mass density per voxel in g/cm^3.
    voxel_size : tuple of (float, float, float)
        Voxel edge lengths in cm along (x, y, z).
    material_names : tuple of str
        Human-readable name for each material index, in order.
        material_names[0] corresponds to material_ids value 1.
        Names are used by the MaterialRegistry to look up simulator-specific
        material files and definitions.
    """

    material_ids: np.ndarray
    densities: np.ndarray
    voxel_size: tuple[float, float, float]
    material_names: tuple[str, ...]

    # ---- validation ------------------------------------------------------

    def __post_init__(self) -> None:
        # Coerce to numpy arrays so users can pass lists / tuples
        self.material_ids = np.asarray(self.material_ids, dtype=np.int32)
        self.densities = np.asarray(self.densities, dtype=np.float32)

        if self.material_ids.shape != self.densities.shape:
            raise ValueError(
                f"material_ids shape {self.material_ids.shape} does not match "
                f"densities shape {self.densities.shape}"
            )
        if self.material_ids.ndim != 3:
            raise ValueError(
                f"phantom arrays must be 3D, got shape {self.material_ids.shape}"
            )
        if len(self.voxel_size) != 3:
            raise ValueError(
                f"voxel_size must be length 3, got {self.voxel_size}"
            )
        if any(v <= 0 for v in self.voxel_size):
            raise ValueError(
                f"voxel_size must be positive, got {self.voxel_size}"
            )
        # Normalize voxel_size to tuple of floats
        self.voxel_size = tuple(float(v) for v in self.voxel_size)

        if len(self.material_names) == 0:
            raise ValueError("material_names must not be empty")

        # Material ID sanity: must be 1-indexed, must not exceed material list length
        max_id = int(self.material_ids.max())
        min_id = int(self.material_ids.min())
        if min_id < 1:
            raise ValueError(
                f"material_ids must be >= 1 (1-indexed); got min value {min_id}. "
                "Value 0 is reserved for undefined voxels."
            )
        if max_id > len(self.material_names):
            raise ValueError(
                f"material_ids references index {max_id} but only "
                f"{len(self.material_names)} material names provided"
            )

        self.material_names = tuple(self.material_names)

    # ---- convenience properties ------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Phantom dimensions in voxels (nx, ny, nz)."""
        return self.material_ids.shape  # type: ignore[return-value]

    @property
    def extent_cm(self) -> tuple[float, float, float]:
        """Physical size of the phantom in cm (sx, sy, sz)."""
        return tuple(n * d for n, d in zip(self.shape, self.voxel_size))  # type: ignore[return-value]

    @property
    def voxel_volume_cm3(self) -> float:
        """Volume of a single voxel in cm^3."""
        dx, dy, dz = self.voxel_size
        return dx * dy * dz

    def mass_of(self, material_name: str) -> float:
        """Total mass of a given material across the phantom, in grams."""
        if material_name not in self.material_names:
            raise KeyError(f"material {material_name!r} not in this phantom")
        idx = self.material_names.index(material_name) + 1  # 1-indexed
        mask = self.material_ids == idx
        return float(self.densities[mask].sum() * self.voxel_volume_cm3)

    # ---- factory methods -------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        material_ids: np.ndarray,
        densities: np.ndarray,
        voxel_size: Sequence[float],
        material_names: Sequence[str],
    ) -> "Phantom":
        """Construct directly from numpy arrays. Thin wrapper for clarity."""
        return cls(
            material_ids=material_ids,
            densities=densities,
            voxel_size=tuple(voxel_size),  # type: ignore[arg-type]
            material_names=tuple(material_names),
        )

    @classmethod
    def uniform(
        cls,
        shape: Sequence[int],
        voxel_size: Sequence[float],
        material: str,
        density: float,
    ) -> "Phantom":
        """A phantom filled entirely with one material."""
        nx, ny, nz = shape
        material_ids = np.ones((nx, ny, nz), dtype=np.int32)
        densities = np.full((nx, ny, nz), density, dtype=np.float32)
        return cls(
            material_ids=material_ids,
            densities=densities,
            voxel_size=tuple(voxel_size),  # type: ignore[arg-type]
            material_names=(material,),
        )

    @classmethod
    def cube(
        cls,
        shape: Sequence[int],
        voxel_size: Sequence[float],
        inner_material: str,
        inner_density: float,
        outer_material: str = "air",
        outer_density: float = 0.0012,
        inner_size_vox: int | None = None,
    ) -> "Phantom":
        """A cube of `inner_material` centered in a box of `outer_material`.

        If `inner_size_vox` is None, a cube filling the central ~55% of the
        grid is used (matches the 5-in-9 pattern of the MCGPU sample).
        """
        nx, ny, nz = shape
        if inner_size_vox is None:
            inner_size_vox = max(1, int(round(min(nx, ny, nz) * 5 / 9)))

        material_ids = np.ones((nx, ny, nz), dtype=np.int32)  # 1 = outer
        densities = np.full((nx, ny, nz), outer_density, dtype=np.float32)

        # Centered inner cube
        def slab(n: int, w: int) -> slice:
            lo = (n - w) // 2
            return slice(lo, lo + w)

        ix = slab(nx, inner_size_vox)
        iy = slab(ny, inner_size_vox)
        iz = slab(nz, inner_size_vox)
        material_ids[ix, iy, iz] = 2  # 2 = inner
        densities[ix, iy, iz] = inner_density

        return cls(
            material_ids=material_ids,
            densities=densities,
            voxel_size=tuple(voxel_size),  # type: ignore[arg-type]
            material_names=(outer_material, inner_material),
        )

    @classmethod
    def cylinder(
        cls,
        shape: Sequence[int],
        voxel_size: Sequence[float],
        radius_cm: float,
        height_cm: float,
        inner_material: str,
        inner_density: float,
        outer_material: str = "air",
        outer_density: float = 0.0012,
        axis: str = "z",
    ) -> "Phantom":
        """A cylinder of `inner_material` along the given axis, centered in
        a background of `outer_material`. Axis is one of 'x', 'y', 'z'.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")

        nx, ny, nz = shape
        dx, dy, dz = voxel_size

        material_ids = np.ones((nx, ny, nz), dtype=np.int32)
        densities = np.full((nx, ny, nz), outer_density, dtype=np.float32)

        # Voxel center coordinates in cm, origin at phantom corner
        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        zs = (np.arange(nz) + 0.5) * dz
        cx, cy, cz = xs.mean(), ys.mean(), zs.mean()

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        if axis == "z":
            in_circle = (X - cx) ** 2 + (Y - cy) ** 2 <= radius_cm**2
            in_height = np.abs(Z - cz) <= height_cm / 2
        elif axis == "y":
            in_circle = (X - cx) ** 2 + (Z - cz) ** 2 <= radius_cm**2
            in_height = np.abs(Y - cy) <= height_cm / 2
        else:  # axis == "x"
            in_circle = (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_cm**2
            in_height = np.abs(X - cx) <= height_cm / 2

        mask = in_circle & in_height
        material_ids[mask] = 2
        densities[mask] = inner_density

        return cls(
            material_ids=material_ids,
            densities=densities,
            voxel_size=tuple(voxel_size),  # type: ignore[arg-type]
            material_names=(outer_material, inner_material),
        )

    @classmethod
    def sphere(
        cls,
        shape: Sequence[int],
        voxel_size: Sequence[float],
        radius_cm: float,
        inner_material: str,
        inner_density: float,
        outer_material: str = "air",
        outer_density: float = 0.0012,
    ) -> "Phantom":
        """A sphere of `inner_material` centered in a background of
        `outer_material`.
        """
        nx, ny, nz = shape
        dx, dy, dz = voxel_size

        material_ids = np.ones((nx, ny, nz), dtype=np.int32)
        densities = np.full((nx, ny, nz), outer_density, dtype=np.float32)

        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        zs = (np.arange(nz) + 0.5) * dz
        cx, cy, cz = xs.mean(), ys.mean(), zs.mean()

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius_cm**2

        material_ids[mask] = 2
        densities[mask] = inner_density

        return cls(
            material_ids=material_ids,
            densities=densities,
            voxel_size=tuple(voxel_size),  # type: ignore[arg-type]
            material_names=(outer_material, inner_material),
        )

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to an npz file following the project storage format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            material_ids=self.material_ids,
            densities=self.densities,
            voxel_size=np.array(self.voxel_size, dtype=np.float32),
            material_names=np.array(self.material_names),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Phantom":
        """Load from an npz file produced by save()."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            return cls(
                material_ids=data["material_ids"],
                densities=data["densities"],
                voxel_size=tuple(float(v) for v in data["voxel_size"]),
                material_names=tuple(str(s) for s in data["material_names"]),
            )

    # ---- equality for testing --------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phantom):
            return NotImplemented
        return (
            np.array_equal(self.material_ids, other.material_ids)
            and np.array_equal(self.densities, other.densities)
            and self.voxel_size == other.voxel_size
            and self.material_names == other.material_names
        )

    def __repr__(self) -> str:
        return (
            f"Phantom(shape={self.shape}, "
            f"voxel_size={self.voxel_size} cm, "
            f"extent={self.extent_cm} cm, "
            f"materials={self.material_names})"
        )