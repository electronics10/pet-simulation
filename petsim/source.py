"""
Source: a radioactivity distribution over a phantom grid.

A Source describes *where the radiotracer is* and *how much*. It's the
activity side of the ground truth (x = phantom geometry + source activity).

Physically, a Source is a 3D numpy array of activity values in Bq, aligned
with a Phantom's voxel grid. A voxel value of 100 Bq means 100 decays per
second are emitted from that voxel on average.

This separation from Phantom mirrors GATE's convention: the same patient
anatomy (Phantom) can be imaged with different tracers (Source).

Isotope information is stored as a simple string plus a small lookup table
for half-life. Positron range and prompt-gamma fraction can be added later
if needed.

MCGPU-PET's .vox format fuses phantom and source into a single text file
where each voxel line reads "material_id density activity". That fusion
happens in the MCGPU backend at write time, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .phantom import Phantom


# Half-lives in seconds for common PET isotopes.
# Source: NIST and IAEA tables. Add more as needed.
ISOTOPE_HALF_LIFE_S = {
    "F18":  6586.2,     # 109.77 minutes
    "C11":  1221.8,     # 20.36 minutes
    "N13":   597.9,     # 9.965 minutes
    "O15":   122.24,    # 2.037 minutes
    "Ga68": 4062.7,     # 67.71 minutes
    "Rb82":    75.0,    # 1.25 minutes
    "Cu64": 45721.2,    # 12.70 hours
    "Zr89": 282240.0,   # 78.4 hours
}


@dataclass
class Source:
    """A radioactivity distribution aligned with a Phantom grid.

    Attributes
    ----------
    activity_Bq : np.ndarray of float, shape (nx, ny, nz)
        Activity per voxel in Bq. Must match the Phantom's shape.
    voxel_size : tuple of (float, float, float)
        Voxel edge lengths in cm, copied from the Phantom for
        self-contained serialization.
    isotope : str
        Name of the radioisotope (e.g. "F18"). Must be a key in
        ISOTOPE_HALF_LIFE_S.
    """

    activity_Bq: np.ndarray
    voxel_size: tuple[float, float, float]
    isotope: str

    # ---- validation ------------------------------------------------------

    def __post_init__(self) -> None:
        self.activity_Bq = np.asarray(self.activity_Bq, dtype=np.float32)

        if self.activity_Bq.ndim != 3:
            raise ValueError(
                f"activity_Bq must be 3D, got shape {self.activity_Bq.shape}"
            )
        if len(self.voxel_size) != 3:
            raise ValueError(
                f"voxel_size must be length 3, got {self.voxel_size}"
            )
        if any(v <= 0 for v in self.voxel_size):
            raise ValueError(
                f"voxel_size must be positive, got {self.voxel_size}"
            )
        if np.any(self.activity_Bq < 0):
            raise ValueError(
                "activity_Bq must be non-negative everywhere; "
                f"found min value {float(self.activity_Bq.min())}"
            )
        if self.isotope not in ISOTOPE_HALF_LIFE_S:
            raise ValueError(
                f"unknown isotope {self.isotope!r}; known isotopes: "
                f"{sorted(ISOTOPE_HALF_LIFE_S.keys())}"
            )

        self.voxel_size = tuple(float(v) for v in self.voxel_size)

    # ---- convenience properties ------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Source dimensions in voxels (nx, ny, nz)."""
        return self.activity_Bq.shape  # type: ignore[return-value]

    @property
    def total_activity_Bq(self) -> float:
        """Sum of activity across all voxels, in Bq."""
        return float(self.activity_Bq.sum())

    @property
    def half_life_s(self) -> float:
        """Half-life of the isotope in seconds."""
        return ISOTOPE_HALF_LIFE_S[self.isotope]

    @property
    def mean_lifetime_s(self) -> float:
        """Mean (exponential) lifetime tau = T_half / ln(2), in seconds.
        This is the value MCGPU-PET uses for its decay model.
        """
        return self.half_life_s / np.log(2)

    def decay_factor(self, time_s: float) -> float:
        """Fraction of activity remaining after `time_s` seconds.

        Useful for e.g. "how much activity is left after 1 hour".
        """
        if time_s < 0:
            raise ValueError(f"time_s must be non-negative; got {time_s}")
        return float(np.exp(-time_s / self.mean_lifetime_s))

    def matches(self, phantom: Phantom) -> bool:
        """Check that this Source is compatible with the given Phantom."""
        return (
            self.shape == phantom.shape
            and self.voxel_size == phantom.voxel_size
        )

    # ---- factory methods -------------------------------------------------

    @classmethod
    def zeros(cls, phantom: Phantom, isotope: str = "F18") -> "Source":
        """A source with zero activity everywhere. Useful as a starting
        point for building up activity distributions manually.
        """
        activity = np.zeros(phantom.shape, dtype=np.float32)
        return cls(
            activity_Bq=activity,
            voxel_size=phantom.voxel_size,
            isotope=isotope,
        )

    @classmethod
    def from_numpy(
        cls,
        phantom: Phantom,
        activity_Bq: np.ndarray,
        isotope: str = "F18",
    ) -> "Source":
        """Construct from an explicit activity array aligned with a phantom."""
        activity_Bq = np.asarray(activity_Bq, dtype=np.float32)
        if activity_Bq.shape != phantom.shape:
            raise ValueError(
                f"activity array shape {activity_Bq.shape} does not match "
                f"phantom shape {phantom.shape}"
            )
        return cls(
            activity_Bq=activity_Bq,
            voxel_size=phantom.voxel_size,
            isotope=isotope,
        )

    @classmethod
    def uniform_in_material(
        cls,
        phantom: Phantom,
        material: str,
        activity_per_voxel_Bq: float,
        isotope: str = "F18",
    ) -> "Source":
        """Constant activity in every voxel belonging to `material`,
        zero elsewhere.

        Useful for background uptake in a tissue (e.g. brain gray matter).
        """
        if material not in phantom.material_names:
            raise KeyError(
                f"material {material!r} not in phantom; available: "
                f"{phantom.material_names}"
            )
        if activity_per_voxel_Bq < 0:
            raise ValueError(
                f"activity_per_voxel_Bq must be non-negative; "
                f"got {activity_per_voxel_Bq}"
            )

        idx = phantom.material_names.index(material) + 1  # 1-indexed
        mask = phantom.material_ids == idx
        activity = np.zeros(phantom.shape, dtype=np.float32)
        activity[mask] = float(activity_per_voxel_Bq)

        return cls(
            activity_Bq=activity,
            voxel_size=phantom.voxel_size,
            isotope=isotope,
        )

    @classmethod
    def with_total_activity(
        cls,
        phantom: Phantom,
        material: str,
        total_activity_Bq: float,
        isotope: str = "F18",
    ) -> "Source":
        """Distribute `total_activity_Bq` uniformly across all voxels of
        the given material. Each voxel gets total / n_voxels.

        Useful when you know the total dose (e.g. 1 MBq of FDG) and want
        to spread it uniformly through a tissue.
        """
        if material not in phantom.material_names:
            raise KeyError(
                f"material {material!r} not in phantom; available: "
                f"{phantom.material_names}"
            )
        if total_activity_Bq < 0:
            raise ValueError(
                f"total_activity_Bq must be non-negative; got {total_activity_Bq}"
            )

        idx = phantom.material_names.index(material) + 1
        mask = phantom.material_ids == idx
        n_voxels = int(mask.sum())
        if n_voxels == 0:
            raise ValueError(
                f"material {material!r} occupies zero voxels in this phantom"
            )

        activity = np.zeros(phantom.shape, dtype=np.float32)
        activity[mask] = total_activity_Bq / n_voxels

        return cls(
            activity_Bq=activity,
            voxel_size=phantom.voxel_size,
            isotope=isotope,
        )

    def add_hot_spot(
        self,
        position_cm: Sequence[float],
        activity_Bq: float,
        radius_cm: float = 0.0,
    ) -> "Source":
        """Return a new Source with a hot spot added at `position_cm`.

        If radius_cm == 0, a single voxel at the given position is set.
        Otherwise, all voxels whose centers fall within radius_cm of the
        position receive the activity (split evenly across them).

        Position is in cm from the phantom corner origin, matching the
        Phantom coordinate system.
        """
        if len(position_cm) != 3:
            raise ValueError(
                f"position_cm must be length 3, got {position_cm}"
            )
        if activity_Bq < 0:
            raise ValueError(
                f"activity_Bq must be non-negative; got {activity_Bq}"
            )
        if radius_cm < 0:
            raise ValueError(f"radius_cm must be non-negative; got {radius_cm}")

        nx, ny, nz = self.shape
        dx, dy, dz = self.voxel_size
        px, py, pz = [float(c) for c in position_cm]

        new_activity = self.activity_Bq.copy()

        if radius_cm == 0.0:
            # Single voxel containing the position
            ix = int(px / dx)
            iy = int(py / dy)
            iz = int(pz / dz)
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                raise ValueError(
                    f"position {position_cm} is outside phantom extent "
                    f"{(nx*dx, ny*dy, nz*dz)} cm"
                )
            new_activity[ix, iy, iz] += float(activity_Bq)
        else:
            xs = (np.arange(nx) + 0.5) * dx
            ys = (np.arange(ny) + 0.5) * dy
            zs = (np.arange(nz) + 0.5) * dz
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            mask = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2 <= radius_cm**2
            n_voxels = int(mask.sum())
            if n_voxels == 0:
                raise ValueError(
                    f"no voxel centers fall within {radius_cm} cm of {position_cm}"
                )
            new_activity[mask] += float(activity_Bq) / n_voxels

        return Source(
            activity_Bq=new_activity,
            voxel_size=self.voxel_size,
            isotope=self.isotope,
        )

    # ---- diagnostics -----------------------------------------------------

    def check_activity_in_air(
        self, phantom: Phantom, air_material: str = "air"
    ) -> tuple[int, float]:
        """Report how many voxels have activity in the air material.

        Returns (n_voxels, total_activity_Bq) for activity-bearing voxels
        that are in the air material. This is almost always a bug.
        Returns (0, 0.0) if there is no air material in the phantom.
        """
        if not self.matches(phantom):
            raise ValueError("Source and Phantom grids do not match")
        if air_material not in phantom.material_names:
            return (0, 0.0)

        idx = phantom.material_names.index(air_material) + 1
        air_mask = phantom.material_ids == idx
        hot_air = air_mask & (self.activity_Bq > 0)
        return int(hot_air.sum()), float(self.activity_Bq[hot_air].sum())

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to an npz file following the project storage format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            activity_Bq=self.activity_Bq,
            voxel_size=np.array(self.voxel_size, dtype=np.float64),
            isotope=np.array(self.isotope),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Source":
        """Load from an npz file produced by save()."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            return cls(
                activity_Bq=data["activity_Bq"],
                voxel_size=tuple(float(v) for v in data["voxel_size"]),
                isotope=str(data["isotope"]),
            )

    # ---- equality for testing --------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Source):
            return NotImplemented
        return (
            np.array_equal(self.activity_Bq, other.activity_Bq)
            and self.voxel_size == other.voxel_size
            and self.isotope == other.isotope
        )

    def __repr__(self) -> str:
        return (
            f"Source(shape={self.shape}, "
            f"isotope={self.isotope!r}, "
            f"total_activity={self.total_activity_Bq:.3g} Bq, "
            f"half_life={self.half_life_s:.1f} s)"
        )