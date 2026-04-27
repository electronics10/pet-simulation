"""
SinogramBinning: a backend-agnostic sinogram layout specification.

The asymmetry problem
---------------------
MCGPU-PET is opinionated about sinogram layout: you give it (span, MRD,
n_radial, n_angular) and it produces a sinogram in its own Michelogram
convention.

GATE is unopinionated: it produces list-mode events and the user must
histogram them.

This class carries the four parameters that fully specify the binning,
plus validation against a Scanner. The actual mapping from list-mode
events to sinogram bins (the Michelogram math) is intentionally NOT
implemented here — it lives in the GATE backend's histogrammer, which
must match MCGPU-PET's specific convention exactly.

For MCGPU-PET, the output sinogram shape is determined by the simulator
itself (we just read the file). For GATE, the histogrammer fills in
whatever shape is implied by accumulating events.

Default values
--------------
SinogramBinning.default_for(scanner) computes sensible defaults from
scanner geometry, so most users don't need to think about these
parameters at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .scanner import Scanner


@dataclass
class SinogramBinning:
    """Backend-agnostic sinogram layout specification.

    Attributes
    ----------
    n_radial_bins : int
        Number of radial bins (transverse direction within each plane).
    n_angular_bins : int
        Number of angular bins (LOR azimuth within each plane).
    span : int
        Michelogram span — controls axial compression. Must be odd.
        Common values: 1 (no compression), 3, 5, 7, 11.
    max_ring_difference : int
        Maximum ring difference (MRD). Caps how much axial obliqueness
        is allowed. Must be < scanner.n_rings.
    """

    n_radial_bins: int
    n_angular_bins: int
    span: int
    max_ring_difference: int

    def __post_init__(self) -> None:
        if self.n_radial_bins <= 0:
            raise ValueError(f"n_radial_bins must be positive; got {self.n_radial_bins}")
        if self.n_angular_bins <= 0:
            raise ValueError(f"n_angular_bins must be positive; got {self.n_angular_bins}")
        if self.span <= 0 or self.span % 2 == 0:
            raise ValueError(
                f"span must be a positive odd integer; got {self.span}"
            )
        if self.max_ring_difference < 0:
            raise ValueError(
                f"max_ring_difference must be non-negative; got {self.max_ring_difference}"
            )

    # ---- validation against scanner ----------------------------------

    def validate(self, scanner: "Scanner") -> None:
        """Check that this binning is compatible with the given scanner.

        Raises ValueError on hard mismatches. Warns on soft non-standard
        choices (which are still allowed).
        """
        if self.max_ring_difference >= scanner.n_rings:
            raise ValueError(
                f"max_ring_difference ({self.max_ring_difference}) must be < "
                f"scanner.n_rings ({scanner.n_rings})"
            )

        expected_angular = scanner.n_crystals_per_ring // 2
        if self.n_angular_bins != expected_angular:
            import warnings
            warnings.warn(
                f"n_angular_bins={self.n_angular_bins} differs from the "
                f"standard value n_crystals_per_ring/2={expected_angular}. "
                f"This is non-standard but legal; sinograms may have "
                f"reduced angular resolution.",
                stacklevel=2,
            )

    # ---- factories ---------------------------------------------------

    @classmethod
    def default_for(cls, scanner: "Scanner") -> "SinogramBinning":
        """Compute sensible default binning for any scanner.

        Defaults:
          - n_radial_bins = n_angular_bins = n_crystals_per_ring / 2
          - span = 3 (mild axial compression, standard for small animal PET)
          - max_ring_difference = n_rings - 1 (use all ring pairs)

        For specific scanners with known clinical conventions, you may
        want to override (e.g. mcgpu_sample uses span=11, MRD=79).
        """
        n = scanner.n_crystals_per_ring // 2
        return cls(
            n_radial_bins=n,
            n_angular_bins=n,
            span=3,
            max_ring_difference=scanner.n_rings - 1,
        )

    # ---- persistence -------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "SinogramBinning":
        """Load from a YAML file produced by save()."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    # ---- equality / repr ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SinogramBinning):
            return NotImplemented
        return asdict(self) == asdict(other)

    def __repr__(self) -> str:
        return (
            f"SinogramBinning(n_radial={self.n_radial_bins}, "
            f"n_angular={self.n_angular_bins}, "
            f"span={self.span}, MRD={self.max_ring_difference})"
        )


# =====================================================================
# Common preset bindings (optional, for convenience)
# =====================================================================

BINNING_PRESETS: dict[str, dict[str, Any]] = {
    # Matches the MCGPU-PET sample simulation. Use with mcgpu_sample scanner.
    "mcgpu_sample": {
        "n_radial_bins": 147,
        "n_angular_bins": 168,
        "span": 11,
        "max_ring_difference": 79,
    },
    # Reasonable defaults for the Bruker Albira (24 rings, 64 crystals/ring).
    "bruker_albira": {
        "n_radial_bins": 32,
        "n_angular_bins": 32,
        "span": 3,
        "max_ring_difference": 23,
    },
}


def preset_binning(name: str, **overrides: Any) -> SinogramBinning:
    """Convenience: get a SinogramBinning from a named preset.

    Example:
        binning = preset_binning("bruker_albira")
        binning = preset_binning("mcgpu_sample", span=3)  # override span
    """
    if name not in BINNING_PRESETS:
        raise KeyError(
            f"unknown binning preset {name!r}; available: {sorted(BINNING_PRESETS)}"
        )
    params = dict(BINNING_PRESETS[name])
    params.update(overrides)
    return SinogramBinning(**params)