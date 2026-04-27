"""
Scanner: PET scanner hardware specification.

A Scanner describes ONLY the physical hardware: detector geometry, energy
window, timing resolution, and crystal-level layout. It deliberately does
NOT include sinogram binning parameters (radial bins, angular bins, span,
MRD), because those are *choices about how to histogram the data*, not
properties of the scanner itself. The same physical scanner can produce
sinograms with many different binning conventions.

Backends own the binning:
  - MCGPU-PET: see MCGPUConfig in petsim/backends/mcgpu.py
  - GATE: see GATEConfig in petsim/backends/gate.py

Two levels of detail are supported:

  1. Idealized cylinder — minimum needed for MCGPU-PET. A scanner is a
     cylindrical detector with a radius, axial length, and energy window.

  2. Full crystal geometry — optional fields describing the crystal /
     module / rsector structure. Required for GATE's detailed geometry tree.

Presets for known scanners (mcgpu_sample, bruker_albira) live in
SCANNER_PRESETS at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Scanner:
    """PET scanner hardware specification.

    Required fields:
      - detector_radius_cm, detector_axial_length_cm
      - n_rings, n_crystals_per_ring (the detector ring structure)
      - energy_window_keV, energy_resolution
      - acquisition_time_s

    Optional fields:
      - coincidence_window_ns (timing)
      - crystal-level geometry (size, material, module/rsector layout)
        — needed for GATE
    """

    # ---- Identification ------------------------------------------------
    name: str

    # ---- Detector geometry (required) ----------------------------------
    detector_radius_cm: float          # inner radius of the detector ring
    detector_axial_length_cm: float    # axial extent of the detector

    # ---- Detector ring structure (required) ----------------------------
    n_rings: int                       # number of detector rings stacked axially
    n_crystals_per_ring: int           # crystals distributed around one ring

    # ---- Physics window (required) -------------------------------------
    energy_window_keV: tuple[float, float]  # (low, high) coincidence window
    energy_resolution: float                # fractional FWHM at 511 keV (e.g. 0.12)

    # ---- Acquisition (required) ----------------------------------------
    acquisition_time_s: float          # total simulated / real scan time

    # ---- Coincidence timing (optional) ---------------------------------
    coincidence_window_ns: float | None = None

    # ---- Crystal-level geometry (optional, for GATE) -------------------
    crystal_size_mm: tuple[float, float, float] | None = None
    crystal_material: str | None = None
    crystals_per_module: tuple[int, int] | None = None     # (axial, tangential)
    modules_per_rsector: tuple[int, int, int] | None = None  # (radial, tang, axial)
    n_rsectors: int | None = None

    # ---- Free-form backend hints ---------------------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        if self.detector_radius_cm <= 0:
            raise ValueError(
                f"detector_radius_cm must be positive; got {self.detector_radius_cm}"
            )
        if self.detector_axial_length_cm <= 0:
            raise ValueError(
                f"detector_axial_length_cm must be positive; "
                f"got {self.detector_axial_length_cm}"
            )
        if self.n_rings <= 0:
            raise ValueError(f"n_rings must be positive; got {self.n_rings}")
        if self.n_crystals_per_ring <= 0:
            raise ValueError(
                f"n_crystals_per_ring must be positive; got {self.n_crystals_per_ring}"
            )
        if self.acquisition_time_s <= 0:
            raise ValueError(
                f"acquisition_time_s must be positive; got {self.acquisition_time_s}"
            )

        lo, hi = self.energy_window_keV
        if lo < 0 or hi <= lo:
            raise ValueError(
                f"energy_window_keV must have 0 <= low < high; got ({lo}, {hi})"
            )
        self.energy_window_keV = (float(lo), float(hi))

        if not 0 < self.energy_resolution < 1:
            raise ValueError(
                f"energy_resolution must be in (0, 1) as a fractional FWHM; "
                f"got {self.energy_resolution}"
            )

        if self.coincidence_window_ns is not None and self.coincidence_window_ns <= 0:
            raise ValueError(
                f"coincidence_window_ns must be positive if provided; "
                f"got {self.coincidence_window_ns}"
            )

    # ---- convenience --------------------------------------------------

    @property
    def energy_window_eV(self) -> tuple[float, float]:
        """Energy window in eV (MCGPU-PET's native unit)."""
        lo, hi = self.energy_window_keV
        return (lo * 1000.0, hi * 1000.0)

    # ---- factories ----------------------------------------------------

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> "Scanner":
        """Construct a Scanner from a named preset, optionally overriding
        individual fields.

        Example:
            s = Scanner.from_preset("mcgpu_sample", acquisition_time_s=10)
            s = Scanner.from_preset("bruker_albira", name="my_custom_run")
        """
        if preset not in SCANNER_PRESETS:
            raise KeyError(
                f"unknown preset {preset!r}; available: {sorted(SCANNER_PRESETS)}"
            )
        params = dict(SCANNER_PRESETS[preset])
        params.update(overrides)
        return cls(**params)

    # ---- persistence --------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a YAML file for human readability."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key] = list(value)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load(cls, path: str | Path) -> "Scanner":
        """Load from a YAML file produced by save()."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        tuple_fields = {
            "energy_window_keV",
            "crystal_size_mm",
            "crystals_per_module",
            "modules_per_rsector",
        }
        for key in tuple_fields:
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])

        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    # ---- equality / repr ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scanner):
            return NotImplemented
        return asdict(self) == asdict(other)

    def __repr__(self) -> str:
        return (
            f"Scanner({self.name!r}, "
            f"R={self.detector_radius_cm}cm, "
            f"L={self.detector_axial_length_cm}cm, "
            f"rings={self.n_rings}x{self.n_crystals_per_ring}, "
            f"E=({self.energy_window_keV[0]}, {self.energy_window_keV[1]}) keV, "
            f"T={self.acquisition_time_s}s)"
        )


# =====================================================================
# Scanner presets
# =====================================================================

SCANNER_PRESETS: dict[str, dict[str, Any]] = {
    # Mirrors the MCGPU-PET sample_simulation/MCGPU-PET.in file.
    # Toy geometry centered on a 9x9x9 cm phantom.
    "mcgpu_sample": {
        "name": "mcgpu_sample",
        "detector_radius_cm": 9.05,
        "detector_axial_length_cm": 12.656,
        "n_rings": 80,
        "n_crystals_per_ring": 336,
        "energy_window_keV": (350.0, 600.0),
        "energy_resolution": 0.12,
        "acquisition_time_s": 1.0,
    },

    # Bruker small-animal PET scanner.
    # Specs extracted from Bruker_PET1.mac (GATE 9.4.1):
    #   - cylindricalPET: Rmax=82mm, Rmin=58mm, height=105mm
    #   - rsector at x=67mm, crystal block 10x50x50 mm
    #   - 8x8 crystals per module, 1x1x3 modules per rsector, 8 rsectors
    #     → 24 rings axial, 64 crystals per ring
    #   - LYSO crystals
    #   - Energy window 350-650 keV, resolution 0.15 at 511 keV
    #   - Coincidence window 10 ns
    "bruker_albira": {
        "name": "bruker_albira",
        "detector_radius_cm": 6.2,         # crystal center at 67mm rsector + 10mm/2 - inner edge
        "detector_axial_length_cm": 10.5,
        "n_rings": 24,                     # 8 crystals * 3 modules
        "n_crystals_per_ring": 64,         # 8 crystals * 8 rsectors
        "energy_window_keV": (350.0, 650.0),
        "energy_resolution": 0.15,
        "acquisition_time_s": 1.0,
        "coincidence_window_ns": 10.0,
        "crystal_size_mm": (10.0, 6.25, 6.25),  # 50mm / 8 crystals
        "crystal_material": "LYSO",
        "crystals_per_module": (8, 8),
        "modules_per_rsector": (1, 1, 3),
        "n_rsectors": 8,
    },
}