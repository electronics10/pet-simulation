"""
Scanner: PET scanner geometry and acquisition parameters.

A Scanner describes the *measurement system* — the forward model A in the
inverse problem y = A(x). It captures everything needed to interpret what
each sinogram bin means physically: detector geometry, energy window,
timing resolution, sinogram binning convention, and normalization
assumptions.

Making A explicit matters because two scanners with the same geometry can
still produce incompatible sinograms if they disagree about:

  - how LORs are indexed (the binning convention)
  - whether TOF information is preserved
  - whether the sinogram has been normalized or attenuation-corrected

Every Scanner therefore carries three fields that force the forward-model
assumptions to be written down, not assumed:

  - binning_convention: free-form string identifying the LOR indexing
    scheme, e.g. "mcgpu_span11_mrd79" or "gate_listmode". Two sinograms
    with different binning_convention values are NOT directly comparable.
  - tof_enabled: whether the measurement preserves arrival-time info.
  - normalization: what corrections have been applied to the sinogram.

Two levels of detail are supported:

  1. Idealized cylinder — the level MCGPU-PET operates at. A scanner is
     a cylindrical detector with a radius, axial length, and energy
     window. This is the minimum information needed to run an MCGPU-PET
     simulation.

  2. Full crystal geometry — optional fields describing the crystal /
     module / rsector structure. These are needed to translate into
     GATE's detailed geometry tree.

Sinogram binning parameters (number of radial bins, angular bins, span,
max ring difference) are part of the Scanner because different scanners
bin their coincidences differently. Storing them here keeps the
sinogram self-describing: you can always recover the physical meaning of
a bin from the Scanner that produced it.

Presets for known scanners (starting with Bruker Albira, from the
existing gate-pet/bruker_pet_sim.py) live in SCANNER_PRESETS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Scanner:
    """PET scanner geometry and acquisition parameters.

    Minimum-required fields for MCGPU-PET:
      - detector_radius_cm, detector_axial_length_cm
      - energy_window_keV
      - energy_resolution
      - acquisition_time_s
      - sinogram binning parameters

    Optional fields for GATE (and descriptive purposes):
      - crystal geometry (size, material)
      - module / rsector layout
    """

    # ---- Identification ------------------------------------------------
    name: str

    # ---- Detector geometry (required) ----------------------------------
    detector_radius_cm: float          # inner radius of the detector ring
    detector_axial_length_cm: float    # axial extent of the detector

    # ---- Physics window (required) -------------------------------------
    energy_window_keV: tuple[float, float]  # (low, high) coincidence window
    energy_resolution: float                # fractional FWHM at 511 keV (e.g. 0.12)

    # ---- Acquisition (required) ----------------------------------------
    acquisition_time_s: float          # total simulated / real scan time

    # ---- Sinogram binning (required) -----------------------------------
    n_radial_bins: int                 # "NRAD" in MCGPU-PET
    n_angular_bins: int                # "NANGLES"
    n_z_slices: int                    # total slices in 3D sinogram
    span: int                          # axial compression (Michelogram span)
    max_ring_difference: int           # "MRD"

    # ---- Forward-model assumptions (required for comparability) --------
    # These three fields make the "A" in y = A(x) explicit. Two sinograms
    # with different values here are NOT directly comparable, even if all
    # the geometry numbers match. See module docstring for rationale.
    binning_convention: str = "unspecified"
    tof_enabled: bool = False
    normalization: str = "none"         # "none" | "attenuation_corrected" | ...

    # ---- Coincidence timing (optional) ---------------------------------
    coincidence_window_ns: float | None = None

    # ---- Crystal-level geometry (optional, for GATE) -------------------
    crystal_size_mm: tuple[float, float, float] | None = None
    crystal_material: str | None = None
    crystals_per_module: tuple[int, int] | None = None     # (axial, tangential)
    modules_per_rsector: tuple[int, int, int] | None = None  # (radial, tang, axial)
    n_rsectors: int | None = None

    # ---- Free-form backend hints ---------------------------------------
    # Useful for storing scanner-specific parameters that don't fit the
    # general schema, e.g. MCGPU-PET's NSEG or NBINS.
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

        for name, value in (
            ("n_radial_bins", self.n_radial_bins),
            ("n_angular_bins", self.n_angular_bins),
            ("n_z_slices", self.n_z_slices),
            ("span", self.span),
            ("max_ring_difference", self.max_ring_difference),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")

        if self.coincidence_window_ns is not None and self.coincidence_window_ns <= 0:
            raise ValueError(
                f"coincidence_window_ns must be positive if provided; "
                f"got {self.coincidence_window_ns}"
            )

        # Validate normalization against known values. Expand this list
        # as real normalization pipelines are implemented.
        allowed_normalizations = {
            "none",
            "attenuation_corrected",
            "normalization_corrected",
            "scatter_corrected",
            "fully_corrected",
        }
        if self.normalization not in allowed_normalizations:
            raise ValueError(
                f"unknown normalization {self.normalization!r}; "
                f"allowed: {sorted(allowed_normalizations)}"
            )

    # ---- convenience --------------------------------------------------

    @property
    def sinogram_shape(self) -> tuple[int, int, int]:
        """Canonical sinogram shape (n_z_slices, n_angular_bins, n_radial_bins).

        This ordering matches what we used in the earlier visualization
        scripts: sinogram[slice, angle, radial].
        """
        return (self.n_z_slices, self.n_angular_bins, self.n_radial_bins)

    @property
    def energy_window_eV(self) -> tuple[float, float]:
        """Energy window in eV (MCGPU-PET's native unit)."""
        lo, hi = self.energy_window_keV
        return (lo * 1000.0, hi * 1000.0)

    def is_compatible_with(self, other: "Scanner") -> bool:
        """Check whether sinograms from two scanners are directly
        comparable bin-for-bin.

        Comparability requires identical sinogram shape AND identical
        forward-model assumptions (binning convention, TOF, normalization).
        Two scanners can share geometry but be incompatible if they
        disagree on indexing or corrections.
        """
        return (
            self.sinogram_shape == other.sinogram_shape
            and self.binning_convention == other.binning_convention
            and self.tof_enabled == other.tof_enabled
            and self.normalization == other.normalization
        )

    # ---- factories ----------------------------------------------------

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> "Scanner":
        """Construct a Scanner from a named preset, optionally overriding
        individual fields.

        Example:
            s = Scanner.from_preset("mcgpu_sample", acquisition_time_s=10)
            s = Scanner.from_preset("mcgpu_sample", name="my_custom_run")
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
        # YAML doesn't round-trip tuples cleanly; convert to lists and
        # convert back on load.
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

        # Convert the fields that should be tuples back from lists.
        tuple_fields = {
            "energy_window_keV",
            "crystal_size_mm",
            "crystals_per_module",
            "modules_per_rsector",
        }
        for key in tuple_fields:
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])

        # Filter out any unknown keys defensively.
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
            f"E=({self.energy_window_keV[0]}, {self.energy_window_keV[1]}) keV, "
            f"T={self.acquisition_time_s}s, "
            f"sino={self.sinogram_shape})"
        )


# =====================================================================
# Scanner presets
# =====================================================================
# These mirror the parameters seen in the sample simulations we have.
# Add more as they are needed.

SCANNER_PRESETS: dict[str, dict[str, Any]] = {
    # Mirrors the MCGPU-PET sample_simulation/MCGPU-PET.in file.
    # The sample is a toy geometry centered on a 9x9x9 cm phantom.
    "mcgpu_sample": {
        "name": "mcgpu_sample",
        "detector_radius_cm": 9.05,
        "detector_axial_length_cm": 12.656,
        "energy_window_keV": (350.0, 600.0),
        "energy_resolution": 0.12,
        "acquisition_time_s": 1.0,
        "n_radial_bins": 147,
        "n_angular_bins": 168,
        "n_z_slices": 1293,
        "span": 11,
        "max_ring_difference": 79,
        "binning_convention": "mcgpu_span11_mrd79",
        "tof_enabled": False,
        "normalization": "none",
    },

    # Mirrors gate-pet/bruker_pet_sim.py:
    #   - CylindricalPET, Rmax 82mm, Rmin 58mm, height 105mm
    #   - LYSO, 10x10x10 mm crystals, 8x8 per module,
    #   - 3 modules per rsector (axial), 8 rsectors (ring)
    #   - Energy window 350-650 keV, coincidence 10 ns
    # Sinogram parameters are placeholders for now; tune as needed when the
    # GATE backend actually builds sinograms.
    "bruker_albira": {
        "name": "bruker_albira",
        "detector_radius_cm": 5.8,
        "detector_axial_length_cm": 10.5,
        "energy_window_keV": (350.0, 650.0),
        "energy_resolution": 0.12,
        "acquisition_time_s": 1.0,
        "n_radial_bins": 128,
        "n_angular_bins": 128,
        "n_z_slices": 63,
        "span": 3,
        "max_ring_difference": 31,
        "binning_convention": "gate_span3_mrd31",
        "tof_enabled": False,
        "normalization": "none",
        "coincidence_window_ns": 10.0,
        "crystal_size_mm": (10.0, 6.0, 6.0),
        "crystal_material": "LYSO",
        "crystals_per_module": (8, 8),
        "modules_per_rsector": (1, 1, 3),
        "n_rsectors": 8,
    },
}