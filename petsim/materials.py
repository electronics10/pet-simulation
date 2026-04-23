"""
Material registry: map human-readable material names to simulator-specific
definitions.

A user writes `Phantom.cylinder(material="brain")`. The backends need to
translate that into:

  - MCGPU-PET: a path to `brain_ICRP110_5-515keV.mcgpu.gz` and a nominal
    density in g/cm^3.
  - GATE: a material name defined in GateMaterials.db and a density.

This module owns that translation. Users interact only with the names;
backends use the registry to resolve names to concrete files at write
time.

The registry is populated from two sources:

  1. The ICRP 110 material zip bundled with MCGPU-PET
     (other_materials_ICRP110_5-515keV__MCGPU-PET.zip), which contains
     cross-section files for 17 common biological tissues.
  2. A small hard-coded table of GATE material names and densities for
     the same tissues.

Registering a material is idempotent — re-registering the same name with
the same definition is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Material:
    """A material definition spanning both simulators.

    Attributes
    ----------
    name : str
        Human-readable name used in the Python API (e.g. "brain").
    nominal_density : float
        Reference density in g/cm^3. Per-voxel density in a Phantom may
        deviate; this is the default used when the user doesn't specify one.
    mcgpu_file : str
        Filename (not path) of the .mcgpu.gz cross-section file for this
        material. Resolved against MaterialRegistry.mcgpu_materials_dir at
        write time.
    gate_name : str
        Name of this material in GateMaterials.db (Geant4 notation).
    """

    name: str
    nominal_density: float
    mcgpu_file: str
    gate_name: str


# =====================================================================
# Default materials
# =====================================================================
# Covers the files from MCGPU-PET's other_materials_ICRP110_5-515keV zip
# plus air and water. GATE names follow the G4_* or custom LYSO style used
# in the existing gate-pet/bruker_pet_sim.py.
#
# Nominal densities are from ICRP Publication 110, which is the standard
# reference for computational human phantoms in radiation dosimetry.
# Source: ICRP, 2009. Adult Reference Computational Phantoms. ICRP
# Publication 110. Ann. ICRP 39 (2).

_DEFAULT_MATERIALS: tuple[Material, ...] = (
    # Basics
    Material("air",   0.00120,  "air_5-515keV.mcgpu.gz",                       "G4_AIR"),
    Material("water", 1.00000,  "water_5-515keV.mcgpu.gz",                     "G4_WATER"),

    # ICRP 110 tissues
    Material("adipose",            0.95000, "adipose_ICRP110_5-515keV.mcgpu.gz",            "G4_ADIPOSE_TISSUE_ICRP"),
    Material("blood",              1.06000, "blood_spleen_ICRP110_5-515keV.mcgpu.gz",       "G4_BLOOD_ICRP"),
    Material("brain",              1.04000, "brain_ICRP110_5-515keV.mcgpu.gz",              "G4_BRAIN_ICRP"),
    Material("breast_glandular",   1.02000, "breast_glandular_ICRP110_5-515keV.mcgpu.gz",   "G4_BREAST_ICRP"),
    Material("cartilage",          1.10000, "cartilage_ICRP110_5-515keV.mcgpu.gz",          "G4_CARTILAGE_ICRP"),
    Material("eyes",               1.05000, "eyes_ICRP110_5-515keV.mcgpu.gz",               "G4_EYE_LENS_ICRP"),
    Material("glands",             1.03000, "glands_others_ICRP110_5-515keV.mcgpu.gz",      "G4_TISSUE_SOFT_ICRP"),
    Material("kidneys",            1.05000, "kidneys_ICRP110_5-515keV.mcgpu.gz",            "G4_KIDNEY_ICRP"),
    Material("liver",              1.06000, "liver_ICRP110_5-515keV.mcgpu.gz",              "G4_LIVER_ICRP"),
    Material("lung",               0.38500, "lung_ICRP110_5-515keV.mcgpu.gz",               "G4_LUNG_ICRP"),
    Material("muscle",             1.05000, "muscle_ICRP110_5-515keV.mcgpu.gz",             "G4_MUSCLE_STRIATED_ICRU"),
    Material("skin",               1.09000, "skin_ICRP110_5-515keV.mcgpu.gz",               "G4_SKIN_ICRP"),
    Material("soft_tissue",        1.03000, "soft_tissue_ICRP110_5-515keV.mcgpu.gz",        "G4_TISSUE_SOFT_ICRP"),
    Material("spongiosa",          1.03000, "spongiosa_ICRP110_5-515keV.mcgpu.gz",          "G4_BONE_CORTICAL_ICRP"),
    Material("stomach_intestines", 1.03000, "stomach_intestines_ICRP110_5-515keV.mcgpu.gz", "G4_TISSUE_SOFT_ICRP"),
)


# =====================================================================
# Registry
# =====================================================================


@dataclass
class MaterialRegistry:
    """A lookup table of materials, plus the directory where MCGPU-PET
    cross-section files live on disk.

    The registry is mutable — new materials can be registered at runtime.
    Default contents match the ICRP 110 tissues bundled with MCGPU-PET
    plus air and water.

    Attributes
    ----------
    mcgpu_materials_dir : Path
        Directory containing the .mcgpu.gz files. Backends use this plus
        the material's mcgpu_file to build the full path.
    materials : dict[str, Material]
        Name -> Material mapping. Keys are lowercased for case-insensitive
        lookup.
    """

    mcgpu_materials_dir: Path
    materials: dict[str, Material] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mcgpu_materials_dir = Path(self.mcgpu_materials_dir)
        if not self.materials:
            for m in _DEFAULT_MATERIALS:
                self.materials[m.name.lower()] = m

    # ---- lookup -------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name.lower() in self.materials

    def __getitem__(self, name: str) -> Material:
        try:
            return self.materials[name.lower()]
        except KeyError:
            raise KeyError(
                f"material {name!r} not in registry; available: "
                f"{sorted(self.materials.keys())}"
            ) from None

    def names(self) -> list[str]:
        """Return all registered material names, sorted."""
        return sorted(self.materials.keys())

    # ---- mutation -----------------------------------------------------

    def register(self, material: Material) -> None:
        """Add or replace a material. Re-registering the same Material
        (by value) is a silent no-op; registering a different definition
        for an existing name overwrites it with a warning.
        """
        key = material.name.lower()
        existing = self.materials.get(key)
        if existing is not None and existing != material:
            import warnings
            warnings.warn(
                f"overwriting material {material.name!r}: "
                f"{existing} -> {material}",
                stacklevel=2,
            )
        self.materials[key] = material

    # ---- MCGPU helpers -----------------------------------------------

    def mcgpu_path(self, name: str) -> Path:
        """Return the full filesystem path to the .mcgpu.gz file for
        a given material. Does not check that the file exists.
        """
        material = self[name]
        return self.mcgpu_materials_dir / material.mcgpu_file

    def verify_mcgpu_files(self, names: list[str] | None = None) -> list[str]:
        """Check that .mcgpu.gz files exist on disk for the named
        materials (or all registered materials if names is None).
        Returns a list of material names whose files are missing.
        """
        if names is None:
            names = self.names()
        missing = []
        for name in names:
            path = self.mcgpu_path(name)
            if not path.exists():
                missing.append(name)
        return missing

    # ---- GATE helpers -------------------------------------------------

    def gate_name(self, name: str) -> str:
        """Return the GATE/Geant4 material name for a given material."""
        return self[name].gate_name