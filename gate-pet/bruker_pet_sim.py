"""
Bruker Small Animal PET Simulation - GATE 10
=============================================
Based on opengate/contrib/pet/siemensbiograph.py and test098_coincidence_actor.py

Scanner specs (from Kishore handover):
  - CylindricalPET, Rmax 82mm, Rmin 58mm, height 105mm
  - LYSO crystals, 10x10x10mm, 8x8 array per module
  - 3 modules per rsector (axial), 8 rsectors (ring)
  - Energy window: 350-650 keV
  - Coincidence window: 10 ns
  - Source: back-to-back 511 keV gammas, 1 MBq, 1 second

Usage:
    uv run python bruker_pet_sim.py
"""

import os
import opengate as gate
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition


def build_simulation():
    sim = gate.Simulation()

    # ── Units ──────────────────────────────────────────────────────────────
    m   = gate.g4_units.m
    mm  = gate.g4_units.mm
    keV = gate.g4_units.keV
    Bq  = gate.g4_units.Bq
    s   = gate.g4_units.s

    # ── Global settings ────────────────────────────────────────────────────
    sim.output_dir            = "./output"
    sim.random_seed           = 12345
    sim.number_of_threads     = 1
    sim.visu                  = False
    sim.check_volumes_overlap = False

    # ── Material database ──────────────────────────────────────────────────
    # GateMaterials.db must be in the working directory
    sim.volume_manager.add_material_database("GateMaterials.db")

    # ── World ──────────────────────────────────────────────────────────────
    world          = sim.world
    world.size     = [1 * m, 1 * m, 1 * m]
    world.material = "G4_AIR"

    # ── Scanner envelope ───────────────────────────────────────────────────
    scanner          = sim.add_volume("Tubs", "scanner")
    scanner.rmax     = 82  * mm
    scanner.rmin     = 58  * mm
    scanner.dz       = 105 * mm / 2
    scanner.material = "G4_AIR"

    # ── rsector: 8 around the ring ─────────────────────────────────────────
    rsector          = sim.add_volume("Box", "rsector")
    rsector.mother   = scanner.name
    rsector.size     = [10.5 * mm, 50.5 * mm, 95.0 * mm]
    rsector.material = "G4_AIR"
    t, r = get_circular_repetition(8, [67.0 * mm, 0, 0], axis=[0, 0, 1])
    rsector.translation = t
    rsector.rotation    = r

    # ── module: 3 axially per rsector ──────────────────────────────────────
    module          = sim.add_volume("Box", "module")
    module.mother   = rsector.name
    module.size     = [10.5 * mm, 50.5 * mm, 50.5 * mm]
    module.material = "G4_AIR"
    module.translation = get_grid_repetition([1, 1, 3], [0, 0, 32.0 * mm])

    # ── crystal: 8x8 per module ────────────────────────────────────────────
    crystal          = sim.add_volume("Box", "crystal")
    crystal.mother   = module.name
    crystal.size     = [10.0 * mm, 6.0 * mm, 6.0 * mm]
    crystal.material = "LYSO"
    crystal.translation = get_grid_repetition([1, 8, 8], [0, 6.3 * mm, 6.3 * mm])

    # ── Physics ────────────────────────────────────────────────────────────
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)

    # ── Digitizer ──────────────────────────────────────────────────────────
    output_file = "bruker_pet_output.root"

    # Hits: raw energy deposits in crystals
    hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
    hc.attached_to = crystal.name
    hc.authorize_repeated_volumes = True
    hc.output_filename = output_file
    hc.attributes = [
        "EventID",
        "PostPosition",
        "TotalEnergyDeposit",
        "PreStepUniqueVolumeID",
        "GlobalTime",
    ]

    # Singles: sum deposits per crystal block
    sc = sim.add_actor("DigitizerAdderActor", "Singles")
    sc.attached_to = hc.attached_to
    sc.authorize_repeated_volumes = True
    sc.input_digi_collection = hc.name
    sc.policy = "EnergyWeightedCentroidPosition"
    sc.group_volume = crystal.name
    sc.output_filename = output_file

    # Energy window: 350-650 keV
    ew = sim.add_actor("DigitizerEnergyWindowsActor", "EnergyWindow")
    ew.attached_to = hc.attached_to
    ew.authorize_repeated_volumes = True
    ew.input_digi_collection = sc.name
    ew.output_filename = output_file
    ew.channels = [{"name": "peak511", "min": 350 * keV, "max": 650 * keV}]

    # Coincidences: 10 ns window
    cc = sim.add_actor("CoincidenceSorterActor", "Coincidences")
    cc.input_digi_collection = "peak511"
    cc.window = 10e-9 * s
    cc.output_filename = output_file

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.output_filename = "stats.txt"

    # ── Source ─────────────────────────────────────────────────────────────
    # back_to_back: emits two collinear 511 keV gammas — the correct PET source
    source = sim.add_source("GenericSource", "source")
    source.particle             = "back_to_back"
    source.activity             = 1e6 * Bq
    source.position.type        = "point"
    source.position.translation = [0, 0, 0]
    source.direction.type       = "iso"

    # ── Timing: 1 second for sanity check ─────────────────────────────────
    sim.run_timing_intervals = [[0, 1 * s]]

    return sim


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    sim = build_simulation()
    # start_new_process=True is required when running multiple times or in interactive mode
    output = sim.run(start_new_process=True)
    print(output)
