# Introdcution

This project is created for positron emission tomography (PET) simulation, serving as an important part for the pipeline of PET scattering correction. There are differet ways and simulation software to achieve Monte Carlo PET simulation. The most standard one is through GATE (CPU only), based on GEANT4. 

**Bruker scanner specs (Lab Device)**

| Parameter           | Value               |
| ------------------- | ------------------- |
| Scanner geometry    | CylindricalPET      |
| Rmax / Rmin         | 82 mm / 58 mm       |
| Axial height        | 105 mm              |
| Crystal material    | LYSO                |
| Crystal size        | 10 × 10 × 10 mm     |
| Crystal array       | 8 × 8 per module    |
| Modules per rsector | 3 (axial)           |
| Rsectors            | 8 (transaxial ring) |
| Energy resolution   | 15% at 511 keV      |
| Energy window       | 350–650 keV         |
| Coincidence window  | 10 ns               |


#  Monte Carlo PET simulation environment setup

##  GATE 9.x (legacy, abandoned)

GATE 9.4.1 was used for Monte Carlo PET simulation. It runs on `.mac`  files. I chose Docker for simplicity and portability. Our intended simulation models a Bruker small animal PET scanner and produces coincidence data for later usage.

Docker image

```bash
docker pull opengatecollaboration/gate:9.4.1-docker
```

Smoke test:

```bash
docker run --rm opengatecollaboration/gate:9.4.1-docker Gate --version
# Expected: GATE version 9.4.1 / Geant4 11.3.0
```

The image's entrypoint script automatically does `cd /APP` and runs `Gate $1`, so the workflow is:

```bash
cd /path/to/your/macro/folder
docker run --rm -v $PWD:/APP opengatecollaboration/gate:9.4.1-docker macro_name.mac
```

`-v $PWD:/APP` mounts the current directory into the container at `/APP`, making your `.mac` files and `GateMaterials.db` visible to GATE. ROOT file written to `output/` relative to the macro directory. 

## GATE 10 (Python)

GATE 10 (`opengate`) is a Python-based Monte Carlo PET simulation framework built on Geant4. It replaces the `.mac` scripting of GATE 9. The simulation models a Bruker small animal PET scanner and produces coincidence data.

**Do not use GATE 9** (Docker-based, `.mac` files). The digitizer API changed in v9.3 and Kishore's macros are broken. GATE 10 is cleaner, Python-native, and integrates naturally with other projects.

```
uv init gate_sim --python 3.12
cd gate_sim
uv add opengate
mkdir output
```

To simulate the device:
```
# first run will download ~2GB of Geant4 data
uv run python bruker_pet_sim.py
```

# GATE 10 Simulation

## Simulation structure

A GATE 10 simulation has this order:

1. `gate.Simulation()` — main object
2. Materials database
3. Volumes (world → scanner → rsector → module → crystal)
4. Physics
5. Digitizer actors (Hits → Singles → EnergyWindow → Coincidences)
6. Stats actor
7. Source
8. `sim.run_timing_intervals`
9. `sim.run(start_new_process=True)`


## Volume repetition

GATE 10 uses `translation` and `rotation` lists to repeat volumes. Two utility functions handle this:

```python
from opengate.geometry.utility import get_grid_repetition, get_circular_repetition

# Grid repetition (axial modules, crystal arrays)
module.translation = get_grid_repetition([1, 1, 3], [0, 0, 32.0 * mm])
crystal.translation = get_grid_repetition([1, 8, 8], [0, 6.3 * mm, 6.3 * mm])

# Circular repetition (rsectors around the ring)
# Returns BOTH translations and rotations — assign both
t, r = get_circular_repetition(8, [67.0 * mm, 0, 0], axis=[0, 0, 1])
rsector.translation = t
rsector.rotation    = r
```

> `get_circular_repetition` returns a tuple `(translations, rotations)`. Forgetting to unpack and assign both will raise a `ValueError`.


## Digitizer chain

```python
# 1. Hits: raw energy deposits in crystals
hc = sim.add_actor("DigitizerHitsCollectionActor", "Hits")
hc.attached_to = crystal.name
hc.authorize_repeated_volumes = True
hc.output_filename = "output.root"
hc.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit",
                  "PreStepUniqueVolumeID", "GlobalTime"]

# 2. Singles: sum deposits per crystal block
sc = sim.add_actor("DigitizerAdderActor", "Singles")
sc.attached_to = hc.attached_to
sc.authorize_repeated_volumes = True
sc.input_digi_collection = hc.name
sc.policy = "EnergyWeightedCentroidPosition"
sc.group_volume = crystal.name
sc.output_filename = "output.root"

# 3. Energy window: 350-650 keV
ew = sim.add_actor("DigitizerEnergyWindowsActor", "EnergyWindow")
ew.attached_to = hc.attached_to
ew.authorize_repeated_volumes = True
ew.input_digi_collection = sc.name
ew.output_filename = "output.root"
ew.channels = [{"name": "peak511", "min": 350 * keV, "max": 650 * keV}]

# 4. Coincidences
cc = sim.add_actor("CoincidenceSorterActor", "Coincidences")
cc.input_digi_collection = "peak511"   # name of the energy window channel
cc.window = 10e-9 * s
cc.output_filename = "output.root"
```

### Critical rules

- Always use `attached_to`, never `mother`, for actors
- Always set `authorize_repeated_volumes = True` on every digitizer actor — without this, hits in repeated crystal volumes are silently ignored
- `CoincidenceSorterActor` takes the energy window **channel name** (`"peak511"`), not the actor name
- All actors writing to the same ROOT file share one `output_filename`


## Source

Use `back_to_back` for PET — it emits two collinear 511 keV gammas directly, which is physically correct and faster than simulating positron transport:

```python
source = sim.add_source("GenericSource", "source")
source.particle             = "back_to_back"
source.activity             = 1e6 * gate.g4_units.Bq
source.position.type        = "point"
source.position.translation = [0, 0, 0]
source.direction.type       = "iso"
```

For a more realistic simulation, use an ion source (e.g. F18) or a voxelized phantom source.


## Physics

```python
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
sim.physics_manager.set_production_cut("world", "all", 1 * mm)
```

`option3` is the standard choice for PET. For higher accuracy use `option4` (slower).


## Running

```python
# start_new_process=True is required when:
# - running multiple times in the same script
# - working in an interactive terminal or notebook
sim.run(start_new_process=True)
```


## Checking output

```python
import uproot
f = uproot.open("output/bruker_pet_output.root")
for key in f.keys():
    print(f"{key}: {f[key].num_entries} entries")
```

Expected output after 1 second of 1 MBq simulation:

```
Hits;1:        ~900k entries
Singles;1:     ~600k entries
peak511;1:     ~150k entries
Coincidences;1: ~10k entries
```

## Scattering Effect

The current simulation uses a point source with no phantom — no scatter is generated. To simulate scatter:

1. Add a cylindrical water phantom (or voxelized rat CT) as the activity medium
2. Switch source to a voxelized source confined to the phantom
3. Use `PreStepUniqueVolumeID` in hits to label scatter vs true coincidences
4. The scatter fraction becomes the training label for the DL model
