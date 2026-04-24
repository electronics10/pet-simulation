"""Backends for simulating petsim scenes with external simulators.

Each backend takes the simulator-agnostic Run (phantom + source + scanner)
and translates it into the inputs that a specific simulator accepts, runs
the simulator, and parses the outputs back into a Sinogram.

Currently available:

  - mcgpu: MCGPU-PET (GPU Monte Carlo, fast, CUDA required)
      - write_vox, write_in: input-file writers
      - MCGPUBackend: full runtime (stage + invoke + parse)

  - gate:  GATE 10 via the opengate Python API (CPU/GPU, realistic)
      - GATEBackend: full runtime (build + run + parse)
"""

from .mcgpu import MCGPUConfig, write_in, write_vox
from .mcgpu_runtime import MCGPUBackend, MCGPURunResult
from .gate import GATEBackend, GATEConfig, GATERunResult

__all__ = [
    # mcgpu input writers
    "write_vox", "write_in", "MCGPUConfig",
    # mcgpu runtime
    "MCGPUBackend", "MCGPURunResult",
    # gate runtime
    "GATEBackend", "GATEConfig", "GATERunResult",
]