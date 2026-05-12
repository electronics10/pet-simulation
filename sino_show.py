import numpy as np
import matplotlib.pyplot as plt
from petsim import Sinogram

sino = Sinogram.load("./runs/gate_0001/sinogram.npz")
trues_2d = sino.trues.sum(axis=0)
scatter_2d = sino.scatter.sum(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(trues_2d, aspect='auto', cmap='hot')
axes[0].set_title(f"Trues ({sino.total_trues:,} counts)")
axes[1].imshow(scatter_2d, aspect='auto', cmap='hot')
axes[1].set_title(f"Scatter ({sino.total_scatter:,} counts, SF={sino.scatter_fraction:.1%})")
plt.tight_layout()
plt.savefig("./gate_sinograms.png", dpi=150)
# plt.show()