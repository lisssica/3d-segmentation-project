#!/usr/bin/env python3
"""
Usage:
  python viz_renders.py <assembly> [n_samples=6] [data_dir=preprocessed_data]
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

assembly = sys.argv[1]
n        = int(sys.argv[2])   if len(sys.argv) > 2 else 6
data_dir = sys.argv[3]        if len(sys.argv) > 3 else "preprocessed_data"

renders_dir = Path(data_dir) / assembly / "renders"
files = sorted(renders_dir.glob("*.npz"))
print(f"{assembly}: {len(files)} frames")

idx = np.linspace(0, len(files)-1, n, dtype=int)

palette = np.random.default_rng(42).integers(60, 240, (300, 3), dtype=np.uint8)
palette[0] = [20, 20, 20]

fig, axes = plt.subplots(n, 3, figsize=(12, 3*n))
fig.suptitle(assembly, fontsize=14)

for row, i in enumerate(idx):
    d   = np.load(files[i])
    nc  = d["normals_camera"]
    dep = d["depth_camera"]
    seg = d["seg_mask"]

    dep_vis = np.zeros_like(dep)
    valid = dep > 0
    if valid.any():
        v = dep[valid]
        dep_vis[valid] = (v - v.min()) / (v.max() - v.min() + 1e-8)

    seg_rgb = palette[np.clip(seg, 0, 299)]
    seg_rgb[seg < 0] = 0

    axes[row, 0].imshow(nc)
    axes[row, 0].set_ylabel(f"frame {i+1}", fontsize=8)
    axes[row, 0].set_title("normals_camera" if row == 0 else "")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(dep_vis, cmap="plasma")
    axes[row, 1].set_title("depth_camera" if row == 0 else "")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(seg_rgb)
    axes[row, 2].set_title("seg_mask" if row == 0 else "")
    axes[row, 2].axis("off")

plt.tight_layout()
out = Path(data_dir) / assembly / "viz_renders.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
