#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch3d.io import load_obj

assembly = sys.argv[1]
base = Path("preprocessed_data") / assembly

if not (base / "outer_face_mask.npy").exists():
    print(f"outer_face_mask.npy not found in {base}")
    print("Run: python step2_extract_outer_surface.py <assembly>")
    sys.exit(1)

verts, fi, _ = load_obj(str(base / "combined_mesh.obj"), load_textures=False)
faces = fi.verts_idx.numpy()
outer = np.load(base / "outer_face_mask.npy")
centroids = verts.numpy()[faces].mean(axis=1)

print(f"Assembly: {assembly}")
print(f"Outer:  {outer.sum()} граней  ({outer.mean():.1%})")
print(f"Hidden: {(~outer).sum()} граней  ({(~outer).mean():.1%})")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"{assembly}: outer={outer.mean():.1%}, hidden={(~outer).mean():.1%}")

views = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
for ax, (title, xi, yi) in zip(axes, views):
    ax.scatter(centroids[outer, xi], centroids[outer, yi],
               s=0.5, c="green", alpha=0.4, label=f"outer ({outer.sum()})")
    ax.scatter(centroids[~outer, xi], centroids[~outer, yi],
               s=1.5, c="red", alpha=0.7, label=f"hidden ({(~outer).sum()})")
    ax.set_title(title)
    ax.set_aspect("equal")
    if xi == 0 and yi == 1:
        ax.legend(markerscale=5, fontsize=8)

plt.tight_layout()
plt.show()
