#!/usr/bin/env python3
import sys
import json
from pathlib import Path

import numpy as np
import trimesh

assembly = sys.argv[1]
base = Path("preprocessed_data") / assembly

mesh = trimesh.load(str(base / "combined_mesh.obj"), force="mesh", process=False)
labels = np.load(base / "face_labels.npy")
inst_map = json.loads((base / "instance_map.json").read_text())

rng = np.random.default_rng(42)
n = int(labels.max()) + 1
colors = rng.integers(60, 255, size=(n, 3), dtype=np.uint8)
colors[0] = [80, 80, 80]

face_colors = np.zeros((len(labels), 4), dtype=np.uint8)
face_colors[:, :3] = colors[labels]
face_colors[:, 3] = 255

mesh.visual.face_colors = face_colors

print(f"Assembly:  {assembly}")
print(f"Instances: {len(inst_map)}")
print(f"Faces:     {len(labels)}")
print("\nFirst 5 labels:")
for k in list(inst_map.keys())[:5]:
    print(f"  {k}: {inst_map[k]['name']}")

trimesh.Scene([mesh]).show()
