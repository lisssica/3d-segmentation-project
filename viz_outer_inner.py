#!/usr/bin/env python3
"""
Usage:
  python viz_outer_inner.py <assembly> outer    # только внешние
  python viz_outer_inner.py <assembly> inner    # только внутренние
  python viz_outer_inner.py <assembly> both     # вместе (outer=цветные, inner=красные)
"""
import sys
import json
import numpy as np
import trimesh
from pathlib import Path

assembly = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "both"
base = Path("preprocessed_data") / assembly

mesh = trimesh.load(str(base / "combined_mesh.obj"), force="mesh", process=False)
face_labels = np.load(base / "face_labels.npy")
outer_mask  = np.load(base / "outer_face_mask.npy")

rng = np.random.default_rng(42)
n_labels = int(face_labels.max()) + 1
palette = rng.integers(80, 240, size=(n_labels, 3), dtype=np.uint8)
palette[0] = [80, 80, 80]

def make_submesh(mask):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    old_face_ids = np.where(mask)[0]
    sub_faces = faces[old_face_ids]
    used = np.unique(sub_faces)
    v_map = np.full(len(verts), -1, dtype=np.int64)
    v_map[used] = np.arange(len(used))
    sub_verts = verts[used]
    sub_faces_new = v_map[sub_faces]
    labels_sub = face_labels[old_face_ids]
    return trimesh.Trimesh(vertices=sub_verts, faces=sub_faces_new, process=False), labels_sub

scene = trimesh.Scene()

if mode in ("outer", "both"):
    m, lbls = make_submesh(outer_mask)
    fc = np.zeros((len(lbls), 4), dtype=np.uint8)
    fc[:, :3] = palette[lbls]
    fc[:, 3]  = 255
    m.visual.face_colors = fc
    scene.add_geometry(m, node_name="outer")

if mode in ("inner", "both"):
    m, lbls = make_submesh(~outer_mask)
    fc = np.zeros((len(lbls), 4), dtype=np.uint8)
    if mode == "both":
        fc[:] = [180, 40, 40, 160]   # красноватые полупрозрачные
    else:
        fc[:, :3] = palette[lbls]
        fc[:, 3]  = 255
    m.visual.face_colors = fc
    scene.add_geometry(m, node_name="inner")

outer_n  = outer_mask.sum()
inner_n  = (~outer_mask).sum()
print(f"Assembly: {assembly}  mode={mode}")
print(f"  outer: {outer_n} граней ({outer_mask.mean():.1%})")
print(f"  inner: {inner_n} граней ({(~outer_mask).mean():.1%})")

scene.show()
