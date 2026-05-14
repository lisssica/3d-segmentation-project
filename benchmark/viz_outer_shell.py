"""Interactive 3D viewer for outer/inner shell separation.

Usage:
  python -m benchmark.viz_outer_shell <assembly_id> [--mode outer-inner|outer|inner]

Modes:
  outer-inner (default) — both faces, outer = blue, inner = red (semi-transparent)
  outer                  — only outer faces (blue)
  inner                  — only inner faces (red)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

from .config import DATASET_DIR, LABELED_DIR, OUTER_SHELLS_DIR


COLOR_OUTER = np.array([50, 100, 255, 255], dtype=np.uint8)
COLOR_INNER = np.array([220, 60, 60, 255], dtype=np.uint8)


def _resolve_paths(assembly_id):
    """Search dataset/ first (new layout), then outer_shells/ (legacy)."""
    candidates = [
        (DATASET_DIR / assembly_id / "combined_mesh.obj",
         DATASET_DIR / assembly_id / "outer_face_mask.npy"),
        (LABELED_DIR / assembly_id / "combined_mesh.obj",
         OUTER_SHELLS_DIR / assembly_id / "outer_face_mask.npy"),
    ]
    for mesh_path, mask_path in candidates:
        if mesh_path.exists() and mask_path.exists():
            return mesh_path, mask_path
    return candidates[0]


def load_colored_mesh(assembly_id, mode="outer-inner"):
    mesh_path, mask_path = _resolve_paths(assembly_id)
    if not mesh_path.exists():
        raise FileNotFoundError(f"{mesh_path} not found — run benchmark.build_dataset first")
    if not mask_path.exists():
        raise FileNotFoundError(f"{mask_path} not found — run benchmark.build_dataset first")

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, skip_materials=True)
    outer = np.load(mask_path).astype(bool)

    if mode == "outer":
        keep = outer
    elif mode == "inner":
        keep = ~outer
    elif mode in ("outer-inner", "all"):  # 'all' kept for backward-compat
        keep = np.ones_like(outer)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if not keep.all():
        mesh = mesh.submesh([np.where(keep)[0]], append=True)
        outer = outer[keep]

    face_colors = np.empty((mesh.faces.shape[0], 4), dtype=np.uint8)
    face_colors[outer] = COLOR_OUTER
    face_colors[~outer] = COLOR_INNER
    mesh.visual.face_colors = face_colors
    return mesh, outer


def main():
    p = argparse.ArgumentParser(description="Interactive 3D viewer for outer/inner shells")
    p.add_argument("assembly_id", help="assembly id (e.g. 16550_e88d6986)")
    p.add_argument("--mode", choices=["outer-inner", "outer", "inner"],
                   default="outer-inner",
                   help="outer-inner = both coloured (default), outer = only outer, inner = only inner")
    args = p.parse_args()

    mesh, outer = load_colored_mesh(args.assembly_id, mode=args.mode)
    print(f"Assembly: {args.assembly_id}")
    print(f"  Faces shown: {mesh.faces.shape[0]:,}  "
          f"(outer in selection: {int(outer.sum()):,} / {len(outer):,} = {outer.mean()*100:.1f}%)")
    print(f"  Mode: {args.mode}")
    print("Opening trimesh viewer — drag to rotate, scroll to zoom, 'q' to quit.")

    scene = trimesh.Scene([mesh])
    # smooth=False → flat shading, чтобы цвет каждой грани не размывался
    # gouraud-интерполяцией между вершинами
    scene.show(smooth=False)


if __name__ == "__main__":
    main()
