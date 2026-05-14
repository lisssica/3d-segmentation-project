"""Visualize a single frame from a built dataset.

Usage:
  python -m benchmark.viz_frame <assembly_id> --frame N --mode 2d|3d

  2d:  matplotlib 3-panel (normals_camera, depth, seg_mask) — static image
  3d:  trimesh.Scene with faces visible in this frame coloured by their label,
       all other faces shown grey (label=0) — interactive viewer.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

import numpy as np

from .config import DATASET_DIR


def _frame_path(assembly_dir, frame_num):
    p = assembly_dir / "frames" / f"frame_{frame_num:04d}.npz"
    if not p.exists():
        # Try without zero-padding (or different number of digits)
        candidates = sorted((assembly_dir / "frames").glob(f"frame_*{frame_num}.npz"))
        if candidates:
            p = candidates[0]
    if not p.exists():
        raise FileNotFoundError(f"frame {frame_num} not found in {assembly_dir/'frames'}")
    return p


def show_2d(frame_npz, assembly_id, frame_num):
    """3-panel matplotlib: normals_camera / depth / seg_mask."""
    matplotlib.use("TkAgg") if matplotlib.get_backend() == "agg" else None
    import matplotlib.pyplot as plt

    data = np.load(frame_npz)
    mask = data["pix_to_face"] >= 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    normals = (data["normals_camera"] + 1) * 0.5
    normals[~mask] = 0.5
    axes[0].imshow(normals)
    axes[0].set_title("normals_camera")

    if mask.any():
        depth = data["depth"]
        depth_vis = np.where(mask, depth, np.nan)
        im = axes[1].imshow(depth_vis, cmap="viridis")
        fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02, label="camera-z")
    else:
        axes[1].imshow(data["depth"], cmap="viridis")
    axes[1].set_title("depth (linear camera-space z)")

    seg = data["seg_mask"]
    seg_vis = np.ma.masked_where(seg < 0, seg)
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad(color="black")
    axes[2].imshow(seg_vis, cmap=cmap)
    n_lbl = int(seg[mask].max()) if mask.any() and seg[mask].size else 0
    axes[2].set_title(f"seg_mask ({n_lbl} labels visible)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{assembly_id} — frame {frame_num}")
    fig.tight_layout()
    plt.show()


def show_3d(frame_npz, mesh_path, face_labels_path, assembly_id, frame_num):
    """Trimesh viewer: faces hit by pix_to_face get their original seg label
    colour, all others are dim grey. Flat shading — colour = label, no gradient."""
    import trimesh
    import matplotlib.pyplot as plt

    data = np.load(frame_npz)
    pix = data["pix_to_face"]
    visible_face_ids = np.unique(pix[pix >= 0])

    face_labels = np.load(face_labels_path).astype(np.int32)
    n_faces = face_labels.shape[0]
    all_labels = sorted(set(int(l) for l in face_labels.tolist()))

    # Pick a colour per global label so the same component gets the same colour
    # regardless of which frame we look at. tab20 has 20 distinct hues.
    cmap = plt.get_cmap("tab20")
    label_color = {}
    for i, lbl in enumerate(all_labels):
        rgba = cmap(i % 20)
        label_color[lbl] = np.array([rgba[0] * 255, rgba[1] * 255, rgba[2] * 255, 255],
                                     dtype=np.uint8)

    # Default: all faces are dim grey (opaque, so no alpha-blending → no "gradient")
    face_colors = np.full((n_faces, 4), [110, 110, 110, 255], dtype=np.uint8)

    # Visible faces: paint with their label colour (single solid colour per face)
    if visible_face_ids.size:
        valid_ids = visible_face_ids[
            (visible_face_ids >= 0) & (visible_face_ids < n_faces)
        ]
        for fid in valid_ids:
            face_colors[fid] = label_color[int(face_labels[fid])]

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, skip_materials=True)
    mesh.visual.face_colors = face_colors

    visible_labels = sorted({int(face_labels[fid]) for fid in valid_ids}) if visible_face_ids.size else []
    print(f"Assembly: {assembly_id}  frame: {frame_num}")
    print(f"  Faces visible in this frame: {visible_face_ids.size:,} / {n_faces:,}"
          f" ({visible_face_ids.size/n_faces*100:.1f}%)")
    print(f"  Visible labels:               {visible_labels}")
    print("Opening trimesh viewer (flat shading) — drag to rotate, scroll to zoom, 'q' to quit.")

    scene = trimesh.Scene([mesh])
    # smooth=False disables gouraud/per-vertex interpolation → solid colour per face
    scene.show(smooth=False)


def main():
    ap = argparse.ArgumentParser(description="Visualize a single frame from the dataset.")
    ap.add_argument("assembly_id", help="assembly id (e.g. 16550_e88d6986)")
    ap.add_argument("--frame", type=int, required=True,
                    help="frame number (1-indexed)")
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d",
                    help="2d=matplotlib image, 3d=trimesh viewer with frame's faces highlighted")
    args = ap.parse_args()

    assembly_dir = DATASET_DIR / args.assembly_id
    if not assembly_dir.exists():
        sys.exit(f"Error: {assembly_dir} not found — run benchmark.build_dataset first")

    frame_npz = _frame_path(assembly_dir, args.frame)

    if args.mode == "2d":
        show_2d(frame_npz, args.assembly_id, args.frame)
    else:
        mesh_path = assembly_dir / "combined_mesh.obj"
        face_labels_path = assembly_dir / "face_labels.npy"
        if not mesh_path.exists() or not face_labels_path.exists():
            sys.exit(f"Error: combined_mesh.obj or face_labels.npy missing in {assembly_dir}")
        show_3d(frame_npz, mesh_path, face_labels_path, args.assembly_id, args.frame)


if __name__ == "__main__":
    main()
