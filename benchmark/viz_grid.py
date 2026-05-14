"""Build a small thumbnail grid of all frames (normals by default).

Usage:
  python -m benchmark.viz_grid <assembly_id> [--thumb 128] [--channel normals|depth|seg]
                                              [--out path.png]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATASET_DIR


def _frame_to_rgb(data, channel):
    mask = data["pix_to_face"] >= 0
    H, W = data["pix_to_face"].shape
    if channel == "normals":
        rgb = (data["normals_camera"] + 1.0) * 0.5
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[~mask] = 1.0  # white background
        return (rgb * 255).astype(np.uint8)
    if channel == "depth":
        depth = data["depth"]
        if mask.any():
            vmin, vmax = float(depth[mask].min()), float(depth[mask].max())
            if vmax - vmin < 1e-9:
                vmax = vmin + 1.0
            norm = np.zeros_like(depth, dtype=np.float32)
            norm[mask] = (depth[mask] - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(depth, dtype=np.float32)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("viridis")
        rgb = cmap(norm)[..., :3]
        rgb[~mask] = 1.0
        return (rgb * 255).astype(np.uint8)
    if channel == "seg":
        seg = data["seg_mask"]
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab20")
        # bring labels to 0..19
        vis = np.zeros((H, W, 3), dtype=np.float32)
        if mask.any():
            for lbl in np.unique(seg[mask]):
                if lbl < 0:
                    continue
                vis[seg == lbl] = cmap(int(lbl) % 20)[:3]
        vis[~mask] = 1.0
        return (vis * 255).astype(np.uint8)
    raise ValueError(f"unknown channel {channel!r}")


def make_grid(assembly_id, thumb_size=128, channel="normals", out_path=None):
    adir = DATASET_DIR / assembly_id
    frames_dir = adir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"{frames_dir} not found — run benchmark.build_dataset first")
    frames = sorted(frames_dir.glob("frame_*.npz"))
    if not frames:
        raise RuntimeError(f"no frame_*.npz in {frames_dir}")

    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.full((rows * thumb_size, cols * thumb_size, 3), 240, dtype=np.uint8)

    for i, fp in enumerate(frames):
        d = np.load(fp)
        rgb = _frame_to_rgb(d, channel)
        img = Image.fromarray(rgb).resize((thumb_size, thumb_size), Image.BILINEAR)
        arr = np.asarray(img)
        r, c = divmod(i, cols)
        canvas[r * thumb_size:(r + 1) * thumb_size,
               c * thumb_size:(c + 1) * thumb_size] = arr

    if out_path is None:
        out_path = adir / f"grid_{channel}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print(f"  {assembly_id}: {n} frames → {rows}×{cols} grid ({thumb_size}px each)")
    print(f"  → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render thumbnail grid of all frames.")
    ap.add_argument("assembly_id", nargs="?",
                    help="assembly id (e.g. 16550_e88d6986). If omitted with --all, process every assembly.")
    ap.add_argument("--thumb", type=int, default=128, help="thumbnail size in pixels (default 128)")
    ap.add_argument("--channel", choices=["normals", "depth", "seg"], default="normals",
                    help="which channel to render (default normals)")
    ap.add_argument("--out", type=str, default=None,
                    help="output PNG path (default: dataset/<id>/grid_<channel>.png)")
    ap.add_argument("--all", action="store_true",
                    help="process every assembly in dataset/ (--out is ignored)")
    args = ap.parse_args()

    if args.all:
        for aid_dir in sorted(DATASET_DIR.iterdir()):
            if aid_dir.is_dir() and (aid_dir / "frames").exists():
                try:
                    make_grid(aid_dir.name, thumb_size=args.thumb, channel=args.channel)
                except Exception as e:
                    print(f"  [FAIL] {aid_dir.name}: {e}")
    else:
        if not args.assembly_id:
            sys.exit("Error: provide assembly_id or use --all")
        make_grid(args.assembly_id, thumb_size=args.thumb,
                  channel=args.channel, out_path=args.out)


if __name__ == "__main__":
    main()
