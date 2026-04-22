#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def label_to_color(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(123)
    unique = np.unique(labels)
    lut = {int(v): rng.integers(0, 255, size=3, dtype=np.uint8) for v in unique if v >= 0}

    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for k, col in lut.items():
        out[labels == k] = col
    out[labels < 0] = np.array([0, 0, 0], dtype=np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Show render grid with segmentation overlays")
    parser.add_argument("--input_dir", required=True, help="Step3 input_dir")
    parser.add_argument("--n", type=int, default=9, help="How many frames to display")
    args = parser.parse_args()

    render_dir = Path(args.input_dir) / "renders"
    with (render_dir / "frames_metadata.json").open("r", encoding="utf-8") as f:
        frames = json.load(f)

    n = min(args.n, len(frames))
    if n == 0:
        print("No frames to visualize")
        return

    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue

        frame = frames[i]
        img = np.array(Image.open(render_dir / frame["normals_world"]))
        seg = np.load(render_dir / frame["seg_mask"]).astype(np.int32)
        seg_rgb = label_to_color(seg)

        overlay = (0.65 * img + 0.35 * seg_rgb).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"frame {frame['frame_id']}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
