"""Strip benchmark/dataset/ down to git-friendly size.

For every assembly NOT in SAMPLES, deletes everything except 3 preview PNG
(assembly.png, grid_normals.png, grid_seg.png). Sample assemblies keep all
their data (combined_mesh.obj, face_labels.npy, frames/, etc.).

Usage:
  SEG_env/bin/python -m benchmark.prep_for_git           # dry-run
  SEG_env/bin/python -m benchmark.prep_for_git --apply   # actually delete
"""
import argparse
import shutil
import sys
from pathlib import Path

from .config import DATASET_DIR


SAMPLES = [
    "16550_e88d6986",   # 2 bodies, 8.6k tri (болт)
    "20281_a29f9a18",   # 4 bodies, 16.7k tri
    "20467_f1fcc009",   # 6 bodies, 29.2k tri
    "19518_f220b68a",   # 8 bodies, 18.1k tri
    "20322_5a8c6077",   # 10 bodies, 43.2k tri
]

KEEP_FOR_ALL = {"assembly.png", "grid_normals.png", "grid_seg.png"}


def folder_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def fmt_size(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete files (default is dry-run)")
    args = ap.parse_args()

    if not DATASET_DIR.exists():
        sys.exit(f"{DATASET_DIR} not found")

    before_total = folder_size(DATASET_DIR)
    print(f"Before:  {fmt_size(before_total)}  in {DATASET_DIR}")
    print(f"Samples to keep intact ({len(SAMPLES)}):")
    for s in SAMPLES:
        path = DATASET_DIR / s
        if path.exists():
            print(f"  ✓ {s}  ({fmt_size(folder_size(path))})")
        else:
            print(f"  ✗ {s}  (MISSING — sample folder not found)")

    cleaned = 0
    removed_bytes = 0
    for aid_dir in sorted(DATASET_DIR.iterdir()):
        if not aid_dir.is_dir():
            continue
        aid = aid_dir.name
        if aid in SAMPLES:
            continue

        for path in aid_dir.iterdir():
            if path.is_dir() and path.name == "frames":
                size = folder_size(path)
                if args.apply:
                    shutil.rmtree(path)
                removed_bytes += size
            elif path.is_file() and path.name not in KEEP_FOR_ALL:
                size = path.stat().st_size
                if args.apply:
                    path.unlink()
                removed_bytes += size
        cleaned += 1

    after_total = before_total - removed_bytes
    verb = "Removed" if args.apply else "Would remove"
    print(f"\n{verb} {fmt_size(removed_bytes)} from {cleaned} non-sample assemblies")
    print(f"After:   {fmt_size(after_total)}  in {DATASET_DIR}")
    if not args.apply:
        print("\n(dry-run; re-run with --apply to actually delete)")


if __name__ == "__main__":
    main()
