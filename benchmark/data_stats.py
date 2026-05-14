"""Distributions of n_bodies and n_triangles across all assemblies in data/.

Usage:
  SEG_env/bin/python -m benchmark.data_stats
"""
import csv
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DATA_DIR, DATA_STATS_DIR
from .label_mesh_fast import parse_assembly_json
from .utils import count_triangles_obj


def _stats_one(args):
    aid, path_str = args
    from pathlib import Path
    p = Path(path_str)
    pngs = list(p.glob("*.png"))
    n_bodies_png = len(pngs) - (1 if (p / "assembly.png").exists() else 0)
    try:
        n_bodies_json = parse_assembly_json(p)["n_bodies"]
    except Exception:
        n_bodies_json = None
    obj = p / "assembly.obj"
    n_tri = count_triangles_obj(obj) if obj.exists() else None
    return aid, n_bodies_png, n_bodies_json, n_tri


def collect_stats():
    items = [(p.name, str(p)) for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for aid, n_png, n_json, n_tri in ex.map(_stats_one, items):
            rows.append({
                "assembly_id": aid,
                "n_bodies_png": n_png,
                "n_bodies_json": n_json,
                "n_triangles": n_tri,
            })
    rows.sort(key=lambda r: r["assembly_id"])
    return rows


def summary_stats(values, label):
    arr = np.array([v for v in values if v is not None and v > 0])
    if len(arr) == 0:
        print(f"  {label}: no valid values")
        return None
    print(f"\n  {label} — n={len(arr)}")
    print(f"    min={arr.min():>9}  p25={np.percentile(arr,25):>9.0f}  "
          f"median={np.median(arr):>9.0f}  mean={arr.mean():>9.0f}")
    print(f"    p75={np.percentile(arr,75):>9.0f}  p95={np.percentile(arr,95):>9.0f}  "
          f"max={arr.max():>9}")
    return arr


def bucketed(arr, bins, labels):
    counts = []
    for lo, hi in bins:
        mask = (arr >= lo) & (arr < hi)
        counts.append(int(mask.sum()))
    total = len(arr)
    width = max(len(s) for s in labels)
    for lab, c in zip(labels, counts):
        pct = c / total * 100 if total else 0
        bar = "█" * int(round(pct / 2))
        print(f"    {lab:<{width}}  {c:>4} ({pct:>5.1f}%)  {bar}")


def _hist_panel(ax, arr, label, color, log_y=False):
    bins = np.geomspace(max(1, arr.min()), arr.max() + 1, 40)
    ax.hist(arr, bins=bins, color=color, alpha=0.8, edgecolor="white")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.axvline(np.median(arr), color="C3", linestyle="--",
               label=f"median={np.median(arr):.0f}")
    ax.axvline(np.percentile(arr, 95), color="C1", linestyle=":",
               label=f"p95={np.percentile(arr,95):.0f}")
    ax.set_xlabel(label + " (log)")
    ax.set_ylabel("count" + (" (log)" if log_y else ""))
    ax.set_title(f"{label} — {len(arr)} assemblies")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def draw_distributions(rows, out_path):
    bodies_png = np.array([r["n_bodies_png"] for r in rows
                            if r["n_bodies_png"] is not None and r["n_bodies_png"] > 0])
    bodies_json = np.array([r["n_bodies_json"] for r in rows
                             if r["n_bodies_json"] is not None and r["n_bodies_json"] > 0])
    tris = np.array([r["n_triangles"] for r in rows
                      if r["n_triangles"] is not None and r["n_triangles"] > 0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _hist_panel(axes[0], bodies_png, "n_bodies (PNG)", "C0", log_y=True)
    _hist_panel(axes[1], bodies_json, "n_bodies (JSON)", "C4", log_y=True)
    _hist_panel(axes[2], tris, "n_triangles in assembly.obj", "C2", log_y=False)

    fig.suptitle("Dataset distributions across data/")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    DATA_STATS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {DATA_DIR} ...")
    rows = collect_stats()
    print(f"Scanned {len(rows)} assemblies.")

    # Save CSV
    csv_path = DATA_STATS_DIR / "stats.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["assembly_id", "n_bodies_png", "n_bodies_json", "n_triangles"])
        w.writeheader()
        w.writerows(rows)
    print(f"  → {csv_path}")

    # Summary stats
    print("\n=== SUMMARY ===")
    bodies_png = summary_stats([r["n_bodies_png"] for r in rows], "n_bodies_png")
    bodies_json = summary_stats([r["n_bodies_json"] for r in rows], "n_bodies_json")
    tris_arr = summary_stats([r["n_triangles"] for r in rows], "n_triangles")

    # PNG vs JSON disagreement
    pairs = [(r["n_bodies_png"], r["n_bodies_json"]) for r in rows
             if r["n_bodies_png"] is not None and r["n_bodies_json"] is not None]
    if pairs:
        diffs = [p - j for p, j in pairs]
        eq = sum(1 for d in diffs if d == 0)
        print(f"\n  PNG vs JSON agreement: {eq}/{len(pairs)} match ({eq/len(pairs)*100:.1f}%)")
        if eq < len(pairs):
            diff_arr = np.array(diffs)
            print(f"    diff (png-json) median={np.median(diff_arr):.0f}  "
                  f"mean={diff_arr.mean():.2f}  "
                  f"min={diff_arr.min()}  max={diff_arr.max()}")

    # Subset for 2-10 bodies (by JSON, the source of truth)
    if bodies_json is not None:
        sub = [r for r in rows if r["n_bodies_json"] is not None
               and 2 <= r["n_bodies_json"] <= 10]
        print(f"\n  Subset for filter 2 ≤ n_bodies_json ≤ 10: {len(sub)} assemblies")

    # Bucketed tables
    if bodies_json is not None:
        print("\n  n_bodies_json distribution (source of truth):")
        bins_b = [(1, 2), (2, 6), (6, 11), (11, 51), (51, 101), (101, 501), (501, 1e9)]
        labels_b = ["1", "2-5", "6-10", "11-50", "51-100", "101-500", "500+"]
        bucketed(bodies_json, bins_b, labels_b)

    if tris_arr is not None:
        print("\n  n_triangles distribution:")
        bins_t = [(1, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e10)]
        labels_t = ["< 1k", "1k–10k", "10k–100k", "100k–1M", "≥ 1M"]
        bucketed(tris_arr, bins_t, labels_t)

    # Draw
    png_path = DATA_STATS_DIR / "distributions.png"
    draw_distributions(rows, png_path)
    print(f"\n  → {png_path}")


if __name__ == "__main__":
    main()
