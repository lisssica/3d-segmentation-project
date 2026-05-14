"""Build a dataset of assemblies filtered by n_bodies ∈ [MIN, MAX].

Pipeline:
  A. parse_assembly_json for every assembly in data/   (cheap, ~30 s)
  B. filter by n_bodies (default 2..10)
  C. build_combined for each selected (reuse parsed JSON)
  D. extract_outer_shell to convergence, save all frames as .npz
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import (
    DATA_DIR,
    DATASET_DIR,
    DATASET_MAX_BODIES,
    DATASET_MIN_BODIES,
    LOGS_DIR,
    OUTER_CHECK_EVERY,
    OUTER_MAX_VIEWS,
    OUTER_TOL,
)
from .label_mesh_fast import build_combined, parse_assembly_json
from .outer_shell import extract_outer_shell, save_outer_artifacts
from .render_bench import init_moderngl_context


# parse_assembly_json result has numpy arrays — can't directly pickle across
# pool workers cheaply, but inside one worker numpy is fine. We collect just
# (assembly_id, n_bodies) for filtering; the heavy parse runs again in the
# main process for selected ids only.
def _count_bodies_one(args):
    aid, path_str = args
    try:
        n = parse_assembly_json(Path(path_str))["n_bodies"]
    except Exception:
        n = None
    return aid, n


def list_and_count_bodies():
    items = [(p.name, str(p)) for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    counts = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for aid, n in ex.map(_count_bodies_one, items):
            counts[aid] = n
    return counts


def forecast(selected_ids, n_bodies_map):
    """Coarse forecast using already-known per-assembly samples and regression.
    See report/regression.json. Returns dict with totals."""
    reg_path = Path("benchmark/report/regression.json")
    tri_csv = Path("benchmark/report/triangles_assembly_obj.csv")
    if not (reg_path.exists() and tri_csv.exists()):
        return None
    reg = json.loads(reg_path.read_text())
    models = reg["models"]
    k_corr = float(reg["k_correction"])
    lm = reg.get("label_mesh_model")

    def _pred(model, n):
        if model is None:
            return 0.0
        c = model["chosen"]
        if c == "linear":
            return max(0.0, model["linear"]["a"] + model["linear"]["b"] * n)
        return max(0.0, model["power"]["c"] * max(n, 1) ** model["power"]["k"])

    n_tri_map = {}
    with open(tri_csv) as fp:
        for r in csv.DictReader(fp):
            if r["status"] == "ok":
                n_tri_map[r["assembly_id"]] = int(r["n_triangles"])

    # Use median observed n_frames-to-convergence from earlier outer_shells run.
    # Fallback to 100 if not available.
    summary_csv = Path("benchmark/outer_shells/summary.csv")
    n_frames_median = 100
    if summary_csv.exists():
        rows = list(csv.DictReader(open(summary_csv)))
        vals = [int(r["n_frames"]) for r in rows if r.get("converged") in ("True", "true", "1")]
        if vals:
            n_frames_median = int(np.median(vals))

    t_label = 0.0
    t_render = 0.0
    for aid in selected_ids:
        n_tri = n_tri_map.get(aid, 0)
        n_corr = n_tri * k_corr
        t_label += _pred(lm, n_tri)
        t_render += _pred(models["t_load"], n_corr) + n_frames_median * _pred(
            models["t_per_frame"], n_corr
        )

    bytes_per_frame = 200_000  # ~200 KB compressed, conservative
    size_bytes = len(selected_ids) * n_frames_median * bytes_per_frame
    return {
        "n_selected": len(selected_ids),
        "n_frames_median_est": n_frames_median,
        "t_label_mesh_sec": t_label,
        "t_outer_shell_sec": t_render,
        "t_total_sec": t_label + t_render,
        "t_total_min": (t_label + t_render) / 60,
        "size_gb_est": size_bytes / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bodies", type=int, default=DATASET_MIN_BODIES)
    ap.add_argument("--max-bodies", type=int, default=DATASET_MAX_BODIES)
    ap.add_argument("--save-every", type=int, default=1,
                    help="save every Nth frame (default 1 = all)")
    ap.add_argument("--tol", type=float, default=OUTER_TOL,
                    help="convergence threshold for outer-shell")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N selected assemblies (for testing)")
    args = ap.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── A. Count bodies in JSON for ALL assemblies ──
    print("A. Parsing assembly.json for all assemblies in data/ ...")
    t0 = time.perf_counter()
    counts = list_and_count_bodies()
    t_parse = time.perf_counter() - t0
    print(f"   parsed {len(counts)} in {t_parse:.1f}s")

    # ── B. Filter ──
    selected_ids = sorted(
        aid for aid, n in counts.items()
        if n is not None and args.min_bodies <= n <= args.max_bodies
    )
    if args.limit:
        selected_ids = selected_ids[: args.limit]
    print(f"\nB. Selected {len(selected_ids)} assemblies "
          f"with {args.min_bodies} ≤ n_bodies ≤ {args.max_bodies}")
    (DATASET_DIR / "selected.json").write_text(json.dumps({
        "min_bodies": args.min_bodies,
        "max_bodies": args.max_bodies,
        "n_selected": len(selected_ids),
        "selected": selected_ids,
        "counts": {aid: counts[aid] for aid in selected_ids},
    }, indent=2))

    # ── Pre-flight forecast ──
    fc = forecast(selected_ids, counts)
    if fc is not None:
        (DATASET_DIR / "time_forecast.json").write_text(json.dumps(fc, indent=2))
        print(f"\nForecast:")
        print(f"  t_label_mesh:   {fc['t_label_mesh_sec']/60:>6.1f} min")
        print(f"  t_outer_shell:  {fc['t_outer_shell_sec']/60:>6.1f} min "
              f"(N_frames median ~{fc['n_frames_median_est']})")
        print(f"  TOTAL:          {fc['t_total_min']:>6.1f} min")
        print(f"  Disk est:       {fc['size_gb_est']:>6.2f} GB")

    # ── C. Build combined for each selected ──
    print(f"\nC. Building combined_mesh.obj for {len(selected_ids)} assemblies ...")
    t0 = time.perf_counter()
    label_log = LOGS_DIR / "build_dataset_label.log"
    summary_rows = []
    failed_label = []
    with open(label_log, "a") as logf:
        logf.write(f"\n=== Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for i, aid in enumerate(selected_ids, 1):
            out_dir = DATASET_DIR / aid
            try:
                parsed = parse_assembly_json(DATA_DIR / aid)
                t_lbl_start = time.perf_counter()
                stats = build_combined(parsed, out_dir)
                t_label = time.perf_counter() - t_lbl_start
                summary_rows.append({
                    "assembly_id": aid,
                    "n_bodies_json": parsed["n_bodies"],
                    "n_instances": stats["n_instances"],
                    "n_faces": stats["n_faces"],
                    "n_vertices": stats["n_vertices"],
                    "t_label_sec": t_label,
                    "t_outer_sec": None,
                    "n_frames": None,
                    "outer_pct": None,
                    "converged": None,
                })
                logf.write(f"[OK] {aid}  faces={stats['n_faces']}  t={t_label:.3f}\n")
            except Exception as e:
                failed_label.append(aid)
                logf.write(f"[FAIL] {aid}  {e}\n")
            if i % 50 == 0:
                print(f"   {i}/{len(selected_ids)}  ({time.perf_counter()-t0:.1f}s)")
    print(f"   done: {len(summary_rows)} ok, {len(failed_label)} failed, "
          f"{time.perf_counter()-t0:.1f}s")

    # ── D. Outer-shell with frame saving ──
    print(f"\nD. Extracting outer-shell + saving frames (tol={args.tol}, "
          f"save_every={args.save_every}) ...")
    gl = init_moderngl_context()
    print(f"   ModernGL: {gl['info']}")
    t0 = time.perf_counter()
    bench_log = LOGS_DIR / "build_dataset_outer.log"
    failed_outer = []
    with open(bench_log, "a") as logf:
        logf.write(f"\n=== Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for i, row in enumerate(summary_rows, 1):
            aid = row["assembly_id"]
            out_dir = DATASET_DIR / aid
            try:
                result = extract_outer_shell(
                    gl, aid,
                    tol=args.tol,
                    check_every=OUTER_CHECK_EVERY,
                    max_views=OUTER_MAX_VIEWS,
                    labeled_root=DATASET_DIR,
                    use_fixed_dist=True,
                    save_frames_dir=out_dir / "frames",
                    save_every=args.save_every,
                )
                save_outer_artifacts(out_dir, result)
                row["t_outer_sec"] = result["t_total_sec"]
                row["n_frames"] = result["n_frames"]
                row["outer_pct"] = result["outer_pct"]
                row["converged"] = result["converged"]
                logf.write(
                    f"[OK] {aid}  frames={result['n_frames']}  "
                    f"outer={result['outer_pct']*100:.1f}%  "
                    f"t={result['t_total_sec']:.2f}\n"
                )
            except Exception as e:
                failed_outer.append(aid)
                logf.write(f"[FAIL] {aid}  {e}\n")
            if i % 25 == 0:
                print(f"   {i}/{len(summary_rows)}  ({time.perf_counter()-t0:.1f}s)")
    print(f"   done: {len(summary_rows)-len(failed_outer)} ok, "
          f"{len(failed_outer)} failed, {time.perf_counter()-t0:.1f}s")

    # ── Summary ──
    if summary_rows:
        with open(DATASET_DIR / "summary.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    (DATASET_DIR / "build_meta.json").write_text(json.dumps({
        "n_selected": len(selected_ids),
        "n_label_ok": len(summary_rows),
        "n_label_failed": len(failed_label),
        "n_outer_failed": len(failed_outer),
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    # Final disk usage report
    print(f"\nDataset built in {DATASET_DIR}")
    if summary_rows:
        ok_outer = [r for r in summary_rows if r.get("outer_pct") is not None]
        if ok_outer:
            t_total = sum(r["t_label_sec"] + r["t_outer_sec"] for r in ok_outer)
            outer_med = float(np.median([r["outer_pct"] for r in ok_outer]))
            frames_med = int(np.median([r["n_frames"] for r in ok_outer]))
            print(f"  Total wall time (label+outer):  {t_total:.1f} s = {t_total/60:.1f} min")
            print(f"  Median frames to convergence:   {frames_med}")
            print(f"  Median outer_pct:               {outer_med*100:.1f}%")


if __name__ == "__main__":
    main()
