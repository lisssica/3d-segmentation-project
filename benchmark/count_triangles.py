"""Step 4: count triangles in 754 assembly.obj + 10 combined_mesh.obj."""
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import DATA_DIR, LABELED_DIR, REPORT_DIR
from .utils import count_triangles_obj


def _count_one(args):
    aid, path = args
    p = Path(path)
    if not p.exists():
        return aid, None, 0, "missing"
    try:
        n = count_triangles_obj(p)
        return aid, n, p.stat().st_size, "ok"
    except Exception as e:
        return aid, None, p.stat().st_size if p.exists() else 0, f"parse_error: {e}"


def count_all_assembly_obj():
    items = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "assembly.json").exists():
            items.append((d.name, str(d / "assembly.obj")))

    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_count_one, it) for it in items]
        for fut in as_completed(futures):
            aid, n, sz, status = fut.result()
            rows.append({"assembly_id": aid, "n_triangles": n, "file_size_bytes": sz, "status": status})
    rows.sort(key=lambda r: r["assembly_id"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / "triangles_assembly_obj.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["assembly_id", "n_triangles", "file_size_bytes", "status"])
        w.writeheader()
        w.writerows(rows)
    return rows


def count_combined_for_selected():
    sel_file = REPORT_DIR / "assemblies_selected.json"
    if not sel_file.exists():
        raise RuntimeError("assemblies_selected.json not found; run select_and_label first")
    sel = json.loads(sel_file.read_text())["all"]

    rows = []
    for aid in sel:
        p = LABELED_DIR / aid / "combined_mesh.obj"
        if not p.exists():
            rows.append({"assembly_id": aid, "n_triangles": None, "status": "missing"})
            continue
        try:
            n = count_triangles_obj(p)
            rows.append({"assembly_id": aid, "n_triangles": n, "status": "ok"})
        except Exception as e:
            rows.append({"assembly_id": aid, "n_triangles": None, "status": f"parse_error: {e}"})

    out_csv = REPORT_DIR / "triangles_combined.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["assembly_id", "n_triangles", "status"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    print("Counting triangles in all assembly.obj...")
    rows_assembly = count_all_assembly_obj()
    ok = [r for r in rows_assembly if r["status"] == "ok"]
    print(f"  total: {len(rows_assembly)}, ok: {len(ok)}")
    if ok:
        ns = [r["n_triangles"] for r in ok]
        print(f"  triangles min/median/max: {min(ns)} / {sorted(ns)[len(ns)//2]} / {max(ns)}")

    print("\nCounting triangles in 10 selected combined_mesh.obj...")
    rows_combined = count_combined_for_selected()
    for r in rows_combined:
        print(f"  {r['assembly_id']}: {r['n_triangles']} ({r['status']})")


if __name__ == "__main__":
    main()
