"""Step 1+2: select 10 assemblies (8 random + 2 pinned), then run label_mesh_fast."""
import json
import time
import traceback

import numpy as np

from .config import (
    DATA_DIR,
    LABELED_DIR,
    LOGS_DIR,
    N_RANDOM_PICKS,
    PINNED_ASSEMBLIES,
    REPORT_DIR,
    SEED,
)
from .label_mesh_fast import label_mesh


def list_valid_assemblies():
    out = []
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_dir():
            continue
        if (p / "assembly.json").exists() and (p / "assembly.obj").exists():
            out.append(p.name)
    return out


def select_assemblies():
    sel_file = REPORT_DIR / "assemblies_selected.json"
    if sel_file.exists():
        return json.loads(sel_file.read_text())

    all_valid = list_valid_assemblies()
    pinned = [a for a in PINNED_ASSEMBLIES if a in all_valid]
    pool = [a for a in all_valid if a not in pinned]
    rng = np.random.default_rng(SEED)
    random_pick = sorted(rng.choice(pool, size=N_RANDOM_PICKS, replace=False).tolist())

    record = {
        "seed": SEED,
        "pinned": pinned,
        "random": random_pick,
        "all": pinned + random_pick,
        "total_valid_in_data": len(all_valid),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sel_file.write_text(json.dumps(record, indent=2))
    return record


def run_label_mesh_on_selected(selected):
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "label_mesh.log"

    results = {}
    with open(log_file, "a") as logf:
        logf.write(f"\n=== Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for aid in selected:
            out_dir = LABELED_DIR / aid
            done_marker = out_dir / "combined_mesh.obj"
            if done_marker.exists():
                results[aid] = {"status": "cached"}
                logf.write(f"[CACHED] {aid}\n")
                print(f"[CACHED] {aid}")
                continue
            t0 = time.perf_counter()
            try:
                stats = label_mesh(aid, str(LABELED_DIR))
                dt = time.perf_counter() - t0
                results[aid] = {"status": "ok", "t_label_mesh_sec": dt, **stats}
                logf.write(f"[OK]     {aid}  {dt:7.2f}s  faces={stats['n_faces']}\n")
                print(f"[OK]     {aid}  {dt:7.2f}s  faces={stats['n_faces']}")
            except Exception as e:
                results[aid] = {"status": "label_failed", "error": str(e)}
                logf.write(f"[FAIL]   {aid}  {e}\n{traceback.format_exc()}\n")
                print(f"[FAIL]   {aid}  {e}")
    return results


def main():
    record = select_assemblies()
    print(f"Selected {len(record['all'])} assemblies (pool: {record['total_valid_in_data']}).")
    for aid in record["all"]:
        marker = "[PINNED]" if aid in record["pinned"] else "[RANDOM]"
        print(f"  {marker} {aid}")

    results = run_label_mesh_on_selected(record["all"])
    out = {"selection": record, "label_mesh": results}
    (REPORT_DIR / "label_mesh_results.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
