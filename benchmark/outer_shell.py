"""Outer/inner shell separation via render-accumulate-until-convergence."""
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    DATASET_DIST_SCALE,
    FOV,
    LABELED_DIR,
    OUTER_CHECK_EVERY,
    OUTER_MAX_VIEWS,
    OUTER_SHELLS_DIR,
    OUTER_TOL,
    OUTER_ZOOM_RATIO,
    REPORT_DIR,
    SEED,
    ZOOM_THRESHOLD_SEGMENTS,
)
from .render_bench import init_moderngl_context, load_mesh, render_frame
from .utils import (
    compute_face_normals,
    mesh_center_radius,
    random_camera_normal,
    random_camera_zoom,
    stable_hash,
)


def extract_outer_shell(gl, assembly_id, tol=OUTER_TOL,
                        check_every=OUTER_CHECK_EVERY, max_views=OUTER_MAX_VIEWS,
                        labeled_root=None,
                        use_fixed_dist=False,
                        save_frames_dir=None,
                        save_every=1):
    """Accumulate visible faces by random rendering until convergence.

    Args:
        labeled_root: where to find combined_mesh.obj (default: LABELED_DIR)
        use_fixed_dist: if True, every frame uses dist = DATASET_DIST_SCALE · radius,
            target = mesh center, no zoom. Otherwise uses random_camera_normal/zoom mix.
        save_frames_dir: if set, save each frame as .npz here.
        save_every: save every Nth frame (default 1 = all).
    """
    root = Path(labeled_root) if labeled_root is not None else LABELED_DIR
    verts, faces, face_labels = load_mesh(root / assembly_id)
    face_normals_w = compute_face_normals(verts, faces)
    center, radius = mesh_center_radius(verts)
    n_segments = int(face_labels.max()) if face_labels.size else 0
    use_zoom = (not use_fixed_dist) and (n_segments >= ZOOM_THRESHOLD_SEGMENTS)

    covered = np.zeros(faces.shape[0], dtype=bool)
    prev_covered = 0
    history = []
    frames_meta = []
    rng = np.random.default_rng(SEED + stable_hash(assembly_id))

    if save_frames_dir is not None:
        save_frames_dir = Path(save_frames_dir)
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    k = 0
    converged = False
    for k in range(1, max_views + 1):
        if use_fixed_dist:
            elev = float(rng.uniform(-90.0, 90.0))
            azim = float(rng.uniform(0.0, 360.0))
            dist = float(DATASET_DIST_SCALE * radius)
            target = np.asarray(center, dtype=np.float32)
            kind = "fixed"
        else:
            is_zoom = use_zoom and (rng.random() < OUTER_ZOOM_RATIO)
            if is_zoom:
                elev, azim, dist, target = random_camera_zoom(verts, radius, FOV, rng)
                kind = "zoom"
            else:
                elev, azim, dist, target = random_camera_normal(center, radius, FOV, rng)
                kind = "normal"

        out, _ = render_frame(gl, verts, faces, face_normals_w, face_labels,
                              elev, azim, dist, target)
        pix = out["pix_to_face"]
        valid = pix >= 0
        if valid.any():
            covered[pix[valid]] = True

        if save_frames_dir is not None and (k % save_every == 0):
            np.savez_compressed(
                save_frames_dir / f"frame_{k:04d}.npz",
                pix_to_face=out["pix_to_face"],
                depth=out["depth"],
                normals_camera=out["normals_camera"],
                seg_mask=out["seg_mask"],
            )
            frames_meta.append({
                "frame": k, "kind": kind,
                "elev": elev, "azim": azim, "dist": dist,
                "target": [float(x) for x in target.tolist()],
            })

        if k % check_every == 0:
            cur = int(covered.sum())
            rel_delta = (cur - prev_covered) / max(cur, 1)
            history.append({
                "k": k, "covered": cur,
                "frac": cur / faces.shape[0],
                "rel_delta": rel_delta,
            })
            if rel_delta < tol:
                converged = True
                break
            prev_covered = cur

    t_total = time.perf_counter() - t0
    return {
        "assembly_id": assembly_id,
        "n_triangles": int(faces.shape[0]),
        "n_segments": n_segments,
        "n_frames": k,
        "t_total_sec": t_total,
        "outer_pct": float(covered.mean()),
        "converged": bool(converged),
        "tol": tol,
        "history": history,
        "frames_meta": frames_meta,
        "outer_mask": covered,
        "face_labels": face_labels,
        "verts": verts,
        "faces": faces,
        "face_normals_w": face_normals_w,
        "center": center,
        "radius": radius,
    }


def save_outer_artifacts(out_dir, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    outer_mask = result["outer_mask"]
    face_labels = result["face_labels"]

    np.save(out_dir / "outer_face_mask.npy", outer_mask)
    outer_labels = face_labels.copy()
    outer_labels[~outer_mask] = 0
    np.save(out_dir / "outer_face_labels.npy", outer_labels)

    meta = {k: v for k, v in result.items()
            if k not in ("outer_mask", "face_labels", "verts", "faces",
                         "face_normals_w", "center", "radius")}
    (out_dir / "convergence.json").write_text(json.dumps(meta, indent=2, default=float))


def render_viz_snapshot(gl, result, out_path, img_size_used):
    """Render outer/inner colouring from 3 fixed angles; save as 3-panel PNG.

    Uses bbox-based centre and half-diagonal radius (more robust than mean+
    max-distance for meshes whose vertex distribution is highly off-centre).
    """
    verts = result["verts"]
    faces = result["faces"]
    face_normals_w = result["face_normals_w"]
    outer_mask = result["outer_mask"]

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    bbox_radius = float(np.linalg.norm(bbox_max - bbox_min) * 0.5)
    dist_min = bbox_radius / np.tan(np.radians(FOV / 2))
    dist = 1.3 * dist_min  # leave 30% padding

    bin_labels = outer_mask.astype(np.int32) + 1  # 1=inner, 2=outer
    views = [("iso",  25.0, 35.0), ("front", 0.0, 0.0), ("side",  0.0, 90.0)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (name, elev, azim) in zip(axes, views):
        out, _ = render_frame(gl, verts, faces, face_normals_w, bin_labels,
                              elev, azim, dist, bbox_center)
        seg = out["seg_mask"]
        rgb = np.ones((seg.shape[0], seg.shape[1], 3), dtype=np.float32)  # white bg
        rgb[seg == 2] = [0.2, 0.4, 1.0]  # outer = blue
        rgb[seg == 1] = [1.0, 0.3, 0.3]  # inner = red
        ax.imshow(rgb)
        ax.set_title(f"{name}  elev={elev:.0f}° azim={azim:.0f}°")
        ax.axis("off")

    fig.suptitle(
        f"{result['assembly_id']}  —  outer (blue) / inner (red)  "
        f"|  n_frames={result['n_frames']}  t={result['t_total_sec']:.2f}s  "
        f"outer={result['outer_pct']*100:.1f}%  "
        f"n_segments={result['n_segments']}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def predict_total_for_data(summary_rows, reg_path=REPORT_DIR / "regression.json",
                           tri_csv=REPORT_DIR / "triangles_assembly_obj.csv"):
    """Estimate total time for all assemblies in data/ using simple logic
    (no model fitting on n_frames): combine measured N_frames median with the
    already-known t_per_frame regression from render_bench.
    """
    if not reg_path.exists() or not tri_csv.exists():
        return None
    reg = json.loads(reg_path.read_text())
    models = reg.get("models", {})
    k_corr = float(reg.get("k_correction", 1.0))

    def _predict(model, n):
        if model is None:
            return 0.0
        if model["chosen"] == "linear":
            return model["linear"]["a"] + model["linear"]["b"] * n
        return model["power"]["c"] * (max(n, 1) ** model["power"]["k"])

    n_frames_list = [r["n_frames"] for r in summary_rows if r.get("converged")]
    if not n_frames_list:
        n_frames_list = [r["n_frames"] for r in summary_rows]
    n_frames_median = float(np.median(n_frames_list))
    n_frames_mean = float(np.mean(n_frames_list))

    with open(tri_csv) as fp:
        rows = [r for r in csv.DictReader(fp) if r["status"] == "ok"]

    total_logic = 0.0
    for r in rows:
        n_tri = int(r["n_triangles"])
        n_corr = n_tri * k_corr
        t_load = max(0.0, _predict(models.get("t_load"), n_corr))
        t_per_frame = max(0.0, _predict(models.get("t_per_frame"), n_corr))
        total_logic += t_load + n_frames_median * t_per_frame

    mean_t_per_assembly = float(np.mean([r["t_total_sec"] for r in summary_rows]))
    total_naive = mean_t_per_assembly * len(rows)

    return {
        "n_assemblies_in_data": len(rows),
        "n_frames_median": n_frames_median,
        "n_frames_mean": n_frames_mean,
        "k_correction": k_corr,
        "method_logic": {
            "total_sec": total_logic,
            "total_min": total_logic / 60,
            "total_hours": total_logic / 3600,
            "fits_2h_budget": total_logic <= 7200,
            "explanation": "sum over 751 assemblies of t_load(n_tri·k) + "
                           "N_frames_median · t_per_frame(n_tri·k)",
        },
        "method_naive_mean": {
            "total_sec": total_naive,
            "total_min": total_naive / 60,
            "total_hours": total_naive / 3600,
            "explanation": "mean(t_total_sec on 10 assemblies) × 751",
        },
    }


def main():
    sel_file = REPORT_DIR / "assemblies_selected.json"
    if not sel_file.exists():
        raise RuntimeError("Run benchmark.select_and_label first.")
    selected = json.loads(sel_file.read_text())["all"]

    OUTER_SHELLS_DIR.mkdir(parents=True, exist_ok=True)
    gl = init_moderngl_context()
    print(f"ModernGL ready: {gl['info']}")

    summary_rows = []
    img_size = gl.get("info", {}).get("img_size", 512)
    for aid in selected:
        out_dir = OUTER_SHELLS_DIR / aid
        if not (LABELED_DIR / aid / "combined_mesh.obj").exists():
            print(f"  [SKIP] {aid}: combined_mesh.obj missing")
            continue
        print(f"  [RUN]  {aid} ...")
        try:
            result = extract_outer_shell(gl, aid)
        except Exception as e:
            print(f"  [FAIL] {aid}: {e}")
            summary_rows.append({
                "assembly_id": aid, "n_triangles": 0, "n_segments": 0,
                "n_frames": 0, "t_total_sec": 0.0, "outer_pct": 0.0,
                "converged": False, "status": f"error: {e}",
            })
            continue
        save_outer_artifacts(out_dir, result)
        render_viz_snapshot(gl, result, out_dir / "viz_outer_inner.png", img_size)
        print(f"         n_frames={result['n_frames']:4d}  "
              f"t={result['t_total_sec']:6.2f}s  "
              f"outer={result['outer_pct']*100:5.1f}%  "
              f"{'converged' if result['converged'] else 'MAX_VIEWS'}")
        summary_rows.append({
            "assembly_id": aid,
            "n_triangles": result["n_triangles"],
            "n_segments": result["n_segments"],
            "n_frames": result["n_frames"],
            "t_total_sec": result["t_total_sec"],
            "outer_pct": result["outer_pct"],
            "converged": result["converged"],
            "status": "ok",
        })

    if summary_rows:
        with open(OUTER_SHELLS_DIR / "summary.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    if ok_rows:
        pred = predict_total_for_data(ok_rows)
        if pred is not None:
            (OUTER_SHELLS_DIR / "prediction.json").write_text(
                json.dumps(pred, indent=2, default=float)
            )
            print(f"\n=== Prediction for full data/ ({pred['n_assemblies_in_data']} assemblies) ===")
            print(f"  N_frames median over {len(ok_rows)} measured: {pred['n_frames_median']:.0f}")
            print(f"  method_logic:      {pred['method_logic']['total_min']:.1f} min "
                  f"= {pred['method_logic']['total_hours']:.2f} h"
                  f"  ({'OK ≤2h' if pred['method_logic']['fits_2h_budget'] else 'OVER 2h'})")
            print(f"  method_naive_mean: {pred['method_naive_mean']['total_min']:.1f} min "
                  f"= {pred['method_naive_mean']['total_hours']:.2f} h")


if __name__ == "__main__":
    main()
