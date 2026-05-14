"""Step 3: standalone ModernGL renderer (Approach 3) + timing instrumentation."""
import csv
import json
import platform
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import trimesh

from .config import (
    FOV,
    IMG_SIZE,
    LABELED_DIR,
    LOGS_DIR,
    MAX_TRIANGLES_GPU,
    N_FRAMES_PER_ASSEMBLY,
    N_ZOOM_FRAMES_RATIO,
    RENDERS_DIR,
    REPORT_DIR,
    SEED,
    WARMUP_RUNS,
    ZOOM_THRESHOLD_SEGMENTS,
)
from .utils import (
    compute_face_normals,
    elev_azim_to_eye,
    lookat_matrix,
    mesh_center_radius,
    perspective_matrix,
    random_camera_normal,
    random_camera_zoom,
    stable_hash,
)


VERT_SRC = """
#version 330 core
in vec3 in_pos;
in vec3 in_norm;
in int  in_fid;
out vec3 v_nc;
flat out int v_fid;
out float v_depth;
uniform mat4 MVP;
uniform mat4 V;
uniform mat3 NM;
void main() {
    vec4 cam_pos = V * vec4(in_pos, 1.0);
    vec4 clip    = MVP * vec4(in_pos, 1.0);
    gl_Position  = clip;
    v_nc    = normalize(NM * in_norm);
    v_fid   = in_fid;
    v_depth = -cam_pos.z;
}
"""

FRAG_SRC = """
#version 330 core
in vec3 v_nc;
flat in int v_fid;
in float v_depth;
layout(location=0) out vec4  out_normals;
layout(location=1) out float out_depth;
layout(location=2) out int   out_fid;
void main() {
    out_normals = vec4(v_nc, 1.0);
    out_depth   = v_depth;
    out_fid     = v_fid;
}
"""


def init_moderngl_context():
    import moderngl

    ctx = moderngl.create_standalone_context()
    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
    tex_n = ctx.texture((IMG_SIZE, IMG_SIZE), 4, dtype="f4")
    tex_d = ctx.texture((IMG_SIZE, IMG_SIZE), 1, dtype="f4")
    tex_f = ctx.texture((IMG_SIZE, IMG_SIZE), 1, dtype="i4")
    fbo = ctx.framebuffer(
        color_attachments=[tex_n, tex_d, tex_f],
        depth_attachment=ctx.depth_renderbuffer((IMG_SIZE, IMG_SIZE)),
    )
    info = {
        "moderngl": moderngl.__version__,
        "GL_VERSION": ctx.version_code,
        "GL_VENDOR": ctx.info.get("GL_VENDOR", ""),
        "GL_RENDERER": ctx.info.get("GL_RENDERER", ""),
    }
    return {"moderngl": moderngl, "ctx": ctx, "prog": prog,
            "fbo": fbo, "tex_n": tex_n, "tex_d": tex_d, "tex_f": tex_f, "info": info}


def build_vbo_bytes(verts, faces, face_normals_w):
    tv = verts[faces].reshape(-1, 3).astype(np.float32)
    tn = np.repeat(face_normals_w, 3, axis=0).astype(np.float32)
    fids = np.repeat(np.arange(len(faces), dtype=np.int32), 3)
    flat = np.hstack([tv, tn]).astype(np.float32)
    parts = []
    for i in range(len(fids)):
        parts.append(flat[i].tobytes())
        parts.append(struct.pack("i", int(fids[i])))
    return b"".join(parts)


def render_frame(gl, verts, faces, face_normals_w, face_labels, elev, azim, dist, target):
    """Mirror of Approach 3 render_moderngl. `target` is the lookat point (mesh center
    for normal frames, an arbitrary vertex for zoom frames). Returns dict + timings.
    """
    moderngl = gl["moderngl"]
    ctx, prog, fbo = gl["ctx"], gl["prog"], gl["fbo"]
    tex_n, tex_d, tex_f = gl["tex_n"], gl["tex_d"], gl["tex_f"]

    eye = elev_azim_to_eye(target, dist, elev, azim)
    V = lookat_matrix(eye, target.astype(np.float32))
    P = perspective_matrix(FOV)
    MVP = P @ V
    NM = np.linalg.inv(V[:3, :3]).T.astype(np.float32)

    # --- t_vbo: build VBO and upload to GPU ---
    t0 = time.perf_counter()
    vbo_data = build_vbo_bytes(verts, faces, face_normals_w)
    vbo = ctx.buffer(vbo_data)
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f 1i", "in_pos", "in_norm", "in_fid")])
    t_vbo = time.perf_counter() - t0

    # --- t_render: clear, set uniforms, draw, readback ---
    t0 = time.perf_counter()
    fbo.use()
    ctx.clear(0, 0, 0, 0)
    ctx.enable(moderngl.DEPTH_TEST)
    prog["MVP"].write(MVP.T.astype(np.float32).tobytes())
    prog["V"].write(V.T.astype(np.float32).tobytes())
    prog["NM"].write(NM.T.tobytes())
    vao.render(moderngl.TRIANGLES)
    ctx.finish()

    nc = np.flipud(np.frombuffer(tex_n.read(), np.float32).reshape(IMG_SIZE, IMG_SIZE, 4))[:, :, :3].copy()
    dep = np.flipud(np.frombuffer(tex_d.read(), np.float32).reshape(IMG_SIZE, IMG_SIZE)).copy()
    pix = np.flipud(np.frombuffer(tex_f.read(), np.int32).reshape(IMG_SIZE, IMG_SIZE)).copy()

    valid = np.linalg.norm(nc, axis=-1) > 0.01
    pix_out = np.where(valid, pix, -1).astype(np.int32)
    seg = np.full_like(pix_out, -1)
    valid_pix = pix_out >= 0
    if valid_pix.any():
        seg[valid_pix] = face_labels[pix_out[valid_pix]]
    t_render = time.perf_counter() - t0

    vao.release()
    vbo.release()
    return {
        "pix_to_face": pix_out,
        "depth": dep,
        "normals_camera": nc,
        "seg_mask": seg,
    }, {"t_vbo_sec": t_vbo, "t_render_sec": t_render, "eye": eye.tolist()}


def load_mesh(labeled_dir):
    """Load combined_mesh.obj + face_labels.npy. Returns (verts, faces, face_labels)."""
    mesh = trimesh.load(str(labeled_dir / "combined_mesh.obj"), force="mesh", process=False, skip_materials=True)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    face_labels = np.load(labeled_dir / "face_labels.npy").astype(np.int32)
    return verts, faces, face_labels


def bench_assembly(gl, assembly_id):
    out_dir = RENDERS_DIR / assembly_id
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir = LABELED_DIR / assembly_id

    if not (labeled_dir / "combined_mesh.obj").exists():
        return {"assembly_id": assembly_id, "status": "missing_label_output", "errors": ["combined_mesh.obj missing"]}

    # --- per-assembly stages ---
    t0 = time.perf_counter()
    verts, faces, face_labels = load_mesh(labeled_dir)
    t_load = time.perf_counter() - t0

    if faces.shape[0] > MAX_TRIANGLES_GPU:
        return {
            "assembly_id": assembly_id,
            "status": "skipped_size",
            "n_triangles_combined": int(faces.shape[0]),
            "n_vertices_combined": int(verts.shape[0]),
            "t_load_sec": t_load,
        }

    t0 = time.perf_counter()
    face_normals_w = compute_face_normals(verts, faces)
    t_normals = time.perf_counter() - t0

    center, radius = mesh_center_radius(verts)
    n_segments = int(face_labels.max()) if face_labels.size else 0
    use_zoom = n_segments >= ZOOM_THRESHOLD_SEGMENTS
    n_zoom = int(round(N_FRAMES_PER_ASSEMBLY * N_ZOOM_FRAMES_RATIO)) if use_zoom else 0

    rng = np.random.default_rng(SEED + stable_hash(assembly_id))

    frames_info = []
    errors = []
    for frame_i in range(N_FRAMES_PER_ASSEMBLY):
        # Last `n_zoom` frames are close-up "magnifier" views on a random vertex
        is_zoom = frame_i >= (N_FRAMES_PER_ASSEMBLY - n_zoom)
        if is_zoom:
            elev, azim, dist, target = random_camera_zoom(verts, radius, FOV, rng=rng)
        else:
            elev, azim, dist, target = random_camera_normal(center, radius, FOV, rng=rng)
        try:
            out, timings = render_frame(gl, verts, faces, face_normals_w, face_labels,
                                        elev, azim, dist, target)
        except Exception as e:
            errors.append(f"frame {frame_i}: {e}")
            continue
        # Save .npz AFTER timing measurements
        np.savez_compressed(
            out_dir / f"frame_{frame_i:02d}.npz",
            pix_to_face=out["pix_to_face"],
            depth=out["depth"],
            normals_camera=out["normals_camera"],
            seg_mask=out["seg_mask"],
        )
        frames_info.append({
            "frame": frame_i,
            "kind": "zoom" if is_zoom else "normal",
            "elev": elev,
            "azim": azim,
            "dist": dist,
            "target": target.tolist(),
            "t_vbo_sec": timings["t_vbo_sec"],
            "t_render_sec": timings["t_render_sec"],
        })

    status = "ok" if frames_info else "render_failed"
    record = {
        "assembly_id": assembly_id,
        "status": status,
        "n_triangles_combined": int(faces.shape[0]),
        "n_vertices_combined": int(verts.shape[0]),
        "n_segments": n_segments,
        "n_zoom_frames": n_zoom,
        "t_load_sec": t_load,
        "t_normals_sec": t_normals,
        "frames": frames_info,
        "t_total_render_sec": sum(f["t_vbo_sec"] + f["t_render_sec"] for f in frames_info),
        "errors": errors,
    }
    (out_dir / "timings.json").write_text(json.dumps(record, indent=2))
    return record


def aggregate_per_assembly(records):
    rows = []
    for r in records:
        if r.get("status") != "ok" or not r.get("frames"):
            continue
        frs = r["frames"]
        t_vbo = np.array([f["t_vbo_sec"] for f in frs])
        t_render = np.array([f["t_render_sec"] for f in frs])
        t_per_frame_mean = float((t_vbo + t_render).mean())
        rows.append({
            "assembly_id": r["assembly_id"],
            "n_triangles_combined": r["n_triangles_combined"],
            "n_vertices_combined": r["n_vertices_combined"],
            "t_load_sec": r["t_load_sec"],
            "t_normals_sec": r["t_normals_sec"],
            "t_vbo_mean_sec": float(t_vbo.mean()),
            "t_vbo_std_sec": float(t_vbo.std(ddof=0)),
            "t_render_mean_sec": float(t_render.mean()),
            "t_render_std_sec": float(t_render.std(ddof=0)),
            "t_per_frame_mean_sec": t_per_frame_mean,
            "t_total_assembly_sec": r["t_load_sec"] + r["t_normals_sec"]
                                     + N_FRAMES_PER_ASSEMBLY * t_per_frame_mean,
            "n_frames": len(frs),
        })
    return rows


def write_csvs(records, agg_rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # timings_raw.csv: one row per frame
    raw_rows = []
    for r in records:
        if r.get("status") != "ok":
            continue
        for f in r["frames"]:
            raw_rows.append({
                "assembly_id": r["assembly_id"],
                "n_triangles_combined": r["n_triangles_combined"],
                "n_segments": r.get("n_segments", 0),
                "frame": f["frame"],
                "kind": f.get("kind", "normal"),
                "elev": f["elev"],
                "azim": f["azim"],
                "dist": f["dist"],
                "t_vbo_sec": f["t_vbo_sec"],
                "t_render_sec": f["t_render_sec"],
            })
    if raw_rows:
        with open(REPORT_DIR / "timings_raw.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            w.writerows(raw_rows)

    if agg_rows:
        with open(REPORT_DIR / "timings_per_assembly.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)


def main():
    sel_file = REPORT_DIR / "assemblies_selected.json"
    if not sel_file.exists():
        raise RuntimeError("Run select_and_label first.")
    selected = json.loads(sel_file.read_text())["all"]

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "render_bench.log"

    gl = init_moderngl_context()
    print(f"ModernGL ready: {gl['info']}")

    # ── warmup: find smallest mesh, render once, discard timing ──
    sizes = []
    for aid in selected:
        p = LABELED_DIR / aid / "combined_mesh.obj"
        if p.exists():
            sizes.append((p.stat().st_size, aid))
    if sizes and WARMUP_RUNS > 0:
        sizes.sort()
        warm_aid = sizes[0][1]
        print(f"Warmup on {warm_aid} ({WARMUP_RUNS}x, discarded)...")
        try:
            verts, faces, face_labels = load_mesh(LABELED_DIR / warm_aid)
            fn = compute_face_normals(verts, faces)
            c, r = mesh_center_radius(verts)
            rng = np.random.default_rng(0)
            for _ in range(WARMUP_RUNS):
                e, a, d, t = random_camera_normal(c, r, FOV, rng=rng)
                render_frame(gl, verts, faces, fn, face_labels, e, a, d, t)
        except Exception as e:
            print(f"Warmup failed: {e}")

    records = []
    with open(log_path, "a") as logf:
        logf.write(f"\n=== Run at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        logf.write(f"GL info: {json.dumps(gl['info'])}\n")
        for aid in selected:
            t0 = time.perf_counter()
            try:
                rec = bench_assembly(gl, aid)
            except Exception as e:
                rec = {"assembly_id": aid, "status": "render_failed",
                       "errors": [str(e)], "traceback": traceback.format_exc()}
            wall = time.perf_counter() - t0
            records.append(rec)
            n_tri = rec.get("n_triangles_combined", "?")
            t_total = rec.get("t_total_render_sec", float("nan"))
            print(f"  [{rec.get('status', '?'):16s}] {aid:24s} tri={n_tri}  "
                  f"render_total={t_total:.3f}s  wall={wall:.3f}s")
            logf.write(f"[{rec.get('status', '?')}] {aid} {n_tri} {t_total:.4f} wall={wall:.4f}\n")
            if rec.get("errors"):
                for e in rec["errors"]:
                    logf.write(f"    ERR: {e}\n")

    agg_rows = aggregate_per_assembly(records)
    write_csvs(records, agg_rows)

    # Store run metadata for analyze.py
    meta = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "img_size": IMG_SIZE,
        "fov": FOV,
        "n_frames_per_assembly": N_FRAMES_PER_ASSEMBLY,
        "warmup_runs": WARMUP_RUNS,
        "seed": SEED,
        "gl_info": gl["info"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (REPORT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(agg_rows)} per-assembly aggregates → report/timings_per_assembly.csv")
    return records


if __name__ == "__main__":
    main()
