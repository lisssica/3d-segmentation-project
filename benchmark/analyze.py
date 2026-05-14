"""Step 5: regression, total-time prediction, figures, REPORT.md."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    FIGURES_DIR,
    LABELED_DIR,
    N_FRAMES_PER_ASSEMBLY,
    RENDERS_DIR,
    REPORT_DIR,
    SAMPLES_DIR,
)


# ──────────────────────────── regression helpers ────────────────────────────

def fit_linear(x, y):
    """t = a + b*x. Returns (a, b)."""
    b, a = np.polyfit(x, y, 1)
    return a, b


def predict_linear(params, x):
    a, b = params
    return a + b * x


def fit_power(x, y):
    """t = c * x^k. Fit log y = log c + k log x. Returns (c, k)."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return None
    k, log_c = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(np.exp(log_c)), float(k)


def predict_power(params, x):
    c, k = params
    return c * np.power(np.maximum(x, 1), k)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def loocv_mae(x, y, fit_fn, predict_fn):
    errs = []
    n = len(x)
    for i in range(n):
        mask = np.arange(n) != i
        params = fit_fn(x[mask], y[mask])
        if params is None:
            continue
        pred = predict_fn(params, x[i])
        errs.append(abs(pred - y[i]))
    return float(np.mean(errs)) if errs else float("nan")


def fit_both_and_choose(x, y, label):
    out = {"name": label, "n_points": len(x)}
    linp = fit_linear(x, y)
    out["linear"] = {
        "a": float(linp[0]),
        "b": float(linp[1]),
        "r2": float(r2_score(y, predict_linear(linp, x))),
        "loocv_mae": loocv_mae(x, y, fit_linear, predict_linear),
    }
    powp = fit_power(x, y)
    if powp is not None:
        out["power"] = {
            "c": powp[0],
            "k": powp[1],
            "r2": float(r2_score(y, predict_power(powp, x))),
            "loocv_mae": loocv_mae(x, y, fit_power, predict_power),
        }
    else:
        out["power"] = None

    if out["power"] is None or out["linear"]["loocv_mae"] <= out["power"]["loocv_mae"]:
        out["chosen"] = "linear"
    else:
        out["chosen"] = "power"
    return out


def predict_stage(model, n):
    if model["chosen"] == "linear":
        return predict_linear((model["linear"]["a"], model["linear"]["b"]), n)
    return predict_power((model["power"]["c"], model["power"]["k"]), n)


# ─────────────────────────────── data loading ───────────────────────────────

def load_csv(path):
    with open(path) as fp:
        return list(csv.DictReader(fp))


def load_data():
    agg = load_csv(REPORT_DIR / "timings_per_assembly.csv")
    tri_assembly = load_csv(REPORT_DIR / "triangles_assembly_obj.csv")
    tri_combined = load_csv(REPORT_DIR / "triangles_combined.csv")

    def _to_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    def _to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    agg_rows = []
    for r in agg:
        agg_rows.append({
            "assembly_id": r["assembly_id"],
            "n_triangles_combined": _to_int(r["n_triangles_combined"]),
            "t_load_sec": _to_float(r["t_load_sec"]),
            "t_normals_sec": _to_float(r["t_normals_sec"]),
            "t_vbo_mean_sec": _to_float(r["t_vbo_mean_sec"]),
            "t_render_mean_sec": _to_float(r["t_render_mean_sec"]),
            "t_per_frame_mean_sec": _to_float(r["t_per_frame_mean_sec"]),
            "t_total_assembly_sec": _to_float(r["t_total_assembly_sec"]),
        })

    tri_a_map = {r["assembly_id"]: _to_int(r["n_triangles"]) for r in tri_assembly if r["status"] == "ok"}
    tri_c_map = {r["assembly_id"]: _to_int(r["n_triangles"]) for r in tri_combined if r["status"] == "ok"}

    # label_mesh timings (from select_and_label step)
    label_path = REPORT_DIR / "label_mesh_results.json"
    label_map = {}
    if label_path.exists():
        lm = json.loads(label_path.read_text())["label_mesh"]
        for aid, rec in lm.items():
            if rec.get("status") == "ok" and "t_label_mesh_sec" in rec:
                label_map[aid] = float(rec["t_label_mesh_sec"])
    return agg_rows, tri_a_map, tri_c_map, label_map


# ───────────────────────────── prediction core ──────────────────────────────

def fit_all_models(agg_rows):
    n = np.array([r["n_triangles_combined"] for r in agg_rows], dtype=np.float64)
    return {
        "t_load": fit_both_and_choose(n, np.array([r["t_load_sec"] for r in agg_rows]), "t_load"),
        "t_normals": fit_both_and_choose(n, np.array([r["t_normals_sec"] for r in agg_rows]), "t_normals"),
        "t_vbo": fit_both_and_choose(n, np.array([r["t_vbo_mean_sec"] for r in agg_rows]), "t_vbo"),
        "t_render": fit_both_and_choose(n, np.array([r["t_render_mean_sec"] for r in agg_rows]), "t_render"),
        "t_per_frame": fit_both_and_choose(n, np.array([r["t_per_frame_mean_sec"] for r in agg_rows]), "t_per_frame"),
    }


def predict_per_assembly(models, tri_a_map, tri_c_map, k_correction):
    rows = []
    for aid, n_tri in tri_a_map.items():
        if n_tri is None:
            continue
        n_corr = n_tri * k_correction
        pred_load = predict_stage(models["t_load"], n_corr)
        pred_norm = predict_stage(models["t_normals"], n_corr)
        pred_vbo = predict_stage(models["t_vbo"], n_corr)
        pred_render = predict_stage(models["t_render"], n_corr)
        pred_frame = pred_vbo + pred_render
        pred_assembly = pred_load + pred_norm + N_FRAMES_PER_ASSEMBLY * pred_frame

        # naive (no correction)
        pred_assembly_naive = (
            predict_stage(models["t_load"], n_tri)
            + predict_stage(models["t_normals"], n_tri)
            + N_FRAMES_PER_ASSEMBLY * (predict_stage(models["t_vbo"], n_tri)
                                       + predict_stage(models["t_render"], n_tri))
        )
        rows.append({
            "assembly_id": aid,
            "n_tri_assembly_obj": n_tri,
            "n_tri_corrected": n_corr,
            "pred_t_load_sec": max(0.0, pred_load),
            "pred_t_normals_sec": max(0.0, pred_norm),
            "pred_t_vbo_sec": max(0.0, pred_vbo),
            "pred_t_render_sec": max(0.0, pred_render),
            "pred_t_one_frame_sec": max(0.0, pred_frame),
            "pred_t_assembly_sec": max(0.0, pred_assembly),
            "pred_t_assembly_naive_sec": max(0.0, pred_assembly_naive),
        })
    rows.sort(key=lambda r: r["assembly_id"])
    return rows


def bootstrap_total(agg_rows, tri_a_map, k_correction, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_obs = len(agg_rows)
    obs_n = np.array([r["n_triangles_combined"] for r in agg_rows], dtype=np.float64)
    obs_load = np.array([r["t_load_sec"] for r in agg_rows])
    obs_norm = np.array([r["t_normals_sec"] for r in agg_rows])
    obs_vbo = np.array([r["t_vbo_mean_sec"] for r in agg_rows])
    obs_render = np.array([r["t_render_mean_sec"] for r in agg_rows])

    target_n = np.array([v * k_correction for v in tri_a_map.values() if v is not None])
    totals = []
    for _ in range(n_iter):
        idx = rng.integers(0, n_obs, size=n_obs)
        try:
            m_load = fit_both_and_choose(obs_n[idx], obs_load[idx], "")
            m_norm = fit_both_and_choose(obs_n[idx], obs_norm[idx], "")
            m_vbo = fit_both_and_choose(obs_n[idx], obs_vbo[idx], "")
            m_rdr = fit_both_and_choose(obs_n[idx], obs_render[idx], "")
        except Exception:
            continue
        pred = (np.maximum(0, predict_stage(m_load, target_n))
                + np.maximum(0, predict_stage(m_norm, target_n))
                + N_FRAMES_PER_ASSEMBLY * (np.maximum(0, predict_stage(m_vbo, target_n))
                                            + np.maximum(0, predict_stage(m_rdr, target_n))))
        totals.append(pred.sum())
    totals = np.array(totals)
    return float(totals.mean()), float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))


# ───────────────────────────────── figures ──────────────────────────────────

def fig_time_vs_triangles(agg_rows, models, fname, log_axes=False):
    n = np.array([r["n_triangles_combined"] for r in agg_rows])
    t = np.array([r["t_per_frame_mean_sec"] for r in agg_rows]) * 1000  # ms
    xgrid = np.geomspace(max(n.min() * 0.5, 100), n.max() * 1.5, 200)

    m = models["t_per_frame"]
    y_lin = predict_linear((m["linear"]["a"], m["linear"]["b"]), xgrid) * 1000
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n, t, s=60, color="C0", label="measured (mean of 3 frames)", zorder=3)
    ax.plot(xgrid, y_lin, "C1--",
            label=f"linear: t={m['linear']['a']*1000:.2f}+{m['linear']['b']*1e6:.3f}µs·n  R²={m['linear']['r2']:.3f}")
    if m["power"] is not None:
        y_pow = predict_power((m["power"]["c"], m["power"]["k"]), xgrid) * 1000
        ax.plot(xgrid, y_pow, "C2:",
                label=f"power: t={m['power']['c']*1000:.2g}·n^{m['power']['k']:.3f}  R²={m['power']['r2']:.3f}")
    for r in agg_rows:
        ax.annotate(r["assembly_id"].split("_")[0],
                    (r["n_triangles_combined"], r["t_per_frame_mean_sec"] * 1000),
                    fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("n triangles (combined_mesh)")
    ax.set_ylabel("t per frame, ms (VBO+render+readback)")
    ax.set_title(f"ModernGL Approach 3 — frame time vs triangles  ({'log-log' if log_axes else 'linear'})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=120)
    plt.close(fig)


def fig_stage_breakdown(agg_rows, fname):
    agg_rows = sorted(agg_rows, key=lambda r: r["n_triangles_combined"])
    labels = [f"{r['assembly_id'].split('_')[0]} ({r['n_triangles_combined']})" for r in agg_rows]
    t_load = np.array([r["t_load_sec"] for r in agg_rows]) * 1000
    t_norm = np.array([r["t_normals_sec"] for r in agg_rows]) * 1000
    t_vbo = np.array([r["t_vbo_mean_sec"] for r in agg_rows]) * 1000 * N_FRAMES_PER_ASSEMBLY
    t_rdr = np.array([r["t_render_mean_sec"] for r in agg_rows]) * 1000 * N_FRAMES_PER_ASSEMBLY

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels))))
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for arr, lab, col in [(t_load, "load_mesh", "C0"), (t_norm, "compute_normals", "C1"),
                          (t_vbo, f"VBO build (×{N_FRAMES_PER_ASSEMBLY})", "C2"),
                          (t_rdr, f"render+readback (×{N_FRAMES_PER_ASSEMBLY})", "C3")]:
        ax.barh(y, arr, left=left, label=lab, color=col)
        left += arr
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("time, ms (total for assembly: load+normals+3×(VBO+render))")
    ax.set_title("Stage breakdown per assembly")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=120)
    plt.close(fig)


def fig_histogram_754(tri_a_map, tri_c_map, fname):
    vals = [v for v in tri_a_map.values() if v is not None and v > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=np.geomspace(min(vals), max(vals), 50), alpha=0.75, color="C0")
    for aid, n_c in tri_c_map.items():
        n_a = tri_a_map.get(aid)
        if n_a:
            ax.axvline(n_a, color="C3", alpha=0.6, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("n triangles (assembly.obj)")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of triangle counts across {len(vals)} assemblies  (red lines: 10 sampled)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=120)
    plt.close(fig)


def fig_prediction_cdf(pred_rows, fname):
    times = np.array(sorted(r["pred_t_assembly_sec"] for r in pred_rows))
    cum = np.cumsum(times)
    n = np.arange(1, len(times) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(times, n / len(times), color="C0")
    ax1.set_xlabel("predicted time per assembly, sec")
    ax1.set_ylabel("fraction of assemblies")
    ax1.set_title("CDF of predicted per-assembly time")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot(n, cum / 60, color="C2")
    ax2.set_xlabel("# assemblies processed (sorted by predicted time)")
    ax2.set_ylabel("cumulative predicted time, min")
    ax2.set_title(f"Cumulative time over {len(times)} assemblies (total: {cum[-1]/60:.1f} min)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=120)
    plt.close(fig)


def fig_sample_frames(selected_ids):
    """3-panel PNG (normals/depth/seg) from saved .npz — does not affect timings."""
    for aid in selected_ids:
        npz_files = sorted((RENDERS_DIR / aid).glob("frame_*.npz"))
        if not npz_files:
            continue
        out_dir = SAMPLES_DIR / aid
        out_dir.mkdir(parents=True, exist_ok=True)
        for npz_path in npz_files:
            data = np.load(npz_path)
            depth = data["depth"]
            mask = data["pix_to_face"] >= 0
            seg = data["seg_mask"]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            normals = (data["normals_camera"] + 1) * 0.5
            normals[~mask] = 0.5
            axes[0].imshow(normals)
            axes[0].set_title("normals_camera")

            if mask.any():
                vmin = float(depth[mask].min())
                vmax = float(depth[mask].max())
                depth_vis = np.where(mask, depth, np.nan)
                im = axes[1].imshow(depth_vis, cmap="viridis", vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02,
                             label=f"camera-z (range {vmax - vmin:.3f})")
            else:
                axes[1].imshow(depth, cmap="viridis")
            axes[1].set_title("depth (linear, camera-space z)")

            seg_vis = np.ma.masked_where(seg < 0, seg)
            cmap = plt.get_cmap("tab20").copy()
            cmap.set_bad(color="black")
            axes[2].imshow(seg_vis, cmap=cmap)
            axes[2].set_title(f"seg_mask ({int(seg[mask].max()) if mask.any() else 0} labels)")

            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{aid} — {npz_path.stem}")
            fig.tight_layout()
            fig.savefig(out_dir / f"{npz_path.stem}.png", dpi=100)
            plt.close(fig)


# ────────────────────────────────── report ──────────────────────────────────

def write_report(agg_rows, tri_a_map, tri_c_map, models, k_corr,
                 pred_rows, total_naive, total_corr, ci_lo, ci_hi,
                 label_model=None, label_map=None, total_label=0.0, total_full=0.0):
    run_meta = {}
    rm_path = REPORT_DIR / "run_meta.json"
    if rm_path.exists():
        run_meta = json.loads(rm_path.read_text())

    measured_total_sec = sum(r["t_total_assembly_sec"] for r in agg_rows)
    measured_avg_sec = measured_total_sec / max(1, len(agg_rows))

    lines = []
    lines.append("# Benchmark Report — SEG_AIM rendering pipeline (Approach 3 ModernGL)\n")
    lines.append("## Setup\n")
    lines.append(f"- Host: `{run_meta.get('host', '?')}`  ·  Platform: `{run_meta.get('platform', '?')}`")
    lines.append(f"- Python: `{run_meta.get('python', '?')}`  ·  moderngl: `{run_meta.get('gl_info', {}).get('moderngl', '?')}`")
    lines.append(f"- GL: `{run_meta.get('gl_info', {}).get('GL_VENDOR', '?')}` / `{run_meta.get('gl_info', {}).get('GL_RENDERER', '?')}`")
    lines.append(f"- Image size: {run_meta.get('img_size', '?')}  ·  FOV: {run_meta.get('fov', '?')}")
    lines.append(f"- Frames per assembly: {N_FRAMES_PER_ASSEMBLY}  ·  Warmup runs: {run_meta.get('warmup_runs', '?')}")
    lines.append(f"- Timestamp: {run_meta.get('timestamp', '?')}\n")

    lines.append("## Per-assembly timings (10 measured)\n")
    lines.append("| assembly | n_tri (combined) | t_load (ms) | t_normals (ms) | t_vbo (ms) | t_render (ms) | t_per_frame (ms) | t_total (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(agg_rows, key=lambda x: x["n_triangles_combined"]):
        lines.append(
            f"| `{r['assembly_id']}` | {r['n_triangles_combined']:,} | "
            f"{r['t_load_sec']*1000:.1f} | {r['t_normals_sec']*1000:.2f} | "
            f"{r['t_vbo_mean_sec']*1000:.2f} | {r['t_render_mean_sec']*1000:.2f} | "
            f"{r['t_per_frame_mean_sec']*1000:.2f} | {r['t_total_assembly_sec']:.3f} |"
        )
    lines.append("")
    lines.append("![time vs triangles linear](figures/01_time_vs_triangles_linear.png)\n")
    lines.append("![time vs triangles log-log](figures/02_time_vs_triangles_loglog.png)\n")
    lines.append("![stage breakdown](figures/03_stage_breakdown.png)\n")

    lines.append("## Regression (per-frame time)\n")
    m = models["t_per_frame"]
    lines.append(f"- **Linear**: t = {m['linear']['a']*1000:.4f} + {m['linear']['b']*1e6:.4f}·n µs/triangle"
                 f"  ·  R² = {m['linear']['r2']:.4f}  ·  LOOCV-MAE = {m['linear']['loocv_mae']*1000:.3f} ms")
    if m["power"] is not None:
        lines.append(f"- **Power**: t = {m['power']['c']:.3e} · n^{m['power']['k']:.4f}"
                     f"  ·  R² = {m['power']['r2']:.4f}  ·  LOOCV-MAE = {m['power']['loocv_mae']*1000:.3f} ms")
    lines.append(f"- **Chosen**: `{m['chosen']}` (lowest LOOCV-MAE)\n")

    lines.append("## Prediction for full data/ folder\n")
    n_data = sum(1 for v in tri_a_map.values() if v is not None)
    total_tri_data = sum(v for v in tri_a_map.values() if v is not None)
    lines.append(f"- {n_data} assemblies parsed from `data/` (out of 754 total)")
    lines.append(f"- Sum of triangles (assembly.obj): {total_tri_data:,}")
    lines.append(f"- `combined_mesh` / `assembly.obj` triangle ratio (median): **k = {k_corr:.3f}**\n")
    lines.append(f"- **Render-only (naive, assembly.obj n_tri)**: {total_naive:.1f} s "
                 f"= {total_naive/60:.1f} min = {total_naive/3600:.2f} h")
    lines.append(f"- **Render-only (corrected, n_tri × k)**: {total_corr:.1f} s "
                 f"= {total_corr/60:.1f} min = {total_corr/3600:.2f} h")
    lines.append(f"- **Render-only bootstrap 95% CI**: [{ci_lo/60:.1f}, {ci_hi/60:.1f}] min\n")
    lines.append("![histogram of triangle counts](figures/04_histogram_754.png)\n")
    lines.append("![prediction CDF](figures/05_prediction_cdf.png)\n")

    if label_model is not None and total_label > 0:
        lm_chosen = label_model[label_model["chosen"]]
        lines.append("## Full pipeline: label_mesh + render\n")
        if label_model["chosen"] == "linear":
            lines.append(f"- **t_label_mesh ≈ {lm_chosen['a']*1000:.2f} ms + {lm_chosen['b']*1e6:.3f} µs · n_tri**"
                         f"  (R² = {lm_chosen['r2']:.4f}, LOOCV-MAE = {lm_chosen['loocv_mae']*1000:.2f} ms)")
        else:
            lines.append(f"- **t_label_mesh ≈ {lm_chosen['c']:.3e} · n_tri^{lm_chosen['k']:.3f}**"
                         f"  (R² = {lm_chosen['r2']:.4f}, LOOCV-MAE = {lm_chosen['loocv_mae']*1000:.2f} ms)")
        lines.append("")
        lines.append("| stage | total time |")
        lines.append("|---|---:|")
        lines.append(f"| label_mesh on 751 assemblies | {total_label:.1f} s = {total_label/60:.1f} min |")
        lines.append(f"| render ({N_FRAMES_PER_ASSEMBLY} frames/assembly × 751, corrected) | {total_corr:.1f} s = {total_corr/60:.1f} min |")
        lines.append(f"| **FULL pipeline** | **{total_full:.1f} s = {total_full/60:.1f} min = {total_full/3600:.2f} h** |\n")

        if label_map:
            lines.append("### Measured `t_label_mesh` (10 sampled)\n")
            lines.append("| assembly | n_tri (assembly.obj) | measured t_label (s) |")
            lines.append("|---|---:|---:|")
            for aid in sorted(label_map.keys()):
                lines.append(f"| `{aid}` | {tri_a_map.get(aid, '?'):,} | {label_map[aid]:.3f} |")
            lines.append("")

    lines.append("## assembly.obj vs combined_mesh.obj (10 sampled)\n")
    lines.append("| assembly | n_tri (assembly.obj) | n_tri (combined) | ratio |")
    lines.append("|---|---:|---:|---:|")
    for aid in sorted(tri_c_map.keys()):
        n_a, n_c = tri_a_map.get(aid), tri_c_map.get(aid)
        ratio = n_c / n_a if n_a else None
        lines.append(f"| `{aid}` | {n_a if n_a else '?':,} | {n_c if n_c else '?':,} | "
                     f"{ratio:.3f} |" if ratio else f"| `{aid}` | ? | ? | ? |")
    lines.append("")

    lines.append("## Measured vs predicted (10 sampled — sanity check)\n")
    lines.append("| assembly | n_tri (combined) | measured (s) | predicted (s) | rel err |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in sorted(agg_rows, key=lambda x: x["n_triangles_combined"]):
        n_tri = r["n_triangles_combined"]
        pred = (predict_stage(models["t_load"], n_tri)
                + predict_stage(models["t_normals"], n_tri)
                + N_FRAMES_PER_ASSEMBLY * (predict_stage(models["t_vbo"], n_tri)
                                            + predict_stage(models["t_render"], n_tri)))
        measured = r["t_total_assembly_sec"]
        rel = abs(pred - measured) / measured if measured > 0 else 0
        lines.append(f"| `{r['assembly_id']}` | {n_tri:,} | {measured:.3f} | {pred:.3f} | {rel*100:.1f}% |")
    lines.append("")

    lines.append("## Files\n")
    lines.append("- `triangles_assembly_obj.csv` — все ~754 сборок: n_triangles из `assembly.obj`")
    lines.append("- `triangles_combined.csv` — 10 выбранных: n_triangles из `combined_mesh.obj`")
    lines.append("- `timings_raw.csv` — per-frame timings (30 строк)")
    lines.append("- `timings_per_assembly.csv` — агрегаты по сборкам (10 строк)")
    lines.append("- `regression.json` — параметры моделей и метаданные")
    lines.append("- `prediction_per_assembly.csv` — прогноз для всех 754")
    lines.append("- `figures/` — графики и `samples/<id>/frame_NN.png` визуализации")
    lines.append(f"\n_Average measured total time per sampled assembly: {measured_avg_sec:.3f} s_\n")

    (REPORT_DIR / "REPORT.md").write_text("\n".join(lines))


def main():
    agg_rows, tri_a_map, tri_c_map, label_map = load_data()
    if not agg_rows:
        raise RuntimeError("No timings_per_assembly.csv data; run render_bench first.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fitting regression on {len(agg_rows)} measured assemblies...")
    models = fit_all_models(agg_rows)

    # k correction: median of n_combined / n_assembly_obj
    ks = []
    for aid, n_c in tri_c_map.items():
        n_a = tri_a_map.get(aid)
        if n_a and n_c and n_a > 0:
            ks.append(n_c / n_a)
    k_corr = float(np.median(ks)) if ks else 1.0
    print(f"  k correction (median n_combined/n_assembly_obj) = {k_corr:.3f}")

    # ── label_mesh regression: t_label vs n_triangles_assembly_obj ──
    label_model = None
    if label_map:
        pairs = [(tri_a_map.get(aid), t) for aid, t in label_map.items()
                 if tri_a_map.get(aid) is not None]
        if pairs:
            xs = np.array([p[0] for p in pairs], dtype=np.float64)
            ys = np.array([p[1] for p in pairs], dtype=np.float64)
            label_model = fit_both_and_choose(xs, ys, "t_label_mesh")
            print(f"  label_mesh model: chosen={label_model['chosen']} "
                  f"R²={label_model[label_model['chosen']]['r2']:.4f}")

    pred_rows = predict_per_assembly(models, tri_a_map, tri_c_map, k_corr)
    total_naive = sum(r["pred_t_assembly_naive_sec"] for r in pred_rows)
    total_corr = sum(r["pred_t_assembly_sec"] for r in pred_rows)
    print(f"  total render (N={N_FRAMES_PER_ASSEMBLY}): naive {total_naive/60:.1f} min, "
          f"corrected {total_corr/60:.1f} min")

    # Full pipeline: label_mesh + render
    total_label = 0.0
    if label_model is not None:
        for aid, n_tri in tri_a_map.items():
            if n_tri is None:
                continue
            total_label += max(0.0, predict_stage(label_model, n_tri))
    total_full_corr = total_corr + total_label
    print(f"  label_mesh total: {total_label/60:.1f} min")
    print(f"  FULL pipeline (label_mesh + render): {total_full_corr/60:.1f} min "
          f"= {total_full_corr/3600:.2f} h")

    # write prediction CSV
    if pred_rows:
        # Augment with label_mesh prediction column
        for r in pred_rows:
            n_tri = r["n_tri_assembly_obj"]
            r["pred_t_label_mesh_sec"] = (
                max(0.0, predict_stage(label_model, n_tri)) if label_model else 0.0
            )
            r["pred_t_full_pipeline_sec"] = r["pred_t_assembly_sec"] + r["pred_t_label_mesh_sec"]
        with open(REPORT_DIR / "prediction_per_assembly.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(pred_rows[0].keys()))
            w.writeheader()
            w.writerows(pred_rows)

    print("Bootstrap CI (n_iter=1000)...")
    boot_mean, ci_lo, ci_hi = bootstrap_total(agg_rows, tri_a_map, k_corr)
    print(f"  render-only 95% CI: [{ci_lo/60:.1f}, {ci_hi/60:.1f}] min")

    (REPORT_DIR / "regression.json").write_text(json.dumps({
        "models": models,
        "label_mesh_model": label_model,
        "k_correction": k_corr,
        "k_correction_samples": ks,
        "n_frames_per_assembly": N_FRAMES_PER_ASSEMBLY,
        "total_render_naive_sec": total_naive,
        "total_render_corrected_sec": total_corr,
        "total_label_mesh_sec": total_label,
        "total_full_pipeline_sec": total_full_corr,
        "bootstrap_render_ci_95_sec": [ci_lo, ci_hi],
        "bootstrap_render_mean_sec": boot_mean,
    }, indent=2, default=str))

    print("Drawing figures...")
    fig_time_vs_triangles(agg_rows, models, "01_time_vs_triangles_linear.png", log_axes=False)
    fig_time_vs_triangles(agg_rows, models, "02_time_vs_triangles_loglog.png", log_axes=True)
    fig_stage_breakdown(agg_rows, "03_stage_breakdown.png")
    fig_histogram_754(tri_a_map, tri_c_map, "04_histogram_754.png")
    fig_prediction_cdf(pred_rows, "05_prediction_cdf.png")
    fig_sample_frames([r["assembly_id"] for r in agg_rows])

    write_report(agg_rows, tri_a_map, tri_c_map, models, k_corr,
                 pred_rows, total_naive, total_corr, ci_lo, ci_hi,
                 label_model, label_map, total_label, total_full_corr)
    print(f"\n→ {REPORT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
