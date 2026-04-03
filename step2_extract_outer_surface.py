"""
Usage:
  python step2_extract_outer_surface.py <assembly> [img_size=512] [tol=0.005]

Output (preprocessed_data/<assembly>/):
  outer_face_mask.npy, outer_face_labels.npy
  renders/frame_XXXX.npz  — normals_camera, depth_camera, depth_world, seg_mask, pix_to_face
  renders/frames_metadata.json
"""
import sys, json
import numpy as np
from pathlib import Path

import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings,
    MeshRasterizer, look_at_view_transform
)

FOV = 60
CHECK_EVERY = 20
MAX_VIEWS = 2000


def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_extract_outer_surface.py assembly_code [img_size=512] [tol=0.005]")
        sys.exit(1)

    assembly = sys.argv[1]
    img_size = int(sys.argv[2])   if len(sys.argv) > 2 else 512
    tol      = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    data_dir = sys.argv[4]        if len(sys.argv) > 4 else "preprocessed_data"

    base        = Path(data_dir) / assembly
    renders_dir = base / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Assembly: {assembly}  size={img_size}  tol={tol}  device={device}")

    verts, faces_idx, _ = load_obj(str(base / "combined_mesh.obj"), load_textures=False)
    faces       = faces_idx.verts_idx
    face_labels = np.load(base / "face_labels.npy")
    verts_np    = verts.numpy()
    faces_np    = faces.numpy()

    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals_w = (cross / np.clip(norms, 1e-8, None)).astype(np.float32)

    mesh    = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    center  = verts.mean(dim=0)
    radius  = (verts - center).norm(dim=1).max().item()
    dist_min = radius / np.tan(np.deg2rad(FOV / 2))

    covered  = torch.zeros(faces.shape[0], dtype=torch.bool)
    rasterizer = MeshRasterizer(
        raster_settings=RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
    )

    prev_covered, k = 0, 0
    frames_meta = []

    while k < MAX_VIEWS:
        elev_r = float(np.random.uniform(-90, 90))
        azim_r = float(np.random.uniform(0, 360))
        dist_r = float(np.random.uniform(radius, 1.1 * dist_min))

        R, T = look_at_view_transform(
            dist=dist_r, elev=elev_r, azim=azim_r,
            at=(center.tolist(),), device=device
        )
        cam   = FoVPerspectiveCameras(R=R, T=T, fov=FOV, device=device)
        frags = rasterizer(mesh, cameras=cam)

        pix  = frags.pix_to_face[0, :, :, 0].cpu().numpy().astype(np.int32)
        zbuf = frags.zbuf[0, :, :, 0].cpu().numpy().astype(np.float32)
        bary = frags.bary_coords[0, :, :, 0, :].cpu().numpy()

        valid = pix >= 0
        covered[torch.from_numpy(pix[valid])] = True
        k += 1

        seg = np.full_like(pix, -1)
        seg[valid] = face_labels[pix[valid]]

        nc = np.zeros((img_size, img_size, 3), dtype=np.float32)
        dw = np.zeros((img_size, img_size), dtype=np.float32)
        if valid.any():
            nw_v = face_normals_w[pix[valid]]
            R_np = R[0].cpu().numpy()
            nc_v = nw_v @ R_np.T
            nc_v /= np.clip(np.linalg.norm(nc_v, axis=1, keepdims=True), 1e-8, None)
            nc[valid] = (nc_v + 1.0) * 0.5
            fv = verts_np[faces_np[pix[valid]]]        # (N, 3, 3)
            dw[valid] = (bary[valid] * fv[:, :, 2]).sum(axis=1)

        fid = f"{k:04d}"
        np.savez_compressed(renders_dir / f"frame_{fid}.npz",
            normals_camera=nc,
            depth_camera=zbuf,
            depth_world=dw,
            seg_mask=seg.astype(np.int32),
            pix_to_face=pix)

        frames_meta.append({
            "frame_id": k, "file": f"frame_{fid}.npz",
            "elev": elev_r, "azim": azim_r, "dist": dist_r,
            "R": R[0].cpu().numpy().tolist(),
            "T": T[0].cpu().numpy().tolist(), "fov": FOV
        })

        if k % CHECK_EVERY == 0:
            cur       = covered.sum().item()
            rel_delta = (cur - prev_covered) / max(cur, 1)
            print(f"  k={k:>4}: {cur/faces.shape[0]:.1%}  (+{rel_delta:.3%})")
            if rel_delta < tol:
                print(f"Converged at k={k}")
                break
            prev_covered = cur

    outer_mask   = covered.numpy()
    outer_labels = face_labels.copy()
    outer_labels[~outer_mask] = 0

    np.save(base / "outer_face_mask.npy", outer_mask)
    np.save(base / "outer_face_labels.npy", outer_labels)
    (renders_dir / "frames_metadata.json").write_text(json.dumps(frames_meta, indent=2))

    k_outer = len(np.unique(outer_labels[outer_labels > 0]))
    total_mb = sum(f.stat().st_size for f in renders_dir.glob("*.npz")) / 1e6

    print(f"Total faces: {faces.shape[0]}  Outer: {outer_mask.sum()} ({outer_mask.mean():.1%})")
    print(f"Frames: {k}  K_outer: {k_outer}  Size: {total_mb:.0f} MB")
    print(f"Saved to: {base}")


if __name__ == "__main__":
    main()
