import numpy as np
import trimesh
from pathlib import Path

INPUT = Path("output/7778_3a9748b3")

# --- БЛОК 1: Данные step1 ---
mesh = trimesh.load(INPUT / "combined_mesh.obj", force="mesh", process=False)
face_labels = np.load(INPUT / "face_labels.npy")
print("=== Step1 output ===")
print(f"Vertices: {mesh.vertices.shape}")
print(f"Faces:    {mesh.faces.shape}")
print(f"Labels:   {face_labels.shape}, unique={np.unique(face_labels).shape[0]}, max={face_labels.max()}")
print(f"Mesh scale: {mesh.scale:.4f}")
print()

# --- БЛОК 2: Ray casting работает? ---
print("=== Ray casting ===")
try:
    import rtree
    print("rtree: OK")
except ImportError:
    print("rtree: NOT FOUND (ray casting будет медленным/fallback)")

n_test = 5
centroids = mesh.triangles_center[:n_test]
normals = mesh.face_normals[:n_test]
eps = mesh.scale * 1e-6
origins = centroids + normals * eps
try:
    hits = mesh.ray.intersects_any(ray_origins=origins, ray_directions=normals)
    print(f"Ray hits (first {n_test} faces along +normal): {hits}")
    hits_b = mesh.ray.intersects_any(ray_origins=centroids - normals*eps, ray_directions=-normals)
    print(f"Ray hits (first {n_test} faces along -normal): {hits_b}")
    outer = (~hits) | (~hits_b)
    print(f"Outer (loose): {outer}")
except Exception as e:
    print(f"Ray error: {e}")
print()

# --- БЛОК 3: Step2 saved masks ---
print("=== Step2 masks ===")
for name in ["outer_face_mask.npy", "outer_face_mask_a.npy", "outer_face_mask_b.npy", "outer_face_labels.npy"]:
    p = INPUT / name
    if p.exists():
        arr = np.load(p)
        if arr.dtype == bool:
            print(f"{name}: {arr.sum()} / {arr.shape[0]} True")
        else:
            print(f"{name}: nonzero={np.count_nonzero(arr)} / {arr.shape[0]}")
print()

# --- БЛОК 4: PyTorch3D ---
print("=== PyTorch3D ===")
try:
    import torch
    print(f"torch: {torch.__version__}, cuda={torch.cuda.is_available()}")
    from pytorch3d.structures import Meshes
    print("pytorch3d: OK")
except ImportError as e:
    print(f"pytorch3d: {e}")
print()

# --- БЛОК 5: Renders directory ---
print("=== Renders ===")
rd = INPUT / "renders"
if rd.exists():
    files = list(rd.iterdir())
    print(f"renders/ exists, files: {len(files)}")
    for f in sorted(files)[:5]:
        print(f"  {f.name}")
else:
    print("renders/ NOT FOUND — step3 не запускался")
