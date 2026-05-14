import hashlib
from pathlib import Path

import numpy as np

from .config import FOV, SCALE_NORMAL_RANGE, SCALE_ZOOM_RANGE


def mesh_center_radius(verts):
    c = verts.mean(axis=0)
    r = float(np.linalg.norm(verts - c, axis=1).max())
    return c.astype(np.float32), r


def random_camera_normal(center, radius, fov_deg=FOV, rng=None):
    """Whole-mesh framing: object fills ~80-100% of the frame.

    dist = scale · dist_min, scale ∈ SCALE_NORMAL_RANGE (default [1.0, 1.25]).
    scale=1.0 means the bounding sphere fits exactly in the FOV (100% fill);
    scale=1.25 leaves 20% padding (~80% fill).
    Returns (elev, azim, dist, target). target = mesh center.
    """
    if rng is None:
        rng = np.random.default_rng()
    elev = float(rng.uniform(-90.0, 90.0))
    azim = float(rng.uniform(0.0, 360.0))
    dist_min = radius / np.tan(np.radians(fov_deg / 2))
    scale = float(rng.uniform(*SCALE_NORMAL_RANGE))
    dist = scale * dist_min
    return elev, azim, dist, np.asarray(center, dtype=np.float32)


def random_camera_zoom(verts, radius, fov_deg=FOV, rng=None):
    """Close-up ('magnifier') view: target a random surface vertex, get closer.

    dist = scale · dist_min, scale ∈ SCALE_ZOOM_RANGE (default [0.2, 0.6]).
    The camera looks at a randomly-chosen vertex from the mesh, so different
    zoom frames focus on different regions of large multi-segment assemblies.
    Returns (elev, azim, dist, target).
    """
    if rng is None:
        rng = np.random.default_rng()
    elev = float(rng.uniform(-90.0, 90.0))
    azim = float(rng.uniform(0.0, 360.0))
    target = verts[rng.integers(0, verts.shape[0])].astype(np.float32)
    dist_min = radius / np.tan(np.radians(fov_deg / 2))
    scale = float(rng.uniform(*SCALE_ZOOM_RANGE))
    dist = scale * dist_min
    return elev, azim, dist, target


# Backward-compat shim (older callers expecting 3-tuple)
def random_camera_params(center, radius, fov_deg=FOV, rng=None):
    elev, azim, dist, _ = random_camera_normal(center, radius, fov_deg, rng)
    return elev, azim, dist


def compute_face_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    return (cross / np.clip(norms, 1e-8, None)).astype(np.float32)


def lookat_matrix(eye, at, up=None):
    if up is None:
        up = np.array([0.0, 1.0, 0.0], np.float32)
    f = at - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -r @ eye
    M[1, 3] = -u @ eye
    M[2, 3] = f @ eye
    return M


def perspective_matrix(fov_deg, aspect=1.0, near=0.01, far=100000.0):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    M = np.zeros((4, 4), np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1
    return M


def elev_azim_to_eye(center, dist, elev_deg, azim_deg):
    e = np.radians(elev_deg)
    a = np.radians(azim_deg)
    return center + dist * np.array(
        [np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], np.float32
    )


def count_triangles_obj(path):
    path = Path(path)
    with open(path, "r", errors="ignore") as f:
        head = [next(f, "") for _ in range(10)]
    for line in head:
        if "Triangles" in line and ":" in line:
            try:
                return int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
    n = 0
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            if ln.startswith("f "):
                n += 1
    return n


def stable_hash(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "big")
