"""label_mesh, split into a cheap JSON pass and a heavy .obj-merge pass.

Two-stage pipeline:
  parse_assembly_json(assembly_dir)  -> {"n_bodies": int, "bodies": [...]}
  build_combined(parsed, output_dir) -> {"n_instances", "n_faces", "n_vertices"}

label_mesh(assembly_id, output_root) is kept as a thin shim that does both.
"""
import json
from pathlib import Path

import numpy as np
import trimesh

from .config import DATA_DIR


def parse_transform(t):
    o, x, y, z = t["origin"], t["x_axis"], t["y_axis"], t["z_axis"]
    return np.array(
        [
            [x["x"], y["x"], z["x"], o["x"]],
            [x["y"], y["y"], z["y"], o["y"]],
            [x["z"], y["z"], z["z"], o["z"]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _collect_root_bodies(data, assembly_dir, out_bodies, counter):
    root = data.get("root", {})
    comp_name = data["components"].get(root.get("component", ""), {}).get("name", "")
    for body_uuid, body_info in root.get("bodies", {}).items():
        if not body_info.get("is_visible", True):
            continue
        path = assembly_dir / f"{body_uuid}.obj"
        if not path.exists():
            continue
        out_bodies.append({
            "label": counter[0],
            "occ_uuid": "root",
            "body_uuid": body_uuid,
            "obj_path": path,
            "transform_M": np.eye(4, dtype=np.float64),
            "name": root.get("name", "root"),
            "component_name": comp_name,
        })
        counter[0] += 1


def _walk_tree(node, parent_M, data, assembly_dir, out_bodies, counter):
    for occ_uuid, children in node.items():
        occ = data["occurrences"].get(occ_uuid)
        if occ is None or not occ.get("is_visible", True):
            continue
        full_M = parent_M @ parse_transform(occ["transform"])
        comp_name = data["components"].get(occ.get("component", ""), {}).get("name", "")
        for body_uuid, body_info in occ.get("bodies", {}).items():
            if not body_info.get("is_visible", True):
                continue
            path = assembly_dir / f"{body_uuid}.obj"
            if not path.exists():
                continue
            out_bodies.append({
                "label": counter[0],
                "occ_uuid": occ_uuid,
                "body_uuid": body_uuid,
                "obj_path": path,
                "transform_M": full_M,
                "name": occ.get("name", ""),
                "component_name": comp_name,
            })
            counter[0] += 1
        if isinstance(children, dict) and children:
            _walk_tree(children, full_M, data, assembly_dir, out_bodies, counter)


def parse_assembly_json(assembly_dir):
    """Cheap pass: parse assembly.json, walk root + occurrences, collect visible
    bodies whose .obj file exists. No mesh loading. Returns:

        {
          "assembly_id": <name>,
          "n_bodies": int,
          "bodies": [
            {"label", "occ_uuid", "body_uuid", "obj_path",
             "transform_M" (4x4 float64), "name", "component_name"}
          ]
        }
    """
    assembly_dir = Path(assembly_dir)
    json_path = assembly_dir / "assembly.json"
    if not json_path.exists():
        return {"assembly_id": assembly_dir.name, "n_bodies": 0, "bodies": []}
    data = json.loads(json_path.read_text())
    bodies, counter = [], [1]
    _collect_root_bodies(data, assembly_dir, bodies, counter)
    tree = data.get("tree", {}).get("root", {})
    if isinstance(tree, dict):
        _walk_tree(tree, np.eye(4, dtype=np.float64), data, assembly_dir, bodies, counter)
    return {"assembly_id": assembly_dir.name, "n_bodies": len(bodies), "bodies": bodies}


def _load_and_transform(path, M):
    mesh = trimesh.load(str(path), force="mesh", process=False, skip_materials=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return verts.astype(np.float32), faces.astype(np.int32)
    verts_h = np.hstack([verts, np.ones((verts.shape[0], 1), dtype=np.float64)])
    transformed = (M @ verts_h.T).T[:, :3]
    return transformed.astype(np.float32), faces.astype(np.int32)


def _write_obj(path, verts, faces):
    lines = []
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for f in faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    with open(path, "w") as fp:
        fp.writelines(lines)


def build_combined(parsed, output_dir):
    """Heavy pass: take an already-parsed bodies list (from parse_assembly_json),
    load each .obj, apply its transform, merge into combined_mesh.obj + face_labels.

    Saves to output_dir/{combined_mesh.obj, face_labels.npy, instance_map.json}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bodies = parsed["bodies"]
    if not bodies:
        raise RuntimeError(f"No bodies in parsed assembly {parsed.get('assembly_id')}")

    all_verts, all_faces, all_labels, v_offset = [], [], [], 0
    valid_bodies = []
    for b in bodies:
        try:
            verts, faces = _load_and_transform(b["obj_path"], b["transform_M"])
        except Exception:
            continue
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            continue
        all_verts.append(verts)
        all_faces.append(faces + v_offset)
        all_labels.append(np.full((faces.shape[0],), b["label"], dtype=np.int32))
        v_offset += verts.shape[0]
        valid_bodies.append(b)

    if not all_verts:
        raise RuntimeError(f"No loadable bodies for {parsed.get('assembly_id')}")

    verts_cat = np.concatenate(all_verts, axis=0)
    faces_cat = np.concatenate(all_faces, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    _write_obj(output_dir / "combined_mesh.obj", verts_cat, faces_cat)
    np.save(output_dir / "face_labels.npy", labels_np)
    (output_dir / "instance_map.json").write_text(
        json.dumps(
            {
                str(b["label"]): {
                    "occurrence_uuid": b["occ_uuid"],
                    "body_uuid": b["body_uuid"],
                    "name": b["name"],
                    "component_name": b["component_name"],
                }
                for b in valid_bodies
            },
            indent=2,
        )
    )
    return {
        "n_instances": len(valid_bodies),
        "n_faces": int(faces_cat.shape[0]),
        "n_vertices": int(verts_cat.shape[0]),
    }


def label_mesh(assembly, output):
    """Backward-compat shim: parse + build in one call."""
    parsed = parse_assembly_json(DATA_DIR / assembly)
    return build_combined(parsed, Path(output) / assembly)


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m benchmark.label_mesh_fast <assembly> <output_dir>")
        sys.exit(1)
    stats = label_mesh(sys.argv[1], sys.argv[2])
    print(stats)


if __name__ == "__main__":
    main()
