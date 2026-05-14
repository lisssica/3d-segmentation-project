#!/usr/bin/env python3
import sys
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj


def parse_transform(t):
    o, x, y, z = t["origin"], t["x_axis"], t["y_axis"], t["z_axis"]
    return torch.tensor([
        [x["x"], y["x"], z["x"], o["x"]],
        [x["y"], y["y"], z["y"], o["y"]],
        [x["z"], y["z"], z["z"], o["z"]],
        [0., 0., 0., 1.]
    ], dtype=torch.float32)


def load_and_transform(path, M):
    verts, faces_idx, _ = load_obj(str(path), load_textures=False)
    faces = faces_idx.verts_idx
    verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1)], dim=1)
    return (M @ verts_h.T).T[:, :3], faces


def collect_root_bodies(data, assembly_dir, results, counter):
    root = data.get("root", {})
    comp_name = data["components"].get(root.get("component", ""), {}).get("name", "")
    for body_uuid, body_info in root.get("bodies", {}).items():
        if not body_info.get("is_visible", True):
            continue
        path = assembly_dir / f"{body_uuid}.obj"
        if not path.exists():
            continue
        try:
            verts, faces = load_and_transform(path, torch.eye(4))
        except Exception:
            continue
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            continue
        results.append({
            "verts": verts, "faces": faces, "label": counter[0],
            "occ_uuid": "root", "body_uuid": body_uuid,
            "name": root.get("name", "root"), "component_name": comp_name
        })
        counter[0] += 1


def walk_tree(node, parent_M, data, assembly_dir, results, counter):
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
            try:
                verts, faces = load_and_transform(path, full_M)
            except Exception:
                continue
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                continue
            results.append({
                "verts": verts, "faces": faces, "label": counter[0],
                "occ_uuid": occ_uuid, "body_uuid": body_uuid,
                "name": occ.get("name", ""), "component_name": comp_name
            })
            counter[0] += 1
        if isinstance(children, dict) and children:
            walk_tree(children, full_M, data, assembly_dir, results, counter)


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_build_labeled_mesh.py <assembly_code>")
        sys.exit(1)

    assembly = sys.argv[1]
    assembly_dir = Path("data") / assembly
    output_dir = Path("preprocessed_data") / assembly
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads((assembly_dir / "assembly.json").read_text())
    results, counter = [], [1]
    collect_root_bodies(data, assembly_dir, results, counter)
    walk_tree(data["tree"]["root"], torch.eye(4), data, assembly_dir, results, counter)

    if not results:
        raise RuntimeError(f"No bodies loaded for {assembly}")

    all_verts, all_faces, all_labels, v_offset = [], [], [], 0
    for r in results:
        all_verts.append(r["verts"])
        all_faces.append(r["faces"] + v_offset)
        all_labels.append(torch.full((r["faces"].shape[0],), r["label"], dtype=torch.int32))
        v_offset += r["verts"].shape[0]

    verts_cat = torch.cat(all_verts, dim=0)
    faces_cat = torch.cat(all_faces, dim=0)
    labels_np = torch.cat(all_labels, dim=0).numpy()

    save_obj(str(output_dir / "combined_mesh.obj"), verts_cat, faces_cat)
    np.save(output_dir / "face_labels.npy", labels_np)
    (output_dir / "instance_map.json").write_text(json.dumps(
        {str(r["label"]): {"occurrence_uuid": r["occ_uuid"], "body_uuid": r["body_uuid"],
                           "name": r["name"], "component_name": r["component_name"]}
         for r in results}, indent=2))

    print(f"Instances: {len(results)}")
    print(f"Faces:     {labels_np.shape[0]}")
    print(f"Vertices:  {verts_cat.shape[0]}")
    print(f"Saved to:  {output_dir}")


if __name__ == "__main__":
    main()
