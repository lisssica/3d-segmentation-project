import trimesh
import numpy as np
import json
import os
import sys

assembly_name = sys.argv[1]

json_file = os.path.join('data', assembly_name, 'assembly.json')
obj_folder = os.path.join('data', assembly_name)

render_mode = sys.argv[1]

with open(json_file, 'r') as f:
    data = json.load(f)

def transform_to_matrix(transform_data):
    if not transform_data:
        return np.identity(4)
    
    origin = transform_data['origin']
    x_axis = transform_data['x_axis']
    y_axis = transform_data['y_axis']
    z_axis = transform_data['z_axis']
    
    matrix = np.eye(4)
    matrix[0:3, 0] = [x_axis['x'], x_axis['y'], x_axis['z']]
    matrix[0:3, 1] = [y_axis['x'], y_axis['y'], y_axis['z']]
    matrix[0:3, 2] = [z_axis['x'], z_axis['y'], z_axis['z']]
    matrix[0:3, 3] = [origin['x'], origin['y'], origin['z']]
    
    return matrix


scene = trimesh.Scene()

for occ_uuid, occ_data in data['occurrences'].items():
    occ_name = occ_data.get('name', 'Unknown')
    print(f"Occurance: {occ_name}")
    
    comp_uuid = occ_data['component']
    component = data['components'].get(comp_uuid, {})
    
    transform = transform_to_matrix(occ_data.get('transform'))
    
    bodies_in_occ = occ_data.get('bodies', {})
    
    for body_uuid, body_info in bodies_in_occ.items():
        if not body_info.get('is_visible', True):
            continue
        

        obj_file = os.path.join(obj_folder, f"{body_uuid}.obj")
        
        body_mesh = trimesh.load(obj_file)
        body_mesh.apply_transform(transform)
        
        color = np.random.rand(3)
        
        
        body_mesh.visual.vertex_colors = (color * 255).astype(np.uint8)
        scene.add_geometry(body_mesh)
        print(f"  + Add body: {body_uuid}")
                


if scene.geometry:
    print(f"\nBody Count: {len(scene.geometry)}")
    scene.show()
    