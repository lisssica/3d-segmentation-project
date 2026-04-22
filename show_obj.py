import trimesh
import numpy as np
import json
import os
import sys

assembly_name = sys.argv[1]

obj_file = os.path.join('data', assembly_name, 'assembly.obj')


# Load each body once and create transformed instances
scene = trimesh.Scene()
mesh = trimesh.load(obj_file)
scene.add_geometry(mesh)

# Visualize
if scene.geometry:
    print(f"\nBody Count: {len(scene.geometry)}")
    scene.show()