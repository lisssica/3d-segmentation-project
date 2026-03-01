import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

assembly_name = sys.argv[1]
render_mode = sys.argv[2]

mesh = trimesh.load(os.path.join('data', assembly_name, 'assembly.obj'))



if render_mode == 'normal':
    vertex_normals = mesh.vertex_normals
    normal_colors = (vertex_normals + 1) / 2
    mesh_normal = mesh.copy()
    mesh_normal.visual.vertex_colors = (normal_colors * 255).astype(np.uint8)

    mesh_normal.show(title='Normal Map')

elif render_mode == 'depth':

    vertices = mesh.vertices
    z = vertices[:, 2]
    z_min, z_max = z.min(), z.max()
    depth = (z - z_min) / (z_max - z_min)

    mesh_depth = mesh.copy()
    depth_colors = np.zeros((len(vertices), 3))
    depth_colors[:, 0] = depth  
    depth_colors[:, 1] = depth  
    depth_colors[:, 2] = depth  

    mesh_depth.visual.vertex_colors = (depth_colors * 255).astype(np.uint8)
    mesh_depth.show(title='Depth Map')
