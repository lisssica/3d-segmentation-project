import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Загружаем mesh
mesh = trimesh.load('/Users/neonilllai/projects/SEG_AIM/data/7778_3a9748b3/assembly.obj')

# Простой просмотр (откроется окно)
mesh.show()

# Или визуализация в matplotlib (если нужно сохранить картинку)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Получаем вершины и треугольники
vertices = mesh.vertices
faces = mesh.faces

# Рисуем поверхность
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                triangles=faces, alpha=0.7, shade=True)

plt.show()