import trimesh
import numpy as np
import json
import os
import sys
from collections import defaultdict, Counter

assembly_name = sys.argv[1]

json_file = os.path.join('data', assembly_name, 'assembly.json')
obj_folder = os.path.join('data', assembly_name)

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

# Анализ использования каждого body
print("="*60)
print("ANALYSIS: Body usage across components and occurrences")
print("="*60)

# Собираем все компоненты и их тела
component_bodies = {}
for comp_uuid, comp_data in data['components'].items():
    comp_name = comp_data.get('name', 'Unknown')
    bodies = comp_data.get('bodies', [])
    component_bodies[comp_uuid] = {
        'name': comp_name,
        'bodies': bodies
    }
    print(f"\nComponent: {comp_name} ({comp_uuid[:8]}...)")
    for body_uuid in bodies:
        print(f"  └─ Body: {body_uuid[:8]}...")

# Анализируем вхождения тел в разных местах
body_occurrences = defaultdict(list)

for occ_uuid, occ_data in data['occurrences'].items():
    occ_name = occ_data.get('name', 'Unknown')
    comp_uuid = occ_data['component']
    
    # Тела из компонента
    for body_uuid in component_bodies[comp_uuid]['bodies']:
        # Проверяем видимость
        bodies_info = occ_data.get('bodies', {})
        is_visible = bodies_info.get(body_uuid, {}).get('is_visible', True)
        
        if is_visible:
            body_occurrences[body_uuid].append({
                'occurrence': occ_name,
                'component': component_bodies[comp_uuid]['name']
            })

# Выводим статистику
print("\n" + "="*60)
print("BODY INSTANCE STATISTICS")
print("="*60)

for body_uuid, instances in body_occurrences.items():
    print(f"\nBody: {body_uuid}")
    print(f"  Appears in {len(instances)} places:")
    for inst in instances:
        print(f"    - Occurrence: {inst['occurrence']}, Component: {inst['component']}")

# Находим тело 03994022
target_body = "03994022-0526-11ec-bbd2-061e4e83ef1b"
print(f"\n" + "="*60)
print(f"SPECIFIC BODY ANALYSIS: {target_body}")
print("="*60)

if target_body in body_occurrences:
    print(f"Appears in {len(body_occurrences[target_body])} occurrences:")
    for inst in body_occurrences[target_body]:
        print(f"  - {inst['occurrence']} (component: {inst['component']})")
else:
    print("Not found in any visible occurrence!")
    
    # Проверяем, есть ли оно в компонентах
    print("\nChecking if body exists in components:")
    for comp_uuid, comp_data in component_bodies.items():
        if target_body in comp_data['bodies']:
            print(f"  - Found in component: {comp_data['name']} ({comp_uuid})")
    
    # Проверяем видимость
    print("\nChecking visibility settings:")
    for occ_uuid, occ_data in data['occurrences'].items():
        bodies_info = occ_data.get('bodies', {})
        if target_body in bodies_info:
            is_visible = bodies_info[target_body].get('is_visible', True)
            print(f"  - In occurrence {occ_data.get('name')}: visible={is_visible}")

# Теперь создаем сборку ТОЛЬКО с видимыми телами, учитывая что одно тело может появляться в нескольких местах
print("\n" + "="*60)
print("CREATING ASSEMBLY WITH CORRECT INSTANCES")
print("="*60)

# Строим иерархию трансформаций
def compute_full_transform(occ_uuid, parent_transform=np.identity(4), cache={}):
    """Рекурсивно вычисляем полную трансформацию для occurrence"""
    if occ_uuid in cache:
        return cache[occ_uuid]
    
    occ_data = data['occurrences'].get(occ_uuid)
    if not occ_data:
        return parent_transform
    
    # Получаем локальную трансформацию
    local_transform = transform_to_matrix(occ_data.get('transform'))
    
    # Комбинируем с родительской
    full_transform = parent_transform @ local_transform
    
    cache[occ_uuid] = full_transform
    return full_transform

# Собираем все видимые тела с их полными трансформациями
visible_instances = []

# Сначала находим корневые occurrence
root_occurrences = data.get('tree', {}).get('root', {})

for occ_uuid in root_occurrences:
    occ_data = data['occurrences'].get(occ_uuid)
    if not occ_data or not occ_data.get('is_visible', True):
        continue
    
    # Вычисляем полную трансформацию для этого occurrence
    full_transform = compute_full_transform(occ_uuid)
    
    comp_uuid = occ_data['component']
    
    # Добавляем все видимые тела из этого occurrence
    for body_uuid in component_bodies[comp_uuid]['bodies']:
        bodies_info = occ_data.get('bodies', {})
        if bodies_info.get(body_uuid, {}).get('is_visible', True):
            visible_instances.append({
                'body_uuid': body_uuid,
                'transform': full_transform,
                'occurrence': occ_data.get('name'),
                'component': component_bodies[comp_uuid]['name']
            })

# Выводим статистику
print(f"\nFound {len(visible_instances)} visible body instances to render:")
instance_counter = Counter([inst['body_uuid'] for inst in visible_instances])
for body_uuid, count in instance_counter.most_common():
    print(f"  {body_uuid[:8]}... appears {count} times")

# Загружаем и отображаем
scene = trimesh.Scene()
loaded_meshes = {}

for idx, inst in enumerate(visible_instances):
    body_uuid = inst['body_uuid']
    obj_file = os.path.join(obj_folder, f"{body_uuid}.obj")
    
    if not os.path.exists(obj_file):
        print(f"Warning: File not found: {obj_file}")
        continue
    
    # Загружаем mesh если еще не загружена
    if body_uuid not in loaded_meshes:
        try:
            loaded_meshes[body_uuid] = trimesh.load(obj_file)
        except Exception as e:
            print(f"Error loading {body_uuid}: {e}")
            continue
    
    # Создаем копию и применяем трансформацию
    mesh_copy = loaded_meshes[body_uuid].copy()
    mesh_copy.apply_transform(inst['transform'])
    
    # Добавляем в сцену с уникальным именем
    scene.add_geometry(mesh_copy, node_name=f"{body_uuid}_{idx}")
    
    if idx < 10:  # Показываем первые 10 для отладки
        print(f"Added: {body_uuid[:8]}... at {inst['occurrence']} (transform applied)")

print(f"\nTotal instances added: {len(visible_instances)}")
print(f"Unique meshes loaded: {len(loaded_meshes)}")

# Показываем сцену
if scene.geometry:
    scene.show()
else:
    print("No geometries to display!")