# Prompt-Plan: 3D Assembly Instance Segmentation Preprocessing Pipeline

## Контекст задачи
Задача — 3D инстанс-сегментация механических сборок из датасета Fusion 360 Assembly.
Алгоритм должен работать на данных 3D-сканера. Мы синтезируем обучающие данные из CAD-моделей.

**Гранулярность разметки:** каждая пара `(occurrence_uuid, body_uuid)` → уникальный целочисленный label.

**Данные:** JSON с деревом сборки + `.obj` файлы для каждого тела (в локальных координатах компонента).

**JSON структура:**
- `tree.root` — иерархия occurrence (может быть вложенной)
- `occurrences[uuid]` — transform (4×4), bodies (с is_visible), is_visible
- `components[uuid]` — список bodies UUID
- `bodies[uuid]` — метаданные тела (имя, физ. свойства, материал)
- `joints` — суставы между occurrence (кинематика)

**Окружение:** `fusion_env` (Python, trimesh, pytorch3d, numpy, open3d)

---

## Подзадачи

---

### Подзадача 1: Парсинг JSON и построение сборочного меша с разметкой

**Цель:** Из JSON + OBJ файлов построить единый меш сборки, где каждое лицо грани (face) размечено уникальным instance label.

**Скрипт:** `step1_build_labeled_mesh.py`

```
Реализуй скрипт step1_build_labeled_mesh.py на Python для разбора данных Fusion 360 Assembly.

ВХОДНЫЕ ДАННЫЕ:
- data/<assembly_name>/assembly.json
- data/<assembly_name>/<body_uuid>.obj  (меши тел в локальных координатах компонента)

ЗАДАЧА:
1. Прочитать assembly.json
2. Рекурсивно обойти tree.root (дерево occurrence UUID)
3. Для каждого occurrence (включая вложенные):
   a. Проверить is_visible == True
   b. Вычислить полную трансформацию: накапливать матрицы от корня до листа через
      matrix = parent_transform @ local_transform
      где local_transform берётся из occurrence['transform']:
      columns = [x_axis, y_axis, z_axis], translation = origin
   c. Для каждого body в occurrence['bodies'] проверить is_visible == True
   d. Загрузить OBJ файл тела из data/<assembly_name>/<body_uuid>.obj
   e. Применить full_transform к вершинам меша
   f. Для каждого face сохранить метку: label = уникальный int, 
      записать в словарь instance_map[(occurrence_uuid, body_uuid)] = label
      (нумерация с 1, 0 = background)
4. Объединить все трансформированные меши в единый trimesh.Trimesh (np.concatenate vertices/faces)
5. Сохранить:
   - combined_mesh.obj — объединённый меш
   - face_labels.npy — массив int32 длиной N_faces с метками каждой грани
   - instance_map.json — словарь {label: {occurrence_uuid, body_uuid, name, component_name}}

ВАЖНЫЕ ДЕТАЛИ:
- Компонент root (data['root']['component']) обычно не имеет тел сам по себе
- Некоторые bodies имеют volume=0 (degenerate) — пропускать, если меш пустой
- Некоторые тела имеют is_visible=False в occurrence — пропускать
- Одно и то же body_uuid может встречаться в нескольких occurrences — каждый instance = отдельная метка
- Дерево может быть вложенным (occurrence внутри occurrence): рекурсивно обходить tree.root
- При вложенных occurrences parent_transform передаётся рекурсивно

ЗАВИСИМОСТИ: trimesh, numpy, json

ВЫХОДНАЯ СТАТИСТИКА:
- Общее число уникальных instance labels
- Число граней в итоговом меше
- Число тел с is_visible=False (пропущенных)

ТАКЖЕ: добавь параметр --assembly и --output_dir для CLI.
```

**Скрипт визуализации:** `viz1_labeled_mesh.py`
```
Реализуй скрипт viz1_labeled_mesh.py:
1. Загрузить combined_mesh.obj и face_labels.npy
2. Назначить каждому face случайный RGB цвет по его метке (label -> color)
3. Background (label=0) -> серый
4. Показать с помощью trimesh.Scene (или open3d)
5. Вывести легенду первых 10 instance labels с именами из instance_map.json
```

---

### Подзадача 2: Извлечение внешней поверхности (симуляция 3D-сканера)

**Цель:** Убрать внутренние грани (места сочленений, внутренние стенки), оставив только внешнюю оболочку — то, что увидит 3D-сканер.

**Проблема:** У нас нет данных `contacts`. Нужно эвристически определить «видимые» грани.

**Скрипт:** `step2_extract_outer_surface.py`

```
Реализуй скрипт step2_extract_outer_surface.py.

ПОДХОД (несколько вариантов — реализуй оба, сравни):

ВАРИАНТ А: Convex hull / bounding box ray-casting
Для каждой грани в combined_mesh:
1. Вычислить центр грани (face centroid)
2. Запустить луч из центра грани вдоль нормали
3. Если луч немедленно попадает в другую грань этого же меша перед выходом наружу
   -> грань внутренняя (пропустить)
4. Если луч выходит наружу без пересечений -> грань внешняя

Использовать trimesh.ray.ray_triangle.RayMeshIntersector или trimesh.ray.

ВАРИАНТ Б: Convex pairwise intersection detection
Для каждой пары instance_a, instance_b:
1. Проверить, пересекаются ли их bounding boxes (быстрый тест)
2. Если пересекаются — найти грани instance_a, которые находятся ВНУТРИ меша instance_b
   (используя trimesh.contains_points или signed distance field)
3. Пометить эти грани как "контактные/внутренние"
4. Удалить контактные грани из итоговой разметки

ВЫХОДНЫЕ ДАННЫЕ:
- outer_face_mask.npy — bool массив N_faces (True = внешняя)
- outer_face_labels.npy — метки только внешних граней
- Статистика: сколько граней удалено

ВОПРОСЫ ДЛЯ ДОКУМЕНТИРОВАНИЯ:
- Что делать с полостями (труба с отверстием): грани внутри полости технически внешние,
  но 3D-сканер их может не видеть.
- Предложить решение: cast ray вдоль нормали AND против нормали 
  -> если both выходят наружу — точно внешняя; если только одно — возможно полость
```

**Скрипт визуализации:** `viz2_outer_surface.py`

---

### Подзадача 3: Пайплайн рендеринга с PyTorch3D

**Цель:** Для каждой сборки рендерить изображения с нескольких ракурсов.
Каждое изображение: 4 канала (Nx, Ny, Nz в мировых или camera-space координатах + глубина).
Выходные данные: изображения + карта соответствия пиксель→face_id.

**Скрипт:** `step3_pytorch3d_renderer.py`

```
Реализуй скрипт step3_pytorch3d_renderer.py для рендеринга сборки с помощью PyTorch3D.

ВХОДНЫЕ ДАННЫЕ:
- combined_mesh.obj
- face_labels.npy (или outer_face_labels.npy)
- instance_map.json

ПАРАМЕТРЫ КАМЕРЫ:
1. Вычислить bounding sphere: center = centroid меша, radius = max vertex distance from center
2. dist_camera = radius / tan(fov_rad / 2)  -- минимальное расстояние для полного охвата при fov=60°
3. Генерировать камеры в диапазоне [0.5 * dist_camera, 1.5 * dist_camera] (близко/далеко)
4. Углы: azimuth от 0 до 360°, elevation от -60° до +60°, N_cameras штук (параметр)
5. Использовать pytorch3d.renderer.FoVPerspectiveCameras с fov=60°

РЕНДЕРИНГ:
Для каждой камеры рендерить 2 варианта:

ВАРИАНТ A (world normals):
- Нормали граней в мировых координатах (не трансформированные камерой)
- Канал 1-3: face_normals (x, y, z) нормализованные, remapped [0,1]
- Канал 4: depth (расстояние от камеры до грани), нормализованное

ВАРИАНТ B (camera normals):
- Нормали граней в camera space (dot product с camera-to-face direction)
- Канал 1-3: normal в camera coordinates
- Канал 4: depth как в варианте A

ВАЖНО:
Использовать pytorch3d.renderer.MeshRasterizer — он возвращает:
- pix_to_face: (H, W) — index грани для каждого пикселя (-1 = background)
- zbuf: (H, W) — z-buffer (глубина)
- bary_coords: (H, W, 3) — барицентрические координаты
Это позволяет получить face_label для каждого пикселя через face_labels[pix_to_face]

ВЫХОДНЫЕ ДАННЫЕ для каждого кадра:
- normals_world_{frame_id}.png  (3-канальный RGB нормали в мировых координатах)
- normals_camera_{frame_id}.png  (3-канальный RGB нормали в camera space)  
- depth_{frame_id}.npy  (float32 depth map)
- seg_mask_{frame_id}.npy  (int32 H×W с instance labels, -1 = background)
- camera_params_{frame_id}.json  (R, T, fov параметры камеры)

МЕТАДАННЫЕ:
- frames_metadata.json: список всех кадров с путями к файлам

ЗАВИСИМОСТИ: torch, pytorch3d, numpy, PIL
```

**Скрипт визуализации:** `viz3_renders.py` — показать сетку из N рендеров, наложив маску сегментации поверх нормального изображения.

---

### Подзадача 4: Валидация позиции камеры (проверка что камера не внутри объекта)

**Цель:** Для каждой позиции камеры проверить, что она находится вне меша.

**Скрипт:** `step4_camera_validation.py`

```
Реализуй step4_camera_validation.py.

ЗАДАЧА:
Дан список позиций камер (3D точки) и combined_mesh.

Для каждой позиции камеры:
1. Проверить, находится ли точка внутри любого из instance meshes
   -> Использовать trimesh.contains_points(camera_position) для каждого instance
2. Если камера внутри объекта -> отфильтровать этот кадр

ДОПОЛНИТЕЛЬНО:
3. Проверить, что в кадре виден объект (хотя бы N пикселей не-фон в seg_mask)
   -> Фильтровать кадры с менее чем 5% покрытием объекта
4. Предложить стратегию пересэмплирования: если камера невалидна, 
   сделать шаг к center_of_mass сборки до тех пор, пока не окажется снаружи

ВЫХОДНЫЕ ДАННЫЕ:
- valid_camera_mask.npy — bool массив (True = валидная камера)
- Статистика: % отфильтрованных кадров
```

---

### Подзадача 5: Балансировщик покрытия граней

**Цель:** Убедиться, что каждая visible грань меша встречается хотя бы в одном кадре. Если нет — добавить дополнительные кадры для покрытия непокрытых граней.

**Скрипт:** `step5_coverage_balancer.py`

```
Реализуй step5_coverage_balancer.py.

АЛГОРИТМ:
1. Загрузить все seg_mask_*.npy (кадры)
2. Собрать множество покрытых face_ids: union по всем кадрам через pix_to_face
3. Найти непокрытые face: outer_face_ids \ covered_face_ids
4. Для каждой группы непокрытых граней:
   a. Найти нормаль+центр этих граней
   b. Вычислить оптимальную позицию камеры: camera_pos = face_centroid + face_normal * dist
   c. Добавить этот кадр в список рендеров
5. Итерировать до достижения coverage_threshold (параметр, default=95%)
6. Вывести финальную статистику покрытия

МЕТРИКИ:
- face_coverage_rate: процент покрытых граней
- instance_coverage_rate: для каждого instance label процент его граней покрытых
- Гистограмма: распределение числа кадров, покрывающих каждую грань
```

---

### Подзадача 6: Адаптация nnUnet для 2.5D инстанс-сегментации

**Цель:** Использовать nnUnet как baseline. Адаптировать вход (4-канальные изображения) и лосс (permutation-invariant).

**Документ:** `step6_nnunet_adaptation.md`

```
Создай документ step6_nnunet_adaptation.md с описанием:

1. КАК ИСПОЛЬЗОВАТЬ NNUNET С НАШИМИ ДАННЫМИ:
   - Структура датасета nnUnet: nnUNet_raw/Dataset<ID>_<Name>/
     - imagesTr/ — папка с обучающими изображениями 
       формат: <case_id>_<modality>.nii.gz
     - labelsTr/ — папка с масками
       формат: <case_id>.nii.gz (int16, каждый пиксель = instance label)
     - dataset.json — метаданные
   - Наши 4-канальные изображения: (nx, ny, nz, depth) -> 4 modality в nnUnet
   - Маска: seg_mask (int32 instance labels) -> нужна конвертация в nii.gz

2. ПРОБЛЕМА PERMUTATION-INVARIANT LOSS:
   nnUnet по умолчанию использует CrossEntropy + Dice loss, что требует фиксированного
   соответствия меток. У нас N instance с произвольными номерами.
   
   РЕШЕНИЕ — Hungarian Loss:
   - Перед вычислением лосса применить Hungarian algorithm (scipy.optimize.linear_sum_assignment)
     для поиска оптимального соответствия между predicted labels и GT labels
   - Лосс = Dice(pred_permuted, gt)
   
   Предложить как:
   a. Создать кастомный nnUnet trainer с переопределённым _compute_loss
   b. Или использовать post-processing hungarian matching отдельно от обучения

3. ИЗМЕНЕНИЕ ВХОДА:
   - nnUnet по умолчанию работает с медицинскими 3D/2D изображениями
   - Наш вход: 2D изображения H×W с 4 каналами (type: "2d" в nnUnet)
   - Указать num_input_channels=4 в dataset.json modalities

4. ПЛАН ЭКСПЕРИМЕНТА:
   - Baseline: nnUnet 2D с world normals (каналы: Nx, Ny, Nz, Depth_world)
   - Вариант B: nnUnet 2D с camera normals
   - Вариант C: nnUnet 2D с concat world+camera (7 каналов: Nx,Ny,Nz world + Nx,Ny,Nz cam + depth)
   - Метрика: mIoU по инстанс-сегментации (с Hungarian matching)
```

---

### Подзадача 7: Обратная проекция 2D разметки на 3D меш

**Цель:** Используя предсказания 2D-нейросети и карту pix_to_face, размечать 3D меш.

**Скрипт:** `step7_backproject_labels.py`

```
Реализуй step7_backproject_labels.py.

АЛГОРИТМ "мультивью голосование":
1. Для каждого кадра загрузить:
   - pred_seg_mask_k.npy — предсказание 2D сети (H×W с inst labels)
   - pix_to_face_k.npy — соответствие пиксель→face_id
2. Для каждой грани меша собрать голоса: 
   face_votes[face_id][predicted_label] += 1
3. После обработки всех кадров: face_final_label = argmax(face_votes[face_id])
4. Обработка конфликтов:
   - Если у грани 0 голосов (не покрыта ни одним кадром): face_label = nearest_neighbor_label
     (назначить метку ближайшей покрытой грани)
5. Вычислить метрики:
   - 3D mIoU (с Hungarian matching между pred и GT instance_map)
   - Покрытие граней (% с хотя бы 1 голосом)

ВЫХОДНЫЕ ДАННЫЕ:
- mesh_pred_labels.npy — предсказанные метки для каждой грани 3D меша
- Visualize: colored mesh по предсказаниям vs GT
```

---

### Подзадача 8: Пайплайн оркестрации (мастер-скрипт)

**Скрипт:** `run_pipeline.sh`

```
Создай bash скрипт run_pipeline.sh который запускает весь пайплайн:
  conda activate fusion_env
  python step1_build_labeled_mesh.py --assembly $1 --output_dir output/$1
  python step2_extract_outer_surface.py --input_dir output/$1
  python step3_pytorch3d_renderer.py --input_dir output/$1 --n_views 50
  python step4_camera_validation.py --input_dir output/$1
  python step5_coverage_balancer.py --input_dir output/$1 --coverage 0.95
  python step6_convert_to_nnunet.py --input_dir output/$1 --dataset_id 001
С аргументом имени сборки.
```

---

## Открытые вопросы (для исследования)

1. **Контакты/стыки деталей:** Поле `contacts` присутствует в части сборок (например `data/23451_14c8d09e/assembly.json`). Когда оно есть — использовать напрямую: каждый contact описывает пары тел которые соприкасаются → грани в зоне контакта помечать как внутренние. Когда `contacts` отсутствует — эвристика через ray casting или pairwise SDF intersection.

2. **Полости и слепые зоны:** Грани, находящиеся внутри полостей трубок, технически принадлежат внешнему контуру, но сканер их не увидит. Нужен более умный лучевой тест.

3. **Глубина нормали:** Что лучше учится — мировые нормали или нормали относительно камеры? Рекомендуется a/b тест с двумя вариантами.

4. **Качество рендеринга:** Rasterizer даёт дискретное соответствие пиксель→1 грань. При низком разрешении часть граней не попадёт ни в один пиксель. Решение: рендерить в 4K, потом downsample с агрегацией меток.

5. **nnUnet Hungarian loss:** Требует кастомизации trainer. Альтернатива — использовать Mask2Former / Detectron2 с инстанс-сегментацией напрямую.

6. **Дерево вложенных occurrences:** Подзадача 1 должна рекурсивно обходить вложенные occurrences (пример из данных: `72bb4b9a` содержит дочерние). Проверить что compose трансформаций корректен.

---

## Порядок выполнения

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 (параллельно с 3-5)
                     ↓               ↓
                  viz3_renders   viz4_coverage
                                     ↓
                                  Step 7
```
