# 3D Segmentation Project — Датасет и инструменты

Датасет CAD-сборок из Fusion 360 с готовой геометрической разметкой деталей и масками внешней оболочки (outer shell) — для обучения нейросети сегментации в 3D.

## Что внутри

- **`benchmark/dataset/`** — 379 сборок (2–10 деталей в каждой) с preview-сетками. У 5 sample-сборок — полные данные (combined mesh + outer mask + рендер-кадры до сходимости).
- **`benchmark/`** — Python-скрипты: построение датасета, label_mesh, outer-shell extraction (на ModernGL), визуализация.
- **`benchmark/data_stats/`** — распределения по 754 сборкам исходной `data/` (числу деталей и треугольникам в assembly.obj).

## Требования

Python 3.9+. Зависимости по скриптам:

| Скрипт | Модули |
|---|---|
| `viz_frame.py` (режим `2d`) | `numpy`, `matplotlib` |
| `viz_frame.py` (режим `3d`) | `numpy`, `matplotlib`, `trimesh`, `pyglet` |
| `viz_outer_shell.py` | `numpy`, `matplotlib`, `trimesh`, `pyglet` |
| `viz_grid.py` | `numpy`, `matplotlib`, `Pillow` |
| `build_dataset.py` | `numpy`, `matplotlib`, `trimesh`, `moderngl`, `Pillow` |
| `label_mesh_fast.py` | `numpy`, `trimesh` |
| `data_stats.py` | `numpy`, `matplotlib` |
| `outer_shell.py` | `numpy`, `matplotlib`, `trimesh`, `moderngl` |

Установка одной командой:

```bash
pip install numpy matplotlib trimesh moderngl Pillow pyglet
```

Опционально: создать venv и поставить туда же (по умолчанию в проекте `SEG_env/`).

## Структура репозитория

```
SEG_AIM/
├── benchmark/
│   ├── config.py                  # SEED, IMG_SIZE, FOV, пути, OUTER_*, DATASET_*
│   ├── utils.py                   # камеры, нормали, парсер треугольников
│   ├── label_mesh_fast.py         # parse_assembly_json + build_combined
│   ├── data_stats.py              # распределения по data/
│   ├── outer_shell.py             # extract_outer_shell (accumulate-until-converge)
│   ├── render_bench.py            # ModernGL renderer (shaders, framebuffer)
│   ├── build_dataset.py           # orchestrator: parse → filter → combine → render+outer
│   ├── viz_frame.py               # просмотр одного кадра по номеру
│   ├── viz_outer_shell.py         # 3D viewer outer/inner
│   ├── viz_grid.py                # сетки thumbnails из кадров
│   ├── prep_for_git.py            # очистка датасета перед коммитом
│   ├── data_stats/                # распределения 754 сборок (stats.csv + PNG)
│   └── dataset/                   # 379 сборок (2-10 деталей)
│       ├── selected.json
│       ├── summary.csv
│       └── <assembly_id>/
│           ├── assembly.png       # ← все 379
│           ├── grid_normals.png   # ← все 379
│           ├── grid_seg.png       # ← все 379
│           └── (полные данные — только 5 sample)
└── README.md
```

## Файлы в `benchmark/dataset/<assembly_id>/`

| Файл | Что это | Где |
|---|---|---|
| `assembly.png` | Превью сборки из Fusion 360 | **все 379** |
| `grid_normals.png` | Сетка normals со всех ракурсов рендера | **все 379** |
| `grid_seg.png` | Сетка segmentation mask (color per body) | **все 379** |
| `combined_mesh.obj` | Объединённый размеченный меш | только 5 sample |
| `face_labels.npy` | `int32 (M,)`, label на каждую грань (1..N_bodies) | только 5 sample |
| `instance_map.json` | `label → {body_uuid, name, component_name}` | только 5 sample |
| `outer_face_mask.npy` | `bool (M,)`, видна ли грань снаружи | только 5 sample |
| `outer_face_labels.npy` | face_labels с нулями для inner-граней | только 5 sample |
| `convergence.json` | n_frames, t_total_sec, outer_pct, история сходимости | только 5 sample |
| `frames/frame_NNNN.npz` | Один кадр: 4 массива (см. ниже) | только 5 sample |

### Что в `frame_NNNN.npz`

| Ключ | Shape | dtype | Значение |
|---|---|---|---|
| `pix_to_face` | `(512, 512)` | int32 | индекс треугольника на каждом пикселе (`-1` = фон) |
| `depth` | `(512, 512)` | float32 | линейная глубина (camera-space z) |
| `normals_camera` | `(512, 512, 3)` | float32 | нормаль в системе камеры, диапазон `[-1, 1]` |
| `seg_mask` | `(512, 512)` | int32 | segmentation label на пикселе (`-1` = фон) |

## Sample-сборки с полными данными

| n_bodies | assembly_id | n_triangles | n_frames | outer_pct |
|---:|---|---:|---:|---:|
| 2 | `16550_e88d6986` (болт) | 8 610 | 80 | 81% |
| 4 | `20281_a29f9a18` | 16 658 | 120 | 95% |
| 6 | `20467_f1fcc009` | 29 244 | 200 | 68% |
| 8 | `19518_f220b68a` | 18 124 | 60 | 96% |
| 10 | `20322_5a8c6077` | 43 170 | 120 | 31% |

## Скрипты визуализации — варианты вызова

### `viz_frame.py` — просмотр одного кадра по номеру

```bash
# 2D (matplotlib): три картинки бок о бок — normals, depth, seg_mask
python -m benchmark.viz_frame 16550_e88d6986 --frame 1 --mode 2d

# 3D (trimesh viewer): подсвечены только треугольники, попавшие в этот кадр
# (цвет = их segmentation label, остальные — серые)
python -m benchmark.viz_frame 16550_e88d6986 --frame 1 --mode 3d
python -m benchmark.viz_frame 20322_5a8c6077 --frame 42 --mode 3d
```

Доступные frame-номера у sample-сборки — `1` до `n_frames` из таблицы выше (или см. `convergence.json`).

### `viz_outer_shell.py` — 3D viewer внешней/внутренней оболочки

```bash
# Оба слоя одновременно: outer = синий, inner = красный (по умолчанию)
python -m benchmark.viz_outer_shell 16550_e88d6986

# Только outer (только синий — что увидел бы 3D-сканер)
python -m benchmark.viz_outer_shell 16550_e88d6986 --mode outer

# Только inner (только красный — что скрыто внутри сборки)
python -m benchmark.viz_outer_shell 16550_e88d6986 --mode inner

# Явное указание дефолтного режима
python -m benchmark.viz_outer_shell 20467_f1fcc009 --mode outer-inner
```

### `viz_grid.py` — сетка thumbnails со всех кадров

```bash
# Одна сборка, normals по умолчанию, ячейка 128 px
python -m benchmark.viz_grid 16550_e88d6986

# Другой размер ячейки
python -m benchmark.viz_grid 20322_5a8c6077 --thumb 64
python -m benchmark.viz_grid 20467_f1fcc009 --thumb 96

# Другой канал
python -m benchmark.viz_grid 16550_e88d6986 --channel depth
python -m benchmark.viz_grid 16550_e88d6986 --channel seg

# Произвольный путь сохранения
python -m benchmark.viz_grid 16550_e88d6986 --out /tmp/grid.png

# Все сборки сразу (где есть frames/)
python -m benchmark.viz_grid --all --thumb 96
python -m benchmark.viz_grid --all --thumb 96 --channel seg
```

### Управление в trimesh viewer

| Действие | Управление |
|---|---|
| Вращать сцену | ЛКМ + drag |
| Pan (смещение) | ПКМ + drag |
| Zoom | scroll |
| Wireframe | `w` |
| Оси координат | `a` |
| Выход | `q` |

## Построение датасета с нуля

Если у вас есть `data/` (Fusion-экспорт сборок), можно построить датасет заново:

```bash
# 1. Распределения по data/ (опционально — для аналитики)
python -m benchmark.data_stats

# 2. Полный pipeline: parse JSON → filter 2-10 деталей → combined_mesh → outer-shell + frames
python -m benchmark.build_dataset

# 3. Сетки thumbnails для всех собранных
python -m benchmark.viz_grid --all --thumb 96
python -m benchmark.viz_grid --all --thumb 96 --channel seg

# 4. Очистка перед коммитом (оставит только 5 sample + 3 PNG для остальных)
python -m benchmark.prep_for_git           # dry-run
python -m benchmark.prep_for_git --apply
```

Время на 379 сборок (Apple M4): label_mesh ~30 с, outer-shell + 45 000 кадров ~37 мин, сетки ~10 мин.

## Технические детали

- **Рендеринг**: ModernGL 5.12 (OpenGL 4.1 standalone context), MRT — за один draw call получаем 3 attachment'а: normals (rgba32f), depth (r32f), face_id (r32i).
- **Камера**: рендер для outer-shell идёт с фиксированной дистанцией `1.1 · radius` (центр меша = центр bounding sphere). Угол `(elev, azim)` случайный.
- **Outer-shell сходимость**: каждые 20 кадров проверяем `Δcovered / covered`; останавливаемся когда `< 1%`.
- **Без pytorch3d**: `label_mesh_fast.py` написан на `trimesh + numpy` (≈100× быстрее оригинала на pytorch3d).
