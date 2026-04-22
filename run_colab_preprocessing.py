#!/usr/bin/env python3
"""
Запуск в Colab:
  !python run_colab_preprocessing.py /content/drive/MyDrive/SEG_AIM/preprocessed_data
"""
import sys, time, subprocess
from pathlib import Path

assemblies = [
    "21267_35a558cc", "21745_56cbe841", "23331_192866b6", "22983_0ab3faaf",
    "20803_473c87ac", "20551_ff28a959", "22652_dc7405ef", "22617_a286f13c",
    "20530_2246503a", "22298_51433023",
]

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/SEG_AIM/preprocessed_data"
img_size = "512"
tol      = "0.005"

print(f"Data dir: {data_dir}")
print(f"Assemblies: {len(assemblies)}\n")

results = []
for assembly in assemblies:
    base = Path(data_dir) / assembly
    if not (base / "combined_mesh.obj").exists():
        print(f"[SKIP] {assembly} — combined_mesh.obj not found")
        results.append((assembly, "SKIP", 0))
        continue

    print(f"\n{'='*50}")
    print(f"Processing: {assembly}")
    t0 = time.time()

    r = subprocess.run(
        ["python", "step2_extract_outer_surface.py", assembly, img_size, tol, data_dir],
        text=True
    )
    elapsed = time.time() - t0
    status = "OK" if r.returncode == 0 else "ERROR"
    results.append((assembly, status, elapsed))
    print(f"  → {status}  {elapsed:.0f}s")

print(f"\n{'='*50}")
print("SUMMARY:")
total = 0
for assembly, status, t in results:
    print(f"  {status:5s}  {t:5.0f}s  {assembly}")
    total += t
print(f"\nTotal time: {total:.0f}s  ({total/60:.1f} min)")
