from pathlib import Path

results = []
for folder in sorted(Path("data").iterdir()):
    if not folder.is_dir():
        continue
    total_faces = sum(
        sum(1 for line in f.read_text(errors="ignore").splitlines() if line.startswith("f "))
        for f in folder.glob("*.obj")
        if f.name != "assembly.obj"
    )
    if total_faces > 0:
        results.append((total_faces, folder.name))

results.sort(reverse=True)
print(f"{'faces':>10}  assembly")
for faces, name in results[:15]:
    print(f"{faces:>10}  {name}")
