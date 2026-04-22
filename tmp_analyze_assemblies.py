import json
from pathlib import Path

def tree_depth(node, d=0):
    if not isinstance(node, dict) or not node:
        return d
    return max(tree_depth(v, d+1) for v in node.values())

def count_tree_instances(node):
    if not isinstance(node, dict) or not node:
        return 0
    return len(node) + sum(count_tree_instances(v) for v in node.values())

results = []
for folder in sorted(Path("data").iterdir()):
    if not folder.is_dir():
        continue
    jp = folder / "assembly.json"
    if not jp.exists():
        continue
    try:
        data = json.loads(jp.read_text())
    except Exception:
        continue
    tree = data.get("tree", {}).get("root", {})
    root_bodies = len(data.get("root", {}).get("bodies", {}))
    depth = tree_depth(tree)
    n_occ = count_tree_instances(tree)
    results.append((folder.name, depth, n_occ, root_bodies))

results.sort(key=lambda x: x[1], reverse=True)
print("=== Топ-10 по глубине дерева ===")
print(f"{'assembly':<25} depth  occ  root_bodies")
for r in results[:10]:
    print(f"  {r[0]:<23} {r[1]:>5}  {r[2]:>4}  {r[3]:>4}")

results.sort(key=lambda x: x[2], reverse=True)
print("\n=== Топ-10 по числу вхождений (occurrences) ===")
print(f"{'assembly':<25} depth  occ  root_bodies")
for r in results[:10]:
    print(f"  {r[0]:<23} {r[1]:>5}  {r[2]:>4}  {r[3]:>4}")

results.sort(key=lambda x: x[3], reverse=True)
print("\n=== Топ-10 по root_bodies ===")
print(f"{'assembly':<25} depth  occ  root_bodies")
for r in results[:10]:
    print(f"  {r[0]:<23} {r[1]:>5}  {r[2]:>4}  {r[3]:>4}")
