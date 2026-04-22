import subprocess, sys
from pathlib import Path

assemblies = [
    "21267_35a558cc", "21745_56cbe841", "23331_192866b6", "22983_0ab3faaf",
    "20803_473c87ac", "20551_ff28a959", "22652_dc7405ef", "22617_a286f13c",
    "20530_2246503a", "22298_51433023"
]

python = "/Users/neonilllai/miniconda3/envs/fusion_env/bin/python"
for a in assemblies:
    print(f"\n=== {a} ===")
    r = subprocess.run([python, "step1_build_labeled_mesh.py", a],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr:
        print("ERR:", r.stderr.strip()[:200])
