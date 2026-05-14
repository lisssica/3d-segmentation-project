from pathlib import Path

SEED = 42
N_RANDOM_PICKS = 8
PINNED_ASSEMBLIES = ["7778_3a9748b3", "16550_e88d6986"]
N_FRAMES_PER_ASSEMBLY = 10
IMG_SIZE = 512
FOV = 60
MAX_TRIANGLES_GPU = 2_000_000
WARMUP_RUNS = 1

# Camera framing
SCALE_NORMAL_RANGE = (1.0, 1.25)   # whole-mesh framing: 80-100% of frame
SCALE_ZOOM_RANGE = (0.2, 0.6)      # close-up "magnifier" on a random vertex
ZOOM_THRESHOLD_SEGMENTS = 15       # meshes with >= N labels get zoom frames
N_ZOOM_FRAMES_RATIO = 0.3          # fraction of frames that are zoom-ins

# Outer-shell extraction (accumulate-until-convergence)
OUTER_TOL = 0.01                   # stop when new faces / covered < 1%
OUTER_CHECK_EVERY = 20             # convergence check cadence
OUTER_MAX_VIEWS = 1000             # safety cap
OUTER_ZOOM_RATIO = 0.3             # share of zoom frames in the loop

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BENCH_DIR = ROOT / "benchmark"
LABELED_DIR = BENCH_DIR / "labeled_meshes"
RENDERS_DIR = BENCH_DIR / "renders"
REPORT_DIR = BENCH_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"
SAMPLES_DIR = FIGURES_DIR / "samples"
LOGS_DIR = BENCH_DIR / "logs"
OUTER_SHELLS_DIR = BENCH_DIR / "outer_shells"
DATA_STATS_DIR = BENCH_DIR / "data_stats"
DATASET_DIR = BENCH_DIR / "dataset"

DATASET_MIN_BODIES = 2
DATASET_MAX_BODIES = 10
DATASET_DIST_SCALE = 1.1   # camera dist = scale * mesh_radius

PYTHON_BIN = "/Users/neonilllai/projects/SEG_AIM/SEG_env/bin/python"
