#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <assembly_name> [n_views]"
  exit 1
fi

ASSEMBLY="$1"
N_VIEWS="${2:-50}"
OUT_DIR="output/${ASSEMBLY}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate fusion_env || true
fi

python step1_build_labeled_mesh.py --assembly "${ASSEMBLY}" --output_dir "${OUT_DIR}"
python step2_extract_outer_surface.py --input_dir "${OUT_DIR}" --method and
python step3_pytorch3d_renderer.py --input_dir "${OUT_DIR}" --n_views "${N_VIEWS}" --use_outer
python step4_camera_validation.py --input_dir "${OUT_DIR}"
python step5_coverage_balancer.py --input_dir "${OUT_DIR}" --coverage 0.95
python step6_convert_to_nnunet.py --input_dir "${OUT_DIR}" --dataset_id 1

echo "Pipeline completed for ${ASSEMBLY}"
