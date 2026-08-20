#!/usr/bin/env bash
set -euo pipefail

if [[ ${CONDA_DEFAULT_ENV:-} != graspnet ]]; then
  echo "Activate the graspnet Conda environment before running this script." >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ -d "$script_dir/../graspnet_baseline" ]]; then
  baseline_dir=$(cd -- "$script_dir/../graspnet_baseline" && pwd)
else
  baseline_dir=$(cd -- "$script_dir/../../share/graspnet_source/graspnet_baseline" && pwd)
fi

python -c 'import numpy, open3d, PIL, scipy, torch'
python -c 'import torch; assert torch.cuda.is_available(), "GraspNet PointNet2 requires a visible CUDA device; build on the target GPU host."'
(cd "$baseline_dir/pointnet2" && python setup.py build_ext --inplace)
(cd "$baseline_dir/knn" && python setup.py build_ext --inplace)
