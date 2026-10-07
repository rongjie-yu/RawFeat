#!/usr/bin/env bash
# Read-only prerequisites and contract tests; no optimizer updates.
set -euo pipefail
cd "$(dirname "$0")/.."
RAWFEAT_PY="${RAWFEAT_PYTHON:-/homes/rongjie/software/miniconda3/envs/rawfeat/bin/python}"
export CUDA_VISIBLE_DEVICES="${RAWFEAT_GPU:-1}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$RAWFEAT_PY" -m pytest -q tests
"$RAWFEAT_PY" -m rawfeat.cli preflight --config configs/formal.yaml
