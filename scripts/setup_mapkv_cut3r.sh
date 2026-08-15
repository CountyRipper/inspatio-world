#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${MAPKV_CUT3R_ENV_NAME:-mapkv_cut3r}"
VMEM_ROOT="${VMEM_ROOT:-${REPO_ROOT}/third_party/vmem}"
VMEM_COMMIT="39291e4f272f6b4f270691d930926ab5930f942e"
CUT3R_URL="https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"

mkdir -p "$(dirname "${VMEM_ROOT}")"
if [[ ! -d "${VMEM_ROOT}/.git" ]]; then
  git clone https://github.com/runjiali-rl/vmem.git "${VMEM_ROOT}"
fi
ACTUAL_COMMIT="$(git -C "${VMEM_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${VMEM_COMMIT}" ]]; then
  git -C "${VMEM_ROOT}" switch --detach "${VMEM_COMMIT}"
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.11 cmake=3.14.0 -y
fi
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${VMEM_ROOT}/requirements.txt"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install gdown

CUT3R_SRC="${VMEM_ROOT}/extern/CUT3R/src"
(
  cd "${CUT3R_SRC}/croco/models/curope"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python setup.py build_ext --inplace
)

CHECKPOINT="${CUT3R_SRC}/cut3r_512_dpt_4_64.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  "${CONDA_BIN}" run -n "${ENV_NAME}" gdown --fuzzy "${CUT3R_URL}" \
    --output "${CHECKPOINT}"
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
  'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available())'
CUT3R_PYTHON_PATH="$("${CONDA_BIN}" run -n "${ENV_NAME}" which python)"
echo "CUT3R_PYTHON=${CUT3R_PYTHON_PATH}"
echo "VMEM_ROOT=${VMEM_ROOT}"
echo "CUT3R_CHECKPOINT=${CHECKPOINT}"
