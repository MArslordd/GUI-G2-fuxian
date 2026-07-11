#!/usr/bin/env bash
set -euo pipefail

# Stable environment for formal GUI-G2 verl training with real flash-attn.
# Defaults target a conservative CUDA 12.4 / torch 2.6 stack instead of very
# new CUDA 13 stacks, because flash-attn and vLLM wheels are more available.

ENV_NAME=${ENV_NAME:-gui-g2-verl-fa}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
TORCH_VERSION=${TORCH_VERSION:-2.6.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.21.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.6.0}
CUDA_WHEEL=${CUDA_WHEEL:-cu124}
VLLM_VERSION=${VLLM_VERSION:-0.8.5.post1}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.7.4.post1}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_WHEEL}}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_PATH=${VERL_PATH:-$(cd "${REPO_ROOT}/.." && pwd)/verl}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Please load Miniconda/Anaconda first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda env ${ENV_NAME} already exists; reusing it."
else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel packaging ninja psutil \
    -i "${PIP_INDEX_URL}"

python -m pip install \
    "torch==${TORCH_VERSION}+${CUDA_WHEEL}" \
    "torchvision==${TORCHVISION_VERSION}+${CUDA_WHEEL}" \
    "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_WHEEL}" \
    --index-url "${TORCH_INDEX_URL}"

python -m pip install -r "${REPO_ROOT}/requirements-verl.txt" -i "${PIP_INDEX_URL}"

# vLLM wheels are tightly coupled to torch. Keep this version aligned with the
# torch default above unless you intentionally upgrade the whole stack.
python -m pip install "vllm==${VLLM_VERSION}" -i "${PIP_INDEX_URL}"

if [ -d "${VERL_PATH}" ]; then
    python -m pip install -e "${VERL_PATH}" -i "${PIP_INDEX_URL}"
else
    echo "verl source tree was not found at ${VERL_PATH}." >&2
    echo "Set VERL_PATH=/path/to/verl and rerun this script." >&2
    exit 1
fi

# Prefer a prebuilt flash-attn wheel. If no wheel matches, pip may compile from
# source; keep MAX_JOBS modest to avoid exhausting RAM on shared servers.
MAX_JOBS=${MAX_JOBS:-8} python -m pip install \
    "flash-attn==${FLASH_ATTN_VERSION}" \
    --no-build-isolation \
    -i "${PIP_INDEX_URL}"

python - <<'PY'
import sys
import torch
import flash_attn
import vllm
from flash_attn.bert_padding import unpad_input

print("python:", sys.version)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
print("vllm:", vllm.__version__)
print("flash_attn.bert_padding ok")
PY

echo
echo "Environment ready: conda activate ${ENV_NAME}"
