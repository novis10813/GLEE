#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
GLEE_VERSION=${GLEE_VERSION:-0.1.0+torch2.7.1cu128.sm120}
MAX_JOBS=${MAX_JOBS:-$(nproc)}

[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
  printf 'CUDA compiler not found: %s/bin/nvcc\n' "$CUDA_HOME" >&2
  exit 1
}

export CUDA_HOME TORCH_CUDA_ARCH_LIST GLEE_VERSION MAX_JOBS FORCE_CUDA=1
export PATH="$CUDA_HOME/bin:$PATH"

# Run outside this upstream checkout so its historical top-level detectron2
# source cannot shadow the installed compatibility wheel.
(
  cd /tmp
  "$PYTHON" - <<'PY'
import sys
import torch
import detectron2

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert detectron2.__version__ == "0.6+torch2.7.1cu128.sm120", detectron2.__version__
PY
)

cd "$ROOT_DIR"
rm -rf build dist
"$PYTHON" -m build --wheel --no-isolation
wheel=$(find dist -maxdepth 1 -name '*.whl' -print -quit)
[[ -n "$wheel" ]] || { printf 'wheel was not produced\n' >&2; exit 1; }
sha256sum "$wheel" > dist/SHA256SUMS
WHEEL_PATH="$wheel" "$PYTHON" scripts/write_build_metadata.py > dist/build-metadata.json
printf 'Built %s\n' "$wheel"
