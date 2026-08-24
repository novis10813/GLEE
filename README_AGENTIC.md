# GLEE backend for Agentic InstructNav

This compatibility branch packages the GLEE inference snapshot used by
Agentic InstructNav as an installable wheel. It preserves the output behavior
of the `GLEE_SwinL_Scaleup10m.pth` checkpoint while removing the need to keep
GLEE source inside the navigation repository.

## Supported ABI

| Component | Supported value |
| --- | --- |
| Python | CPython 3.12 |
| PyTorch | 2.7.1+cu128 |
| CUDA toolkit | 12.8 |
| GPU code | `sm_120` |
| Detectron2 | `0.6+torch2.7.1cu128.sm120` |
| Platform | Linux x86-64 |

The distribution is named `instructnav-glee-backend` and exposes the `glee`
Python package plus the `MultiScaleDeformableAttention` extension. It does not
include GLEE checkpoints, CLIP weights, PyTorch, CUDA, or NVIDIA drivers.

## Build

Install the matching Torch and Detectron2 wheels first, then run:

```bash
uv pip install --python .venv/bin/python build ninja setuptools wheel
CUDA_HOME=/usr/local/cuda-12.8 \
PYTHON=.venv/bin/python \
bash scripts/build_blackwell_wheel.sh
```

Build isolation stays disabled because PyTorch supplies the extension build API
and native ABI. The command writes a wheel, `SHA256SUMS`, and
`build-metadata.json` under `dist/`.

## Test

```bash
uv pip install --python .venv/bin/python dist/instructnav_glee_backend-*.whl
.venv/bin/python scripts/smoke_test_blackwell.py
```

The smoke test compares the custom CUDA deformable-attention output with its
PyTorch reference implementation on an RTX 5090.

## Source provenance

`src/glee` comes from the GLEE snapshot committed to Agentic InstructNav in
commit `ce81c124dc6e28c6ad4b8e96fc88cc056970742f`. Fifty of its 54 tracked
source/config blobs occur in the FoundationVision/GLEE upstream history. The
remaining compatibility changes preserve the checkpoint's older inference
path and update removed PyTorch tensor APIs in:

- `glee/config.py`
- `glee/models/glee_model.py`
- `glee/models/pixel_decoder/ops/src/ms_deform_attn.h`
- `glee/models/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu`

The FoundationVision/GLEE code is MIT licensed. The deformable-attention source
retains its upstream Apache-2.0 and copyright headers.
