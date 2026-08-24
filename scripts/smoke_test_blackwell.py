"""Validate GLEE's deformable-attention extension on Blackwell."""
import sys

import torch

import glee
from glee.models.pixel_decoder.ops.functions.ms_deform_attn_func import (
    MSDeformAttnFunction,
    ms_deform_attn_core_pytorch,
)


assert sys.version_info[:2] == (3, 12), sys.version
assert glee.__version__ == "0.1.0+torch2.7.1cu128.sm120", glee.__version__
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)

torch.manual_seed(3)
shapes = torch.tensor([(6, 4), (3, 2)], dtype=torch.long, device="cuda")
level_start_index = torch.cat(
    (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
)
spatial_size = int(shapes.prod(1).sum())
value = (torch.rand(1, spatial_size, 2, 4, device="cuda") * 0.01).requires_grad_()
sampling_locations = torch.rand(1, 2, 2, 2, 2, 2, device="cuda").requires_grad_()
attention_weights = torch.rand(1, 2, 2, 2, 2, device="cuda") + 1e-5
attention_weights = (
    attention_weights
    / attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
).requires_grad_()

reference = ms_deform_attn_core_pytorch(
    value, shapes, sampling_locations, attention_weights
)
actual = MSDeformAttnFunction.apply(
    value, shapes, level_start_index, sampling_locations, attention_weights, 2
)
torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-3)
actual.backward(torch.ones_like(actual).contiguous())
assert value.grad is not None
assert sampling_locations.grad is not None
assert attention_weights.grad is not None
print("GLEE Blackwell smoke: OK", glee.__version__, torch.cuda.get_device_name())
