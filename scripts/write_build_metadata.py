"""Emit reproducible metadata for the Agentic GLEE wheel release."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import detectron2
import torch


def command_output(*args):
    return subprocess.check_output(args, text=True).strip()


wheel = Path(os.environ["WHEEL_PATH"])
metadata = {
    "artifact": wheel.name,
    "cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "cuda_toolkit": command_output(
        os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"), "--version"
    ).splitlines()[-1],
    "detectron2": detectron2.__version__,
    "glee_commit": command_output("git", "rev-parse", "HEAD"),
    "glibc": " ".join(platform.libc_ver()),
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}
print(json.dumps(metadata, indent=2, sort_keys=True))
