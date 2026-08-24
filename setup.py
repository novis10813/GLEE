#!/usr/bin/env python3
"""Build the Agentic InstructNav GLEE compatibility package."""
import glob
import os
from pathlib import Path

import torch
from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
OPS_DIR = Path("src") / "glee" / "models" / "pixel_decoder" / "ops"
VERSION = os.environ.get(
    "GLEE_VERSION", "0.1.0+torch2.7.1cu128.sm120"
)


def get_extensions():
    extension_dir = OPS_DIR / "src"
    main_sources = glob.glob(str(extension_dir / "*.cpp"))
    cpu_sources = glob.glob(str(extension_dir / "cpu" / "*.cpp"))
    cuda_sources = glob.glob(str(extension_dir / "cuda" / "*.cu"))
    sources = main_sources + cpu_sources
    extension = CppExtension
    macros = []
    compile_args = {"cxx": []}

    force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
    if (torch.cuda.is_available() or force_cuda) and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += cuda_sources
        macros.append(("WITH_CUDA", None))
        compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise RuntimeError(
            "CUDA 12.8 is required; set CUDA_HOME and FORCE_CUDA=1"
        )

    return [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=[str((ROOT / extension_dir).resolve())],
            define_macros=macros,
            extra_compile_args=compile_args,
        )
    ]


setup(
    name="instructnav-glee-backend",
    version=VERSION,
    description="GLEE compatibility backend for Agentic InstructNav",
    long_description=(ROOT / "README_AGENTIC.md").read_text(),
    long_description_content_type="text/markdown",
    author="FoundationVision and Agentic InstructNav maintainers",
    url="https://github.com/novis10813/GLEE",
    license="MIT",
    python_requires=">=3.12,<3.13",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["glee*"]),
    package_data={"glee": ["configs/*.yaml"]},
    include_package_data=True,
    install_requires=[
        "detectron2==0.6",
        "einops>=0.8,<1",
        "fairscale>=0.4.4,<0.5",
        "numpy>=2.0,<2.4",
        "scipy>=1.13,<2",
        "timm>=0.9,<2",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers>=4.26,<5",
    ],
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": torch.utils.cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=False
        )
    },
    zip_safe=False,
)
