"""Build script for joyomni_ops — a minimal CUDA/ROCm operator library
extracted from sgl-kernel for the JoyOmni V2V DiT pipeline.

Ops provided (no sgl-kernel / sglang dependency):
  - fused_qk_norm_rope_3d_paired : fused RMSNorm(q,k) + 3D RoPE   (bf16)
  - fused_norm_scale_shift       : fused LayerNorm/RMSNorm + adaLN modulate
  - rmsnorm                      : standalone RMSNorm
  - sgl_per_token_quant_fp8      : dynamic per-token bf16 -> fp8_e4m3 quant (CUDA)
  - fp8_scaled_mm                : FP8 per-token x per-channel scaled GEMM (CUDA/cutlass)

NVIDIA GPU arch coverage is chosen from the local nvcc version:
  - always: sm_80, sm_89, sm_90
  - CUDA >= 12.8: also sm_100a (B200) and sm_120a (RTX 5090)
So building on a CUDA 12.8+ toolchain automatically yields Blackwell support.

On ROCm, PyTorch's CUDAExtension hipifies the three light kernels.  The native
CUTLASS FP8 files are omitted; gfx950 FP8 uses torch._scaled_mm/hipBLASLt from
the Python wrapper instead.
"""
import os
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

THIS_DIR = Path(__file__).parent.resolve()
IS_ROCM = torch.version.hip is not None


def _cuda_version():
    """(major, minor) of the nvcc that torch will use, or (0, 0) if unknown."""
    from torch.utils.cpp_extension import CUDA_HOME
    import subprocess
    try:
        nvcc = os.path.join(CUDA_HOME or "/usr/local/cuda", "bin", "nvcc")
        out = subprocess.check_output([nvcc, "--version"], text=True)
        for line in out.splitlines():
            if "release" in line:
                tok = line.split("release")[1].strip().split(",")[0]  # "12.6"
                mj, mn = tok.split(".")[:2]
                return int(mj), int(mn)
    except Exception:
        pass
    return (0, 0)


def _gencodes():
    """-gencode flags. Blackwell (sm_100a/sm_120a) needs nvcc >= 12.8."""
    mj, mn = _cuda_version()
    # Allow override, e.g. JOYOMNI_OPS_CUDA_ARCHS="90;120a"
    override = os.environ.get("JOYOMNI_OPS_CUDA_ARCHS")
    if override:
        flags = []
        for a in override.split(";"):
            a = a.strip()
            if not a:
                continue
            code = f"sm_{a}"
            arch = f"compute_{a}"
            flags += [f"-gencode=arch={arch},code={code}"]
        return flags
    flags = [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    if (mj, mn) >= (12, 8):
        flags += [
            "-gencode=arch=compute_100a,code=sm_100a",
            "-gencode=arch=compute_120a,code=sm_120a",
            # PTX fallback so newer archs (e.g. sm_121) can JIT.
            "-gencode=arch=compute_120,code=compute_120",
        ]
    else:
        # No SASS for sm_100/sm_120 on this toolchain; embed sm_90 PTX so the
        # driver can JIT for Blackwell (sm_100/sm_120) at load time. Correctness
        # only — for tuned Blackwell SASS, build on CUDA >= 12.8.
        flags += ["-gencode=arch=compute_90,code=compute_90"]
        print(
            f"[joyomni_ops] nvcc {mj}.{mn} < 12.8: SASS sm_80/89/90 + sm_90 PTX "
            f"(JIT fallback for Blackwell). Build on CUDA >= 12.8 for native sm_100a/sm_120a."
        )
    return flags


SOURCES = [
    "csrc/pybind.cpp",
    "csrc/fused_qknorm_rope_3d_kernel.cu",
    "csrc/fused_joint_qkv_pack.cu",
    "csrc/fused_norm_scale_shift.cu",
    "csrc/rmsnorm.cu",
]

# FP8 GEMM needs NVIDIA cutlass (and CUDA >= 12.8 for Blackwell).  ROCm uses
# PyTorch's hipBLASLt-backed scaled GEMM and therefore always builds the light
# extension without these two CUDA-only translation units.
NO_FP8_REQUESTED = os.environ.get("JOYOMNI_OPS_NO_FP8", "").lower() in {
    "1", "true", "yes", "on"
}
NO_FP8 = NO_FP8_REQUESTED or IS_ROCM
if IS_ROCM:
    # gfx950 supports the OCP e4m3 format used by torch.float8_e4m3fn.  Keep
    # the CUTLASS GEMM disabled, but build the small native row-quantizer so
    # every Fp8Linear does not need to launch an Inductor/Triton reduction.
    SOURCES += ["csrc/per_token_quant_fp8_rocm.hip"]
elif not NO_FP8:
    SOURCES += ["csrc/per_token_quant_fp8.cu", "csrc/fp8_gemm.cu"]

if IS_ROCM:
    rocm_archs = os.environ.get("JOYOMNI_OPS_ROCM_ARCHS")
    if rocm_archs:
        # CUDAExtension's ROCm flag generator consumes this variable.  Example:
        # JOYOMNI_OPS_ROCM_ARCHS=gfx950 python -m pip install .
        os.environ["PYTORCH_ROCM_ARCH"] = rocm_archs
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
    ]
    # TheRock installs the compiler and device bitcode in Python packages
    # rather than a monolithic /opt/rocm tree.  CUDAExtension locates hipcc,
    # but clang may still need the bitcode directory spelled out explicitly.
    rocm_root = Path(ROCM_HOME) if ROCM_HOME else None
    device_lib_candidates = []
    if rocm_root is not None:
        device_lib_candidates += [
            rocm_root / "amdgcn" / "bitcode",
            rocm_root / "lib" / "llvm" / "amdgcn" / "bitcode",
            rocm_root.parent / "_rocm_sdk_devel" / "lib" / "llvm" / "amdgcn" / "bitcode",
        ]
    for device_lib_dir in device_lib_candidates:
        if (device_lib_dir / "ocml.bc").is_file():
            nvcc_flags.append(f"--rocm-device-lib-path={device_lib_dir}")
            break
    print(
        "[joyomni_ops] ROCm detected: building wavefront-aware light kernels; "
        "FP8 GEMM will use PyTorch/hipBLASLt"
    )
else:
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ] + _gencodes()

cxx_flags = ["-O3", "-std=c++17"]
if NO_FP8:
    nvcc_flags.append("-DJOYOMNI_OPS_NO_FP8")
    cxx_flags.append("-DJOYOMNI_OPS_NO_FP8")
if IS_ROCM:
    nvcc_flags.append("-DJOYOMNI_OPS_ROCM_QUANT")
    cxx_flags.append("-DJOYOMNI_OPS_ROCM_QUANT")

include_dirs = [str(THIS_DIR / "include")]
library_dirs = []
if IS_ROCM and rocm_root is not None:
    rocm_include_candidates = [
        rocm_root / "include",
        rocm_root.parent / "_rocm_sdk_devel" / "include",
    ]
    include_dirs += [
        str(path) for path in rocm_include_candidates
        if (path / "hipblas" / "hipblas.h").is_file()
    ]
    rocm_library_candidates = [
        rocm_root / "lib",
        rocm_root.parent / "_rocm_sdk_devel" / "lib",
    ]
    library_dirs += [
        str(path) for path in rocm_library_candidates
        if (path / "libamdhip64.so").is_file()
    ]

# cutlass (for fp8_scaled_mm). Point JOYOMNI_OPS_CUTLASS_DIR at a cutlass checkout
# (tag 57e3cfb47a2d9e0d46eb6335c3dc411498efa198 matches the reference build).
cutlass_dir = os.environ.get("JOYOMNI_OPS_CUTLASS_DIR")
if not NO_FP8:
    if cutlass_dir:
        include_dirs += [
            os.path.join(cutlass_dir, "include"),
            os.path.join(cutlass_dir, "tools", "util", "include"),
        ]
    else:
        print("[joyomni_ops] JOYOMNI_OPS_CUTLASS_DIR not set — fp8_gemm.cu will fail to "
              "compile. Set it, or build with JOYOMNI_OPS_NO_FP8=1.")

setup(
    name="joyomni_ops",
    version="0.1.0",
    description="Minimal self-contained CUDA/ROCm ops for the JoyOmni V2V DiT pipeline",
    packages=["joyomni_ops"],
    ext_modules=[
        CUDAExtension(
            name="joyomni_ops._C",
            sources=SOURCES,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
