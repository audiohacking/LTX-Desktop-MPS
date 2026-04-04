"""MPS/CUDA compat shim: must not break CPU/CI; idempotent."""

from __future__ import annotations

from services.mps_cuda_compat_shim import apply_mps_cuda_compat_shim


def test_apply_shim_idempotent() -> None:
    apply_mps_cuda_compat_shim()
    apply_mps_cuda_compat_shim()
