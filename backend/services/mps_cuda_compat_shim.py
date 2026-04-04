"""Deprecated no-op: CUDA/MPS sync is owned by ``ltx2_server._setup_cuda_fallback``.

Earlier versions mapped ``torch.cuda.synchronize`` → ``torch.mps.synchronize`` here.
That triggers Metal ``commit an already committed command buffer`` on PyTorch 2.10
MPS when called frequently. ``_setup_cuda_fallback`` now no-ops synchronize on
non-CUDA builds instead. Encode-time staging uses blocking CPU copies in
``ltx_pipeline_common.encode_video_output``.
"""

from __future__ import annotations


def apply_mps_cuda_compat_shim() -> None:
    """Idempotent no-op; kept for tests and any external callers."""
