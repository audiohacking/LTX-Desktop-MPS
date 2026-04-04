"""Map torch.cuda sync/cache APIs to MPS when CUDA is unavailable.

ltx-pipelines (DistilledPipeline, cleanup_memory, etc.) call ``torch.cuda.synchronize()``
and ``torch.cuda.empty_cache()`` at stage boundaries. On Apple Silicon, CUDA is not
compiled in: ``torch.cuda.synchronize()`` raises, or if it were a no-op, **MPS work
would never be synchronized** before the next stage or before tensors are copied to
CPU for encoding. That can produce visibly corrupted output that shows up more often
when generation uses **more VAE chunks** (e.g. crossing from ~5s to 6s at 24fps:
121→145 frames, 3→4 chunks).

This module patches those entry points to call ``torch.mps.*`` when MPS is active
and CUDA is not. CUDA systems are unchanged.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_shim_applied = False


def apply_mps_cuda_compat_shim() -> None:
    """Idempotent; safe to call multiple times."""
    global _shim_applied
    if _shim_applied:
        return
    if torch.cuda.is_available():
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    def synchronize() -> None:
        try:
            torch.mps.synchronize()
        except Exception:
            logger.warning("torch.mps.synchronize() failed", exc_info=True)

    def empty_cache() -> None:
        try:
            torch.mps.empty_cache()
        except Exception:
            logger.warning("torch.mps.empty_cache() failed", exc_info=True)

    torch.cuda.synchronize = synchronize  # type: ignore[method-assign]
    torch.cuda.empty_cache = empty_cache  # type: ignore[method-assign]
    _shim_applied = True
    logger.info("Applied MPS↔torch.cuda sync/cache shim (ltx-pipelines stage boundaries)")
