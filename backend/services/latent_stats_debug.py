"""Latent tensor diagnostics for MPS / long-clip debugging (opt-in only).

**Do not leave hooks enabled in shipped builds.** Wrapping
``denoise_audio_video`` / ``vae_decode_video`` perturbs MPS/Metal scheduling and
has caused crashes during VAE decode (Metal command-buffer asserts) even when
logging only tensor metadata.

Enable **one** of:

* Set ``LATENT_STATS_DEBUG_ENABLED = True`` in this module (local dev / custom
  bundles), or
* Set env ``LTX_DEBUG_LATENT_STATS=1`` when launching Python.

Disable: default (both off), or ``LTX_DEBUG_LATENT_STATS=0``.

Patches ``ltx_pipelines.distilled`` so each distilled generation logs:

* After **each** ``denoise_audio_video`` return (stage 1 and stage 2): video
  and audio latent tensors.
* Immediately **before** ``vae_decode_video``: the video latent passed to the VAE.

On **MPS**, only **metadata** is logged (shape, dtype, device, numel). We do
**not** copy tensors to CPU or run ``isfinite`` on device: doing so before the
next GPU op triggers Metal
``commit an already committed command buffer`` (PyTorch 2.10). CUDA/CPU tensors
still get full numeric stats via a CPU float copy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

# Opt-in: True patches distilled (risky on MPS). Prefer env LTX_DEBUG_LATENT_STATS=1 for CLI dev.
LATENT_STATS_DEBUG_ENABLED = False

logger = logging.getLogger("latent_stats_debug")

_hooks_installed = False
_denoise_pass: list[int] = [0]
_saved_denoise: Any | None = None
_saved_decode: Any | None = None


def latent_stat_dict(t: torch.Tensor) -> dict[str, Any]:
    """Scalar summary of ``t`` for logging and tests.

    For **MPS** tensors, skips numeric stats (no ``.cpu()`` / heavy device ops).
    """
    detached = t.detach()
    n = int(detached.numel())
    base = {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": n,
    }
    if n == 0:
        return {
            **base,
            "finite_frac": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    if detached.device.type == "mps":
        return {
            **base,
            "finite_frac": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    finite = torch.isfinite(detached)
    n_fin = int(finite.sum().item())
    cpu = detached.float().cpu()
    cpu_fin = cpu[torch.isfinite(cpu)]
    if cpu_fin.numel() == 0:
        return {
            **base,
            "finite_frac": float(n_fin) / float(n) if n else 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        **base,
        "finite_frac": float(n_fin) / float(n),
        "min": float(cpu_fin.min().item()),
        "max": float(cpu_fin.max().item()),
        "mean": float(cpu_fin.mean().item()),
        "std": float(cpu_fin.std().item()),
    }


def _log_latent(label: str, t: torch.Tensor) -> None:
    stats = latent_stat_dict(t)
    if t.detach().device.type == "mps":
        logger.info(
            "[LTX_DEBUG_LATENT_STATS] %s | shape=%s dtype=%s device=%s numel=%s "
            "(no MPS numeric read — avoids Metal command-buffer assert)",
            label,
            stats["shape"],
            stats["dtype"],
            stats["device"],
            stats["numel"],
        )
        return
    logger.info(
        "[LTX_DEBUG_LATENT_STATS] %s | shape=%s dtype=%s device=%s numel=%s "
        "finite_frac=%.6f min=%s max=%s mean=%s std=%s",
        label,
        stats["shape"],
        stats["dtype"],
        stats["device"],
        stats["numel"],
        float(stats["finite_frac"] or 0.0),
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["std"],
    )


def uninstall_latent_stats_hooks() -> None:
    """Restore ``ltx_pipelines.distilled`` bindings (tests / teardown)."""
    global _hooks_installed, _saved_denoise, _saved_decode
    if not _hooks_installed or _saved_denoise is None or _saved_decode is None:
        _hooks_installed = False
        _saved_denoise = None
        _saved_decode = None
        return
    import ltx_pipelines.distilled as distilled_mod

    distilled_mod.denoise_audio_video = _saved_denoise  # type: ignore[method-assign]
    distilled_mod.vae_decode_video = _saved_decode  # type: ignore[method-assign]
    _hooks_installed = False
    _saved_denoise = None
    _saved_decode = None


def install_latent_stats_hooks() -> None:
    """Patch ``ltx_pipelines.distilled`` callables; idempotent."""
    global _hooks_installed, _saved_denoise, _saved_decode
    if _hooks_installed:
        return
    if os.environ.get("LTX_DEBUG_LATENT_STATS") == "0":
        return
    if not LATENT_STATS_DEBUG_ENABLED and os.environ.get("LTX_DEBUG_LATENT_STATS") != "1":
        return

    import ltx_pipelines.distilled as distilled_mod

    _orig_denoise = distilled_mod.denoise_audio_video
    _orig_decode = distilled_mod.vae_decode_video
    _saved_denoise = _orig_denoise
    _saved_decode = _orig_decode

    def _wrapped_denoise(*args: Any, **kwargs: Any) -> Any:
        out = _orig_denoise(*args, **kwargs)
        _denoise_pass[0] += 1
        p = _denoise_pass[0]
        video_st, audio_st = out
        _log_latent(f"distilled after denoise_audio_video pass={p} video_state.latent", video_st.latent)
        _log_latent(f"distilled after denoise_audio_video pass={p} audio_state.latent", audio_st.latent)
        return out

    def _wrapped_decode(latent: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        _log_latent("distilled vae_decode_video input (video latent)", latent)
        try:
            return _orig_decode(latent, *args, **kwargs)
        finally:
            _denoise_pass[0] = 0

    distilled_mod.denoise_audio_video = _wrapped_denoise  # type: ignore[method-assign]
    distilled_mod.vae_decode_video = _wrapped_decode  # type: ignore[method-assign]

    _hooks_installed = True
    logger.info(
        "latent_stats_debug: patched ltx_pipelines.distilled "
        "(denoise_audio_video, vae_decode_video)"
    )
