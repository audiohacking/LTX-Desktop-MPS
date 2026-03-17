"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


def _resolve_checkpoint_path_for_pipeline(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[str, str | None]:
    """If checkpoint is .gguf, create merged safetensors and return (path, temp_path_to_cleanup). Else return (checkpoint_path, None)."""
    import logging
    _log = logging.getLogger(__name__)
    if not checkpoint_path.strip().lower().endswith(".gguf"):
        _log.info("Using safetensors checkpoint (not GGUF): %s", checkpoint_path)
        return checkpoint_path, None
    _log.info("Loading GGUF checkpoint from: %s (will merge with VAE, then load to pipeline)", checkpoint_path)
    vae_base: Path | None = Path(checkpoint_path).resolve().parent.parent / "vae"
    if not vae_base.is_dir():
        vae_base = None
    try:
        from services.gguf.loader import create_merged_checkpoint_from_gguf
        # Load GGUF to CPU only during conversion (see backend/docs/GGUF_MEMORY.md). ComfyUI
        # does the same (load to RAM first); we avoid filling MPS until the pipeline loads the merged file.
        merged = create_merged_checkpoint_from_gguf(
            checkpoint_path,
            vae_base,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        _log.info("GGUF merged to temp safetensors; pipeline will load from that path (transformer is from GGUF, not default LTX).")
        return merged, merged
    except Exception as e:
        raise RuntimeError(
            f"Failed to load GGUF checkpoint in-process: {e}. "
            "Ensure diffusers has GGUF support (pip install -U diffusers) and the VAE files are in the vae/ folder next to unet/."
        ) from e


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
        )

    def __init__(self, checkpoint_path: str, gemma_root: str | None, upsampler_path: str, device: torch.device) -> None:
        import logging
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        _log = logging.getLogger(__name__)
        self._gguf_temp_path: str | None = None
        load_path, self._gguf_temp_path = _resolve_checkpoint_path_for_pipeline(checkpoint_path, device)
        from_gguf = self._gguf_temp_path is not None
        _log.info("Pipeline loading checkpoint from: %s (from_gguf=%s). Gemma text encoder: %s", load_path, from_gguf, gemma_root)
        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=load_path,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def __del__(self) -> None:
        if getattr(self, "_gguf_temp_path", None) and os.path.exists(self._gguf_temp_path):
            try:
                os.unlink(self._gguf_temp_path)
            except OSError:
                pass

    def compile_transformer(self) -> None:
        transformer = self.pipeline.model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),  # type: ignore[reportUnknownMemberType]
        )
        setattr(self.pipeline.model_ledger, "transformer", lambda: compiled)
