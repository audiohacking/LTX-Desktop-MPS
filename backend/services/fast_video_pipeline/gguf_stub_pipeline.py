"""Fallback stub when in-process GGUF load fails (e.g. missing diffusers GGUF support).

When LTX_USE_GGUF=1, the app first tries to load the .gguf checkpoint in-process
(convert to safetensors and run the normal pipeline). If that fails, this stub
is used so the app still starts; generate/warmup then raise with setup instructions.
"""

from __future__ import annotations

from typing import Final

from api_types import ImageConditioningInput

_MSG = (
    "GGUF in-process load failed at startup. "
    "Install diffusers with GGUF support (pip install -U diffusers), ensure the VAE files are in the vae/ folder next to unet/, then restart. "
    "Or run without LTX_USE_GGUF for standard safetensors inference."
)


class GGUFStubFastVideoPipeline:
    """Stub that satisfies FastVideoPipeline protocol; raises on generate/warmup."""

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: object,
    ) -> "GGUFStubFastVideoPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device
        return GGUFStubFastVideoPipeline()

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
        raise RuntimeError(_MSG)

    def warmup(self, output_path: str) -> None:
        raise RuntimeError(_MSG)

    def compile_transformer(self) -> None:
        pass
