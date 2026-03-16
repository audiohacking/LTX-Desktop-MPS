"""Z-Image-Turbo image generation pipeline wrapper."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from PIL.Image import Image as PILImage

from services.services_utils import ImagePipelineOutputLike, PILImageType, get_device_type

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ZImageOutput:
    images: Sequence[PILImageType]


class ZitImageGenerationPipeline:
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "ZitImageGenerationPipeline":
        return ZitImageGenerationPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        from diffusers.pipelines.auto_pipeline import ZImagePipeline  # type: ignore[reportUnknownVariableType]
        self._device: str | None = None
        self._cpu_offload_active = False
        self.pipeline = ZImagePipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            torch_dtype=torch.bfloat16,
        )
        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        if self._cpu_offload_active:
            # When cpu offload is active, the generator should stay on the target device
            # but we need to be careful with "cuda" vs "mps"
            return self._device or "cpu"
        if self._device is not None:
            return self._device

        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise RuntimeError("Unexpected ZIT pipeline output format: missing images sequence")

        images_list = cast(Sequence[object], images)
        validated_images: list[PILImageType] = []
        for image in images_list:
            if not isinstance(image, PILImage):
                raise RuntimeError("Unexpected ZIT pipeline output format: images must be PIL.Image instances")
            validated_images.append(image)

        return _ZImageOutput(images=validated_images)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        # ZImagePipeline ignores guidance_scale, so we drop it explicitly.
        _ = guidance_scale
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)
        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=0.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device == "cuda":
            # CUDA supports model CPU offloading for reduced VRAM usage
            logger.info("[ZIT] Moving pipeline to CUDA with CPU offload")
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        elif runtime_device == "mps":
            # MPS does NOT support enable_model_cpu_offload (CUDA-only API).
            # Use direct device placement instead, with attention slicing for memory.
            logger.info("[ZIT] Moving pipeline to MPS (direct placement)")
            self.pipeline.to("mps")  # type: ignore[reportUnknownMemberType]
            try:
                self.pipeline.enable_attention_slicing()  # type: ignore[reportUnknownMemberType]
            except Exception as exc:
                logger.warning("[ZIT] Could not enable attention slicing: %s", exc)
            self._cpu_offload_active = False
        else:
            logger.info("[ZIT] Moving pipeline to %s", runtime_device)
            self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device
