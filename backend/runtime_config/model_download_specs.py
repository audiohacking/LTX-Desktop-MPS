"""Canonical model download specs and required-model policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from state.app_state_types import ModelFileType


@dataclass(frozen=True, slots=True)
class ModelFileDownloadSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name


MODEL_FILE_ORDER: tuple[ModelFileType, ...] = (
    "checkpoint",
    "upsampler",
    "text_encoder",
    "zit",
)


DEFAULT_MODEL_DOWNLOAD_SPECS: dict[ModelFileType, ModelFileDownloadSpec] = {
    "checkpoint": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
        expected_size_bytes=43_000_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="Main transformer model",
    ),
    "upsampler": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        expected_size_bytes=1_900_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="2x Upscaler",
    ),
    "text_encoder": ModelFileDownloadSpec(
        relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
        expected_size_bytes=25_000_000_000,
        is_folder=True,
        repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
        description="Gemma text encoder (bfloat16)",
    ),
    "zit": ModelFileDownloadSpec(
        relative_path=Path("Z-Image-Turbo"),
        expected_size_bytes=31_000_000_000,
        is_folder=True,
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        description="Z-Image-Turbo model for text-to-image generation",
    ),
}


DEFAULT_REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = frozenset(
    {"checkpoint", "upsampler", "zit"}
)

# -----------------------------------------------------------------------------
# GGUF model set (Unsloth LTX-2.3-GGUF) for Apple MPS / lower VRAM
# https://huggingface.co/unsloth/LTX-2.3-GGUF
# -----------------------------------------------------------------------------

# GGUF checkpoint filename pattern: ltx-2.3-22b-dev-{Q}.gguf
GGUF_CHECKPOINT_BASENAME = "ltx-2.3-22b-dev"
GGUF_CHECKPOINT_SIZE_BYTES = 15_000_000_000  # ~14 GB typical for Q4

# ComfyUI subfolder names (under a ComfyUI models base path)
COMFYUI_UNET_SUBDIR = "unet"
COMFYUI_VAE_SUBDIR = "vae"
COMFYUI_TEXT_ENCODERS_SUBDIR = "text_encoders"
COMFYUI_LATENT_UPSCALE_SUBDIR = "latent_upscale_models"

# GGUF text encoder: same as reference ComfyUI workflow (Unsloth README)
# text_encoders/ must contain: Gemma GGUF + embeddings connectors safetensors
GGUF_GEMMA_GGUF_GLOB = "gemma-3-12b-it-qat-*.gguf"  # e.g. gemma-3-12b-it-qat-UD-Q4_K_XL.gguf
GGUF_EMBEDDINGS_CONNECTORS_FILENAME = "ltx-2.3-22b-dev_embeddings_connectors.safetensors"


def gguf_checkpoint_filename(quantization: str) -> str:
    """Return the GGUF checkpoint filename for the given quantization (e.g. Q4_K_M)."""
    return f"{GGUF_CHECKPOINT_BASENAME}-{quantization}.gguf"


def get_gguf_model_download_specs(quantization: str) -> dict[ModelFileType, ModelFileDownloadSpec]:
    """Build GGUF model specs for the given quantization (used for download and path layout)."""
    return {
        "checkpoint": ModelFileDownloadSpec(
            relative_path=Path(gguf_checkpoint_filename(quantization)),
            expected_size_bytes=GGUF_CHECKPOINT_SIZE_BYTES,
            is_folder=False,
            repo_id="unsloth/LTX-2.3-GGUF",
            description="Main transformer model (GGUF quantized)",
        ),
        "upsampler": ModelFileDownloadSpec(
            relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            expected_size_bytes=1_900_000_000,
            is_folder=False,
            repo_id="Lightricks/LTX-2.3",
            description="2x Upscaler",
        ),
        # Same layout as reference ComfyUI workflow: text_encoders/ with Gemma GGUF + embeddings connectors
        "text_encoder": ModelFileDownloadSpec(
            relative_path=Path(COMFYUI_TEXT_ENCODERS_SUBDIR),
            expected_size_bytes=3_000_000_000,  # connectors ~2.3GB + Gemma GGUF varies
            is_folder=True,
            repo_id="unsloth/LTX-2.3-GGUF",
            description="Text encoders dir: Gemma GGUF + ltx-2.3-22b-dev_embeddings_connectors.safetensors (same as ComfyUI)",
        ),
        "zit": ModelFileDownloadSpec(
            relative_path=Path("Z-Image-Turbo"),
            expected_size_bytes=31_000_000_000,
            is_folder=True,
            repo_id="Tongyi-MAI/Z-Image-Turbo",
            description="Z-Image-Turbo model for text-to-image generation",
        ),
    }


def resolve_gguf_text_encoder_paths(text_encoder_dir: Path) -> tuple[Path, Path] | None:
    """Resolve Gemma GGUF and embeddings connectors paths from the text_encoders dir (reference ComfyUI layout).

    Returns (gemma_gguf_path, connectors_path) if both exist, else None.
    """
    if not text_encoder_dir.is_dir():
        return None
    candidates = list(text_encoder_dir.glob(GGUF_GEMMA_GGUF_GLOB))
    if not candidates:
        return None
    gemma_gguf = candidates[0]
    connectors = text_encoder_dir / GGUF_EMBEDDINGS_CONNECTORS_FILENAME
    if not connectors.is_file():
        return None
    return (gemma_gguf, connectors)


# Legacy alias: default GGUF specs (Q4_K_M)
GGUF_MODEL_DOWNLOAD_SPECS: dict[ModelFileType, ModelFileDownloadSpec] = get_gguf_model_download_specs("Q4_K_M")

# GGUF video workflow (unsloth/LTX-2.3-GGUF) requires unet + text_encoders (+ optional upscaler).
# Z-Image-Turbo (zit) is for text-to-image only, not part of LTX-2.3-GGUF.
GGUF_REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = frozenset(
    {"checkpoint", "upsampler"}
)


def resolve_required_model_types(
    base_required: frozenset[ModelFileType],
    has_api_key: bool,
    use_local_text_encoder: bool = False,
) -> frozenset[ModelFileType]:
    if not base_required:
        return base_required
    if has_api_key and not use_local_text_encoder:
        return base_required
    return cast(frozenset[ModelFileType], base_required | {"text_encoder"})
