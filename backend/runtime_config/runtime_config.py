"""Runtime configuration model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from runtime_config.model_download_specs import (
    COMFYUI_LATENT_UPSCALE_SUBDIR,
    COMFYUI_TEXT_ENCODERS_SUBDIR,
    COMFYUI_UNET_SUBDIR,
    ModelFileDownloadSpec,
    gguf_checkpoint_filename,
)
from state.app_state_types import ModelFileType

# Max depth to search under ComfyUI base for model files/dirs (avoids unbounded scans)
_COMFYUI_SEARCH_DEPTH = 8


def _find_under_base(base: Path, name: str, is_folder: bool) -> Path | None:
    """Find a file or directory named *name* anywhere under *base* (bounded depth, then rglob fallback)."""
    if not base.is_dir():
        return None
    for depth in range(1, _COMFYUI_SEARCH_DEPTH + 1):
        parts = ["*"] * depth + [name]
        pattern = "/".join(parts)
        try:
            for p in base.glob(pattern):
                if p.is_dir() if is_folder else p.is_file():
                    return p
        except OSError:
            continue
    # Fallback: search anywhere under base (e.g. user selected ComfyUI/ and file is in models/unet/...)
    try:
        for p in base.rglob(name):
            if p.is_dir() if is_folder else p.is_file():
                return p
    except OSError:
        pass
    return None


@dataclass
class RuntimeConfig:
    device: torch.device
    models_dir: Path
    model_download_specs: Mapping[ModelFileType, ModelFileDownloadSpec]
    required_model_types: frozenset[ModelFileType]
    outputs_dir: Path
    ic_lora_dir: Path
    settings_file: Path
    ltx_api_base_url: str
    force_api_generations: bool
    use_sage_attention: bool
    use_gguf: bool
    gguf_quantization: str
    comfyui_models_base: Path | None
    camera_motion_prompts: dict[str, str]
    default_negative_prompt: str

    def spec_for(self, model_type: ModelFileType) -> ModelFileDownloadSpec:
        return self.model_download_specs[model_type]

    def model_path(self, model_type: ModelFileType) -> Path:
        """Return the path to the model file or folder. Uses ComfyUI layout when comfyui_models_base is set.
        If the canonical path does not exist, searches under the ComfyUI base for the expected filename/dir
        in any subdirectory (bounded depth) so users can point at a folder that contains models in subdirs.
        """
        spec = self.spec_for(model_type)
        canonical = self.models_dir / spec.relative_path
        base = self.comfyui_models_base
        if base is None or model_type not in ("checkpoint", "upsampler", "text_encoder"):
            return canonical
        # Canonical ComfyUI layout paths
        if model_type == "checkpoint":
            canonical = base / COMFYUI_UNET_SUBDIR / gguf_checkpoint_filename(self.gguf_quantization)
        elif model_type == "upsampler":
            canonical = base / COMFYUI_LATENT_UPSCALE_SUBDIR / spec.relative_path.name
        elif model_type == "text_encoder":
            canonical = base / COMFYUI_TEXT_ENCODERS_SUBDIR
        if canonical.exists() and (canonical.is_dir() if spec.is_folder else canonical.is_file()):
            return canonical
        # Search under base for the expected name in any subdirectory
        name = spec.relative_path.name if spec.is_folder else spec.relative_path.name
        if model_type == "checkpoint":
            name = gguf_checkpoint_filename(self.gguf_quantization)
        found = _find_under_base(base, name, spec.is_folder)
        if found is not None:
            return found
        return canonical

    def model_path_with_base(self, model_type: ModelFileType, comfyui_base: Path) -> Path:
        """Same as model_path but using the given base instead of comfyui_models_base. Used for one-off scan at a chosen path."""
        spec = self.spec_for(model_type)
        canonical = self.models_dir / spec.relative_path
        if model_type not in ("checkpoint", "upsampler", "text_encoder"):
            return canonical
        base = comfyui_base
        if model_type == "checkpoint":
            canonical = base / COMFYUI_UNET_SUBDIR / gguf_checkpoint_filename(self.gguf_quantization)
        elif model_type == "upsampler":
            canonical = base / COMFYUI_LATENT_UPSCALE_SUBDIR / spec.relative_path.name
        elif model_type == "text_encoder":
            canonical = base / COMFYUI_TEXT_ENCODERS_SUBDIR
        if canonical.exists() and (canonical.is_dir() if spec.is_folder else canonical.is_file()):
            return canonical
        name = spec.relative_path.name
        if model_type == "checkpoint":
            name = gguf_checkpoint_filename(self.gguf_quantization)
        found = _find_under_base(base, name, spec.is_folder)
        if found is not None:
            return found
        return canonical

    @property
    def downloading_dir(self) -> Path:
        return self.models_dir / ".downloading"

    def downloading_path(self, model_type: ModelFileType) -> Path:
        """Return the staging path under downloading_dir for a model type."""
        spec = self.spec_for(model_type)
        if spec.is_folder:
            return self.downloading_dir / spec.relative_path
        return self.downloading_dir
