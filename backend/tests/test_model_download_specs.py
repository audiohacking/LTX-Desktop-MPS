"""Tests for model download spec consistency and RuntimeConfig path derivation."""

from __future__ import annotations

from typing import get_args

from state import RuntimeConfig
from runtime_config.model_download_specs import (
    DEFAULT_MODEL_DOWNLOAD_SPECS,
    DEFAULT_REQUIRED_MODEL_TYPES,
    MODEL_FILE_ORDER,
    resolve_required_model_types,
)
from state.app_state_types import ModelFileType


def _build_config(tmp_path, use_gguf: bool = False, gguf_quantization: str = "Q4_K_M", comfyui_models_base=None):
    from runtime_config.model_download_specs import (
        get_gguf_model_download_specs,
        GGUF_REQUIRED_MODEL_TYPES,
    )

    models_dir = tmp_path / "models"
    specs = get_gguf_model_download_specs(gguf_quantization) if use_gguf else DEFAULT_MODEL_DOWNLOAD_SPECS
    required = GGUF_REQUIRED_MODEL_TYPES if use_gguf else DEFAULT_REQUIRED_MODEL_TYPES
    return RuntimeConfig(
        device="cpu",
        models_dir=models_dir,
        model_download_specs=specs,
        required_model_types=required,
        outputs_dir=tmp_path / "outputs",
        ic_lora_dir=models_dir / "ic-loras",
        settings_file=tmp_path / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        force_api_generations=False,
        use_sage_attention=False,
        use_gguf=use_gguf,
        gguf_quantization=gguf_quantization,
        comfyui_models_base=comfyui_models_base,
        camera_motion_prompts={},
        default_negative_prompt="",
    )


def test_specs_cover_all_model_types():
    expected_types = set(get_args(ModelFileType))
    assert set(DEFAULT_MODEL_DOWNLOAD_SPECS.keys()) == expected_types
    assert set(MODEL_FILE_ORDER) == expected_types


def test_model_path_resolves_from_relative_path(tmp_path):
    config = _build_config(tmp_path)
    spec = config.spec_for("text_encoder")
    assert config.model_path("text_encoder") == config.models_dir / spec.relative_path


def test_downloading_path_is_derived_from_specs(tmp_path):
    config = _build_config(tmp_path)

    assert config.downloading_path("checkpoint") == config.downloading_dir
    assert config.downloading_path("zit") == config.downloading_dir / "Z-Image-Turbo"
    assert config.downloading_path("text_encoder") == config.downloading_dir / "gemma-3-12b-it-qat-q4_0-unquantized"


def test_required_model_types_remain_dynamic_for_text_encoder():
    required_with_api = resolve_required_model_types(DEFAULT_REQUIRED_MODEL_TYPES, has_api_key=True)
    required_without_api = resolve_required_model_types(DEFAULT_REQUIRED_MODEL_TYPES, has_api_key=False)

    assert "text_encoder" not in required_with_api
    assert "text_encoder" in required_without_api


def test_required_model_types_empty_base_stays_empty():
    required = resolve_required_model_types(
        frozenset(),
        has_api_key=False,
    )
    assert required == frozenset()


def test_gguf_specs_checkpoint_is_gguf_file(tmp_path):
    config = _build_config(tmp_path, use_gguf=True)
    spec = config.spec_for("checkpoint")
    assert spec.relative_path.suffix == ".gguf"
    assert spec.repo_id == "unsloth/LTX-2.3-GGUF"
    assert config.model_path("checkpoint") == config.models_dir / spec.relative_path


def test_gguf_quantization_affects_checkpoint_filename(tmp_path):
    config_q4 = _build_config(tmp_path, use_gguf=True, gguf_quantization="Q4_K_M")
    config_q8 = _build_config(tmp_path, use_gguf=True, gguf_quantization="Q8_0")
    assert "Q4_K_M" in str(config_q4.model_path("checkpoint"))
    assert "Q8_0" in str(config_q8.model_path("checkpoint"))


def test_comfyui_base_used_for_checkpoint_upsampler_text_encoder(tmp_path):
    comfy_base = tmp_path / "ComfyUI" / "models"
    comfy_base.mkdir(parents=True)
    config = _build_config(tmp_path, use_gguf=True, comfyui_models_base=comfy_base)
    assert config.model_path("checkpoint") == comfy_base / "unet" / "ltx-2.3-22b-dev-Q4_K_M.gguf"
    assert config.model_path("upsampler") == comfy_base / "latent_upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    assert config.model_path("text_encoder") == comfy_base / "text_encoders"
    # zit stays under app models_dir
    assert config.models_dir in config.model_path("zit").parents


def test_comfyui_search_finds_models_in_subdirs(tmp_path):
    """When canonical ComfyUI paths don't exist, model_path() finds files/dirs by name under base."""
    comfy_base = tmp_path / "ComfyUI" / "models"
    comfy_base.mkdir(parents=True)
    # Place checkpoint in custom subdir (canonical unet/ not used)
    (comfy_base / "checkpoints").mkdir()
    ckpt = comfy_base / "checkpoints" / "ltx-2.3-22b-dev-Q4_K_M.gguf"
    ckpt.write_bytes(b"x")
    # Place upsampler in canonical subdir so it's found by canonical path
    (comfy_base / "latent_upscale_models").mkdir()
    up = comfy_base / "latent_upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    up.write_bytes(b"y")
    # Place text_encoder (folder) in a custom subdir so canonical text_encoders/ is missing and we search
    # GGUF spec expects a dir named "text_encoders" (same as reference ComfyUI workflow)
    (comfy_base / "my_models").mkdir()
    te_dir = comfy_base / "my_models" / "text_encoders"
    te_dir.mkdir()
    (te_dir / "placeholder").write_text("")

    config = _build_config(tmp_path, use_gguf=True, comfyui_models_base=comfy_base)
    assert config.model_path("checkpoint") == ckpt
    assert config.model_path("upsampler") == up
    assert config.model_path("text_encoder") == te_dir
