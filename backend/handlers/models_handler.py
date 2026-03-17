"""Model availability and model status handlers."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import CheckPathResponse, ModelFileStatus, ModelInfo, ModelsStatusResponse, TextEncoderStatus
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_required_model_types
from state.app_state_types import AppState, AvailableFiles

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


def resolve_comfyui_base_from_settings(comfyui_models_path: str) -> Path | None:
    """Resolve ComfyUI models base from path string; on macOS default to ~/Documents/ComfyUI/models when empty."""
    raw = (os.environ.get("LTX_COMFYUI_MODELS_PATH") or comfyui_models_path or "").strip()
    if not raw and platform.system() == "Darwin":
        raw = os.path.expanduser("~/Documents/ComfyUI/models")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock)
        self._config = config

    @staticmethod
    def _path_size(path: Path, is_folder: bool) -> int:
        if not is_folder:
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _scan_available_files(self) -> AvailableFiles:
        """Scan for model files. Uses current settings' ComfyUI path (and ENV/Darwin default) so that
        as soon as the user sets or saves the path, we see the models without restart.
        """
        files: AvailableFiles = {}
        # Use current ComfyUI base from state (and ENV), not only the one baked into config at startup.
        # Only use it when in GGUF mode so we look for GGUF/ComfyUI layout; otherwise use config paths.
        comfyui_base: Path | None = resolve_comfyui_base_from_settings(
            self.state.app_settings.comfyui_models_path or ""
        )
        use_comfyui_base = (
            self._config.use_gguf
            and comfyui_base is not None
            and comfyui_base.is_dir()
        )
        base_for_scan: Path | None = comfyui_base if use_comfyui_base else None

        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            if base_for_scan is not None and model_type in ("checkpoint", "upsampler", "text_encoder"):
                path = self._config.model_path_with_base(model_type, base_for_scan)
            else:
                path = self._config.model_path(model_type)
            if spec.is_folder:
                ready = path.exists() and any(path.iterdir()) if path.exists() else False
                if not ready and model_type == "text_encoder" and self._config.use_gguf and base_for_scan is not None:
                    # GGUF: accept text_encoders dir with any content (mmproj/gemma .gguf layout)
                    te_dir: Path = base_for_scan / "text_encoders"
                    if te_dir.is_dir() and any(te_dir.iterdir()):
                        path = te_dir
                        ready = True
                files[model_type] = path if ready else None
            else:
                files[model_type] = path if path.exists() else None
        return files

    @with_state_lock
    def refresh_available_files(self) -> AvailableFiles:
        self.state.available_files = self._scan_available_files()
        return self.state.available_files.copy()

    def get_text_encoder_status(self) -> TextEncoderStatus:
        files = self.refresh_available_files()
        text_encoder_path = files["text_encoder"]
        exists = text_encoder_path is not None
        text_spec = self._config.spec_for("text_encoder")
        size_bytes = self._path_size(text_encoder_path, is_folder=True) if exists else 0
        expected = text_spec.expected_size_bytes

        return TextEncoderStatus(
            downloaded=exists,
            size_bytes=size_bytes if exists else expected,
            size_gb=round((size_bytes if exists else expected) / (1024**3), 1),
            expected_size_gb=round(expected / (1024**3), 1),
        )

    def get_models_list(self) -> list[ModelInfo]:
        pro_steps = self.state.app_settings.pro_model.steps
        pro_upscaler = self.state.app_settings.pro_model.use_upscaler
        return [
            ModelInfo(id="fast", name="Fast (Distilled)", description="8 steps + 2x upscaler"),
            ModelInfo(
                id="pro",
                name="Pro (Full)",
                description=f"{pro_steps} steps" + (" + 2x upscaler" if pro_upscaler else " (native resolution)"),
            ),
        ]

    def get_models_status(self, has_api_key: bool | None = None) -> ModelsStatusResponse:
        files = self.refresh_available_files()
        settings = self.state.app_settings.model_copy(deep=True)

        if has_api_key is None:
            has_api_key = bool(settings.ltx_api_key)

        models: list[ModelFileStatus] = []
        total_size = 0
        downloaded_size = 0
        required_types = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key,
            settings.use_local_text_encoder,
        )

        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            path = files[model_type]
            exists = path is not None
            actual_size = self._path_size(path, is_folder=spec.is_folder) if exists else 0
            required = model_type in required_types
            if required:
                total_size += spec.expected_size_bytes
                if exists:
                    downloaded_size += actual_size

            description = spec.description
            optional_reason: str | None = None
            if model_type == "text_encoder":
                description += " (optional with API key)" if has_api_key else ""
                optional_reason = "Uses LTX API for text encoding" if has_api_key else None

            models.append(
                ModelFileStatus(
                    name=spec.name,
                    description=description,
                    downloaded=exists,
                    size=actual_size if exists else spec.expected_size_bytes,
                    expected_size=spec.expected_size_bytes,
                    required=required,
                    is_folder=spec.is_folder,
                    optional_reason=optional_reason if model_type == "text_encoder" else None,
                )
            )

        all_downloaded = all(model.downloaded for model in models if model.required)

        return ModelsStatusResponse(
            models=models,
            all_downloaded=all_downloaded,
            total_size=total_size,
            downloaded_size=downloaded_size,
            total_size_gb=round(total_size / (1024**3), 1),
            downloaded_size_gb=round(downloaded_size / (1024**3), 1),
            models_path=str(self._config.models_dir),
            has_api_key=has_api_key,
            text_encoder_status=self.get_text_encoder_status(),
            use_local_text_encoder=settings.use_local_text_encoder,
        )

    def check_models_at_path(self, comfyui_models_path: str) -> CheckPathResponse:
        """Scan for required models under the given path (e.g. user-selected folder). Returns all_downloaded.
        Only requires checkpoint, upsampler, and text_encoder that can live under that path (zit stays in app data).
        For GGUF, text_encoder is accepted if we find the gemma folder OR a text_encoders dir with content.
        """
        base = Path(comfyui_models_path).expanduser().resolve()
        if not base.is_dir():
            return CheckPathResponse(all_downloaded=False)
        settings = self.state.app_settings.model_copy(deep=True)
        has_api_key = bool(settings.ltx_api_key)
        required_types = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key,
            settings.use_local_text_encoder,
        )
        # Only require models that can live under the selected path (skip zit; it lives in app models_dir).
        # For GGUF, require only checkpoint (and text_encoder if needed); upsampler optional for this check.
        for model_type in required_types:
            if model_type not in ("checkpoint", "upsampler", "text_encoder"):
                continue
            if model_type == "upsampler" and self._config.use_gguf:
                continue  # optional for GGUF check-path
            path = self._config.model_path_with_base(model_type, base)
            spec = self._config.spec_for(model_type)
            if spec.is_folder:
                if not path.exists() or not any(path.iterdir()):
                    # GGUF: accept text_encoders dir with any content (mmproj/gemma .gguf layout)
                    if model_type == "text_encoder" and self._config.use_gguf:
                        text_encoders_dir = base / "text_encoders"
                        if text_encoders_dir.is_dir() and any(text_encoders_dir.iterdir()):
                            continue
                    return CheckPathResponse(all_downloaded=False)
            else:
                if not path.exists():
                    return CheckPathResponse(all_downloaded=False)
        return CheckPathResponse(all_downloaded=True)
