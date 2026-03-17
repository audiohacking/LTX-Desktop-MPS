"""Text encoder patching and API embedding service."""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from services.http_client.http_client import HTTPClient
from services.services_utils import PromptInput, TensorOrNone, device_supports_fp8, sync_device
from state.app_state_types import CachedTextEncoder, TextEncodingResult

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class LTXTextEncoder:
    """Stateless text encoding operations with idempotent monkey-patching."""

    def __init__(self, device: torch.device, http: HTTPClient, ltx_api_base_url: str) -> None:
        self.device = device
        self.http = http
        self.ltx_api_base_url = ltx_api_base_url
        self._model_ledger_patched = False
        self._encode_text_patched = False

    def install_patches(
        self,
        state_getter: Callable[[], AppState],
        config: RuntimeConfig | None = None,
    ) -> None:
        self._install_model_ledger_patch(state_getter, config)
        self._install_encode_text_patch(state_getter, config)

    def _install_model_ledger_patch(
        self,
        state_getter: Callable[[], AppState],
        config: RuntimeConfig | None = None,
    ) -> None:
        if self._model_ledger_patched:
            return

        try:
            from ltx_pipelines.utils import ModelLedger
            from ltx_pipelines.utils import helpers as ltx_utils

            from runtime_config.model_download_specs import resolve_gguf_text_encoder_paths

            original_text_encoder = ModelLedger.text_encoder
            original_cleanup_memory = ltx_utils.cleanup_memory

            def _quantize_linear_weights_fp8(module: object) -> None:
                """Cast all Linear weights to float8_e4m3fn and patch forward to upcast."""
                for child in module.modules():  # type: ignore[union-attr]
                    if not isinstance(child, torch.nn.Linear):
                        continue
                    if not device_supports_fp8(self.device):
                        continue
                    child.weight.data = child.weight.data.to(torch.float8_e4m3fn)
                    if child.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                        child.bias.data = child.bias.data.to(torch.float8_e4m3fn)

                    def _make_upcast_forward(lin: torch.nn.Linear) -> Callable[..., torch.Tensor]:
                        def _fwd(x: torch.Tensor, **kw: object) -> torch.Tensor:
                            w = lin.weight.to(x.dtype)
                            b = lin.bias.to(x.dtype) if lin.bias is not None else None  # pyright: ignore[reportUnnecessaryComparison]
                            return torch.nn.functional.linear(x, w, b)
                        return _fwd

                    child.forward = _make_upcast_forward(child)  # type: ignore[assignment]

            def patched_text_encoder(self_model_ledger: ModelLedger) -> object:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is None:
                    return original_text_encoder(self_model_ledger)

                if te_state.api_embeddings is not None:
                    return DummyTextEncoder()

                # GGUF mode: use quantized Gemma GGUF + connectors (same as reference ComfyUI)
                if config is not None and getattr(config, "use_gguf", False):
                    from pathlib import Path

                    paths = resolve_gguf_text_encoder_paths(Path(config.model_path("text_encoder")))
                    if paths is not None:
                        return DummyTextEncoder()

                if te_state.cached_encoder is not None:
                    try:
                        te_state.cached_encoder.to(self.device)
                        sync_device(self.device)
                    except Exception:
                        logger.warning("Failed to move cached text encoder to %s", self.device, exc_info=True)
                    return te_state.cached_encoder

                saved_device = self_model_ledger.device
                # We force everything to CPU during initial build/loading to ensure
                # weights are actually loaded (not left on 'meta') and to allow
                # our custom quantization to run on a real device.
                self_model_ledger.device = torch.device("cpu")
                
                # Some versions of ltx-pipelines use _target_device() for the build() call.
                # If it returns 'meta', the subsequent .to(self.device) in the original
                # code will fail for meta tensors.
                original_target_device_method = getattr(self_model_ledger, "_target_device", None)
                if original_target_device_method:
                    setattr(self_model_ledger, "_target_device", lambda: torch.device("cpu"))

                try:
                    te_state.cached_encoder = cast(
                        CachedTextEncoder, original_text_encoder(self_model_ledger)
                    )
                finally:
                    self_model_ledger.device = saved_device
                    if original_target_device_method:
                        setattr(self_model_ledger, "_target_device", original_target_device_method)

                _quantize_linear_weights_fp8(te_state.cached_encoder)

                te_state.cached_encoder.to(self.device)
                sync_device(self.device)
                return te_state.cached_encoder

            def patched_cleanup_memory() -> None:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is not None and te_state.cached_encoder is not None:
                    try:
                        te_state.cached_encoder.to(torch.device("cpu"))
                    except Exception:
                        logger.warning("Failed to move cached text encoder to CPU", exc_info=True)
                original_cleanup_memory()

            setattr(ModelLedger, "text_encoder", patched_text_encoder)

            for module_name in (
                "ltx_pipelines.utils.helpers",
                "ltx_pipelines.distilled",
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
            ):
                try:
                    module = __import__(module_name, fromlist=["cleanup_memory"])
                    if hasattr(module, "cleanup_memory"):
                        setattr(module, "cleanup_memory", patched_cleanup_memory)
                except Exception:
                    logger.warning("Failed to patch cleanup_memory for module %s", module_name, exc_info=True)

            self._model_ledger_patched = True
            logger.info("Installed ModelLedger text encoder patch")
        except Exception as exc:
            logger.warning("Failed to patch ModelLedger: %s", exc, exc_info=True)

    def _install_encode_text_patch(
        self,
        state_getter: Callable[[], AppState],
        config: RuntimeConfig | None = None,
    ) -> None:
        if self._encode_text_patched:
            return

        try:
            from pathlib import Path

            from ltx_core.text_encoders import gemma as text_enc_module
            from ltx_pipelines import distilled as distilled_module

            from runtime_config.model_download_specs import resolve_gguf_text_encoder_paths
            from services.gguf_text_encoder.encode import encode_text_gguf

            original_encode_text = text_enc_module.encode_text

            def patched_encode_text(
                text_encoder: object,
                prompts: PromptInput,
                *args: object,
                **kwargs: object,
            ) -> list[tuple[torch.Tensor, TensorOrNone]]:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is not None and te_state.api_embeddings is not None:
                    video_context = te_state.api_embeddings.video_context
                    audio_context = te_state.api_embeddings.audio_context
                    num_prompts = len(prompts) if not isinstance(prompts, str) else 1
                    out: list[tuple[torch.Tensor, TensorOrNone]] = []
                    for i in range(num_prompts):
                        if i == 0:
                            out.append((video_context, audio_context))
                        else:
                            zero_video = torch.zeros_like(video_context)
                            zero_audio = torch.zeros_like(audio_context) if audio_context is not None else None
                            out.append((zero_video, zero_audio))
                    return out

                prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
                # GGUF mode: quantized Gemma GGUF + embeddings connectors (same as reference ComfyUI)
                if config is not None and getattr(config, "use_gguf", False):
                    paths = resolve_gguf_text_encoder_paths(Path(config.model_path("text_encoder")))
                    if paths is not None:
                        gemma_path, connectors_path = paths
                        return encode_text_gguf(
                            gemma_path,
                            connectors_path,
                            prompt_list,
                            self.device,
                            dtype=torch.bfloat16,
                        )

                return cast(
                    list[tuple[torch.Tensor, TensorOrNone]],
                    original_encode_text(cast(Any, text_encoder), prompt_list, *args, **kwargs),
                )

            setattr(text_enc_module, "encode_text", patched_encode_text)
            setattr(distilled_module, "encode_text", patched_encode_text)

            for module_name in (
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
            ):
                try:
                    module = __import__(module_name, fromlist=["encode_text"])
                    setattr(module, "encode_text", patched_encode_text)
                except Exception:
                    logger.warning("Failed to patch encode_text for module %s", module_name, exc_info=True)

            self._encode_text_patched = True
            logger.info("Installed encode_text API embeddings patch")
        except Exception as exc:
            logger.warning("Failed to patch encode_text: %s", exc, exc_info=True)

    def get_model_id_from_checkpoint(self, checkpoint_path: str) -> str | None:
        if checkpoint_path.endswith(".gguf"):
            return None  # GGUF files have no safetensors metadata; use local encoder or fixed id
        try:
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata and "encrypted_wandb_properties" in metadata:
                    return metadata["encrypted_wandb_properties"]
        except Exception as exc:
            logger.warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
        return None

    def encode_via_api(self, prompt: str, api_key: str, checkpoint_path: str, enhance_prompt: bool) -> TextEncodingResult | None:
        model_id = self.get_model_id_from_checkpoint(checkpoint_path)
        if not model_id:
            return None

        try:
            start = time.time()
            response = self.http.post(
                f"{self.ltx_api_base_url}/v1/prompt-embedding",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json_payload={
                    "prompt": prompt,
                    "model_id": model_id,
                    "enhance_prompt": enhance_prompt,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.warning("LTX API error %s: %s", response.status_code, response.text)
                return None

            conditioning = pickle.loads(response.content)  # noqa: S301
            if not conditioning or len(conditioning) == 0:
                logger.warning("LTX API returned unexpected conditioning format")
                return None

            embeddings = conditioning[0][0]
            video_dim = 4096
            if embeddings.shape[-1] > video_dim:
                video_context = embeddings[..., :video_dim].contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = embeddings[..., video_dim:].contiguous().to(dtype=torch.bfloat16, device=self.device)
            else:
                video_context = embeddings.contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = None

            logger.info("Text encoded via API in %.1fs", time.time() - start)
            return TextEncodingResult(video_context=video_context, audio_context=audio_context)

        except Exception as exc:
            logger.warning("LTX API encoding failed: %s", exc, exc_info=True)
            return None


class DummyTextEncoder:
    pass
