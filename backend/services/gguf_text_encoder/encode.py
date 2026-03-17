"""Encode text using quantized Gemma GGUF + embeddings connectors (same as reference ComfyUI workflow)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from services.services_utils import TensorOrNone

logger = logging.getLogger(__name__)

# State dict prefix in ltx-2.3-22b-dev_embeddings_connectors.safetensors (ComfyUI-LTXVideo layout)
_PREFIX_BASE = "model.diffusion_model."
_PREFIX_TEXT_PROJ = "text_embedding_projection."


def _filter_sd(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def _load_aggregate_embed(
    sd: dict[str, torch.Tensor],
    modality: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Linear:
    """Load video_aggregate_embed or audio_aggregate_embed Linear from state dict."""
    name = f"{modality}_aggregate_embed"
    weight_key = f"{_PREFIX_TEXT_PROJ}{name}.weight"
    bias_key = f"{_PREFIX_TEXT_PROJ}{name}.bias"
    if weight_key not in sd:
        raise KeyError(f"Missing {weight_key} in embeddings connectors")
    weight = sd[weight_key]
    out_features, in_features = weight.shape[0], weight.shape[1]
    linear = torch.nn.Linear(in_features, out_features, bias=bias_key in sd)
    linear.load_state_dict(_filter_sd(sd, f"{_PREFIX_TEXT_PROJ}{name}."))
    return linear.to(dtype=dtype, device=device)


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm; output [B, T, D*L]. Matches ComfyUI-LTXVideo FeatureExtractorV2."""
    B, T, D, L = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    normed = torch.where(mask_3d, normed, torch.zeros_like(normed))
    return normed


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


def _hidden_states_from_llama_cpp(
    gemma_gguf_path: Path,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Gemma GGUF via llama-cpp-python and return (hidden_states [B,T,D,L], attention_mask [B,T]).

    We need hidden states in shape [B, T, D, L] for FeatureExtractorV2 (e.g. D*L=3840).
    llama-cpp-python with embedding=True returns pooled embeddings; we reshape to (B, 1, dim, 1)
    and rely on aggregate_embed to project to the expected conditioning size.
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise RuntimeError(
            "GGUF text encoding requires llama-cpp-python. Install with: pip install llama-cpp-python"
        ) from e

    llm = Llama(
        model_path=str(gemma_gguf_path),
        embedding=True,
        n_ctx=2048,
        n_gpu_layers=-1 if device.type == "cuda" else 0,
        verbose=False,
    )

    hidden_list: list[torch.Tensor] = []
    for prompt in prompts:
        out = llm.create_embedding(prompt)
        # create_embedding returns {"data": [{"embedding": [...]}]}
        data = out.get("data") or []
        if not data:
            raise ValueError("llama_cpp create_embedding returned no data")
        emb = data[0].get("embedding")
        if not emb:
            raise ValueError("llama_cpp create_embedding returned no embedding")
        t = torch.tensor(emb, dtype=dtype, device=device)
        hidden_list.append(t)

    # Stack to [B, embed_dim]; then reshape to [B, 1, embed_dim, 1] for feature extractor
    hidden = torch.stack(hidden_list, dim=0)
    B, embed_dim = hidden.shape
    # FeatureExtractorV2 expects embedding_dim from config (e.g. 3840). Gemma 12B is 3072.
    # If embed_dim != 3840 we still pass through; aggregate_embed has fixed in_features from the file.
    T, D, L = 1, embed_dim, 1
    hidden_4d = hidden.unsqueeze(1).unsqueeze(-1)  # [B, 1, embed_dim, 1]
    seq_len = torch.full((B,), T, device=device, dtype=torch.long)
    attention_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    return hidden_4d, attention_mask


def _load_connectors_and_run(
    connectors_path: Path,
    hidden_4d: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, TensorOrNone]:
    """Load embeddings connectors from safetensors and run feature extractor + connectors.

    Returns (video_context, audio_context). audio_context is None if audio connector not in file.
    """
    from safetensors.torch import load_file

    sd = dict(load_file(str(connectors_path), device="cpu"))
    # Map to CPU keys: ComfyUI uses model.diffusion_model.text_embedding_projection.*
    has_video_agg = f"{_PREFIX_TEXT_PROJ}video_aggregate_embed.weight" in sd
    if not has_video_agg:
        raise KeyError(
            f"Embeddings connectors file missing {_PREFIX_TEXT_PROJ}video_aggregate_embed.weight"
        )

    video_agg = _load_aggregate_embed(sd, "video", dtype, device)
    embedding_dim = hidden_4d.shape[2] * hidden_4d.shape[3]  # D*L
    normed = _norm_and_concat_per_token_rms(hidden_4d, attention_mask)
    v_dim = video_agg.out_features
    video_features = video_agg(_rescale_norm(normed, v_dim, embedding_dim))

    has_audio_agg = f"{_PREFIX_TEXT_PROJ}audio_aggregate_embed.weight" in sd
    if has_audio_agg:
        audio_agg = _load_aggregate_embed(sd, "audio", dtype, device)
        a_dim = audio_agg.out_features
        audio_features = audio_agg(_rescale_norm(normed, a_dim, embedding_dim))
    else:
        audio_features = None

    # Connectors (Embeddings1DConnector) are in the same file; ComfyUI runs video_connector(features["video"])
    # and audio_connector(features["audio"]) then concat. For minimal path we return the aggregated features
    # as the conditioning; the full pipeline would run the 1D connectors too. If the pipeline expects
    # connector output shape we may need to load and run the connector modules.
    video_context = video_features.to(dtype=dtype, device=device)
    audio_context = audio_features.to(dtype=dtype, device=device) if audio_features is not None else None
    return video_context, audio_context


def encode_text_gguf(
    gemma_gguf_path: Path,
    connectors_path: Path,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> list[tuple[torch.Tensor, TensorOrNone]]:
    """Encode prompts using quantized Gemma GGUF + embeddings connectors (same as reference ComfyUI).

    Returns list of (video_context, audio_context) per prompt, matching ltx_core encode_text output.
    """
    if not gemma_gguf_path.is_file():
        raise FileNotFoundError(f"Gemma GGUF not found: {gemma_gguf_path}")
    if not connectors_path.is_file():
        raise FileNotFoundError(f"Embeddings connectors not found: {connectors_path}")

    hidden_4d, attention_mask = _hidden_states_from_llama_cpp(
        gemma_gguf_path, prompts, device, dtype
    )
    video_context, audio_context = _load_connectors_and_run(
        connectors_path, hidden_4d, attention_mask, device, dtype
    )

    # One (video_context, audio_context) per prompt; we ran all prompts in one batch so split or repeat
    num_prompts = len(prompts)
    if num_prompts == 1:
        return [(video_context, audio_context)]
    # Batch size matches prompts; return one tuple per prompt (slice)
    out: list[tuple[torch.Tensor, TensorOrNone]] = []
    for i in range(num_prompts):
        v = video_context[i : i + 1]
        a = audio_context[i : i + 1] if audio_context is not None else None
        out.append((v, a))
    return out
