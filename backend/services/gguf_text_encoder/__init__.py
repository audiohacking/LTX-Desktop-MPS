"""GGUF text encoder: quantized Gemma GGUF + embeddings connectors (same as reference ComfyUI workflow)."""

from __future__ import annotations

from services.gguf_text_encoder.encode import encode_text_gguf

__all__ = ["encode_text_gguf"]
