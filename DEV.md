# Development notes

Developer-oriented notes for building and extending LTX Desktop. For user-facing docs see [README.md](README.md) and [docs/](docs/).

## GGUF mode

GGUF mode lets you run quantized LTX models (e.g. from [Unsloth LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF)) for lower VRAM use and better compatibility on Apple Silicon. The implementation aims to match the **reference ComfyUI workflow** (Unsloth README / DualCLIPLoaderGGUF + UnetLoaderGGUF) so the same assets work in both.

### Enabling GGUF mode

- **Runtime:** set `LTX_USE_GGUF=1` (e.g. in the environment or via your launch script).
- **UI:** when the app supports it, a settings toggle or model-type choice may enable GGUF (see `use_gguf` in backend state/settings).

### Model layout (GGUF)

When `use_gguf` is true, the app uses **GGUF model specs** instead of the default full-weight specs:

| Model type     | Default (non-GGUF)                    | GGUF (same as reference ComfyUI)                    |
|----------------|---------------------------------------|-----------------------------------------------------|
| Checkpoint     | `ltx-2.3-22b-distilled.safetensors`  | `ltx-2.3-22b-dev-{Q}.gguf` (e.g. Q4_K_M) in `unet/` |
| Text encoder   | `gemma-3-12b-it-qat-q4_0-unquantized/` (~25 GB) | **`text_encoders/`** with Gemma GGUF + connectors   |
| VAE            | (in checkpoint or separate)           | `vae/*.safetensors` next to `unet/`                 |
| Upsampler      | Same in both                          | Same                                                |

**Text encoders directory (GGUF):** must contain exactly what the reference ComfyUI workflow uses:

- **Gemma GGUF:** e.g. `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` (any file matching `gemma-3-12b-it-qat-*.gguf`).
- **Embeddings connectors:** `ltx-2.3-22b-dev_embeddings_connectors.safetensors`.

No full unquantized Gemma folder is used in GGUF mode.

### Backend changes for GGUF

- **Specs and paths**  
  - `backend/runtime_config/model_download_specs.py`: `get_gguf_model_download_specs()` defines GGUF checkpoint, upsampler, and **text_encoder** (as `text_encoders/` folder).  
  - `resolve_gguf_text_encoder_paths(text_encoder_dir)` returns `(gemma_gguf_path, connectors_path)` when both files are present.

- **Checkpoint loading**  
  - GGUF checkpoint is converted to a merged safetensors (transformer from GGUF + VAE from `vae/`) on **CPU** only to avoid filling MPS during conversion.  
  - See `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py` and `backend/services/gguf/` (loader, merge).

- **Text encoding in GGUF mode**  
  - When `use_gguf` and the two text encoder files exist, the app uses **quantized Gemma GGUF + embeddings connectors** instead of the full Gemma:  
    - `backend/services/gguf_text_encoder/encode.py`: `encode_text_gguf()` runs the Gemma GGUF (via **llama-cpp-python**, optional dep) and applies the same feature-extractor + aggregate-embed logic as ComfyUI-LTXVideo.  
  - Patches in `backend/services/text_encoder/ltx_text_encoder.py`:  
    - **ModelLedger.text_encoder:** returns a dummy encoder when GGUF paths are present (no full Gemma load).  
    - **encode_text:** when GGUF paths are present, calls `encode_text_gguf()` and returns its result.

- **Optional dependency**  
  - GGUF text encoding: `pip install llama-cpp-python` or `uv pip install -e ".[gguf-text]"` (see `backend/pyproject.toml` optional `gguf-text` extra).

- **Port and memory**  
  - Default backend port is **8010** to avoid clashing with ComfyUI (8000).  
  - On macOS, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9` is set to cap MPS memory and reduce risk of system freezes.

### Docs

- **Memory and behaviour:** [backend/docs/GGUF_MEMORY.md](backend/docs/GGUF_MEMORY.md) — why ComfyUI avoids OOM, what our in-process path does, MPS cap, and (after GGUF text encoder) use of quantized Gemma + connectors.
- **Integration plan (historical):** [backend/docs/GGUF_INTEGRATION_PLAN.md](backend/docs/GGUF_INTEGRATION_PLAN.md).

### Testing

- Run backend tests: `pnpm backend:test`.  
- GGUF-related tests live under `backend/tests/` (e.g. `test_model_download_specs.py` for GGUF specs and paths, `test_gguf.py` if present).

---

## Other notes

- **Backend architecture:** see [backend/architecture.md](backend/architecture.md).  
- **Agent/CLAUDE guidance:** see [AGENTS.md](AGENTS.md).
