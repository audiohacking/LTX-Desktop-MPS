# LTX Desktop GGUF Integration Plan

## Current Architecture Summary

### Model wiring

- **Config**: `RuntimeConfig` in `runtime_config/runtime_config.py` holds `model_download_specs` (per `ModelFileType`) and derives paths via `model_path(model_type)` = `models_dir / spec.relative_path`.
- **Specs**: `runtime_config/model_download_specs.py` defines `DEFAULT_MODEL_DOWNLOAD_SPECS` and `DEFAULT_REQUIRED_MODEL_TYPES`:
  - **checkpoint**: `ltx-2.3-22b-distilled.safetensors` (Lightricks/LTX-2.3), ~43 GB
  - **upsampler**: `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (Lightricks/LTX-2.3)
  - **text_encoder**: folder `gemma-3-12b-it-qat-q4_0-unquantized` (Lightricks)
  - **zit**: folder `Z-Image-Turbo` (Tongyi-MAI)
- **Downloads**: `DownloadHandler` uses `config.spec_for(model_type)` for `repo_id`, `relative_path`, `expected_size_bytes`; downloads via `ModelDownloader` (HuggingFace); moves from `.downloading/` to final path.
- **Pipelines**: All LTX video pipelines take a **checkpoint path (safetensors)**:
  - `PipelinesHandler._create_video_pipeline()` → `config.model_path("checkpoint")` + upsampler + gemma root → `LTXFastVideoPipeline.create(...)` → `DistilledPipeline(distilled_checkpoint_path=..., ...)` (from `ltx_pipelines.distilled`).
  - Same pattern for IC-LoRA, A2V, Retake: they all use `model_path("checkpoint")` and pass it to ltx-pipelines/ltx-core.
- **Inference**: `ltx_pipelines` and `ltx_core` load the main transformer from **safetensors** only (PyTorch + `safetensors`). There is no GGUF loading in the official stack.

### Where paths are used

| Consumer | What it uses |
|----------|----------------|
| `PipelinesHandler` | `model_path("checkpoint")`, `model_path("upsampler")`, `model_path("zit")` |
| `TextHandler` | `model_path("text_encoder")`, `model_path("checkpoint")` (for API model_id from checkpoint) |
| `ModelsHandler` | `spec_for()` and `model_path()` for status and listing |
| `DownloadHandler` | `spec_for()` for repo_id, name, relative_path, expected_size; `model_path()` for final destination |
| `HealthHandler` | `model_path("zit")` for ZIT presence |

### Environment already in use

- `LTX_APP_DATA_DIR`: app data root (required).
- `USE_SAGE_ATTENTION`: "1" to enable SageAttention (disabled on MPS).
- `BACKEND_DEBUG`: "1" for debugpy.
- `LTX_PORT`: server port.

---

## Unsloth LTX-2.3-GGUF (target)

- **Repo**: https://huggingface.co/unsloth/LTX-2.3-GGUF
- **Checkpoint**: Single GGUF file for the transformer (e.g. `ltx-2.3-22b-dev-Q4_K_M.gguf` or UD variants). Multiple quantizations (Q2_K–Q8_0, BF16).
- **Auxiliary** (from same page): VAE (video/audio) and text encoder related files are **safetensors** or separate GGUF (e.g. Gemma mmproj). Example from their instructions:
  - `unet/`: main GGUF
  - `vae/`: `ltx-2.3-22b-dev_video_vae.safetensors`, `ltx-2.3-22b-dev_audio_vae.safetensors`
  - `text_encoders/`: `ltx-2.3-22b-dev_embeddings_connectors.safetensors`, Gemma GGUF + mmproj
- **Inference**: Documented for **ComfyUI + ComfyUI-GGUF** (city96). The official ltx-pipelines Python package does **not** load GGUF; inference with GGUF today is via ComfyUI on Apple MPS.

---

## Goal

- Add an **ENV switch** (e.g. `LTX_USE_GGUF=1`) so that:
  1. **Download specs** point to Unsloth LTX-2.3-GGUF assets (checkpoint = GGUF; VAE/connectors/etc. as on the page).
  2. **Path resolution** returns the GGUF file for the “checkpoint” slot when GGUF mode is on.
  3. **Standard safetensors path remains** when the ENV is not set; all existing tests and behavior stay valid.
- Focus: macOS / MPS; no removal of current model path or pipeline behavior for the default (safetensors) path.

---

## Inference constraint

- Current app uses `DistilledPipeline` (and related pipelines) with `distilled_checkpoint_path=<safetensors>`. Those loaders expect **safetensors**; passing a `.gguf` path would fail at load time.
- **Options for “use” GGUF:**
  1. **Later**: Integrate a GGUF-capable backend (e.g. Python bindings to the same stack ComfyUI-GGUF uses, or ComfyUI API when in GGUF mode). Not in scope for the initial patch.
  2. **Initial patch**: Only **download + path wiring**. When `LTX_USE_GGUF=1`, the app downloads the GGUF set and exposes the GGUF checkpoint path via `model_path("checkpoint")`. Pipeline creation can either:
     - **A)** Remain unchanged: still call `DistilledPipeline` with that path → will fail at load with a clear error (safetensors open on .gguf). We document that “GGUF mode prepares assets; use ComfyUI for inference until a GGUF backend is integrated,” or
     - **B)** When GGUF mode is on, use a **stub pipeline** that raises a clear “GGUF inference not yet implemented” and avoid calling ltx_pipelines with a .gguf path.

Recommendation: **Option B** for the first PR — ENV + GGUF specs + path wiring + conditional pipeline: when `LTX_USE_GGUF=1`, create a stub (or a dedicated “GGUF pipeline” class that raises a helpful error) so the app does not attempt to open GGUF as safetensors. This keeps tests passable and standard path untouched.

---

## Implementation plan

### 1. ENV and runtime flag

- In `ltx2_server.py` (or a small `runtime_config` module used at bootstrap):
  - Read `LTX_USE_GGUF` (e.g. `os.environ.get("LTX_USE_GGUF", "0") == "1"`).
  - Pass a boolean `use_gguf: bool` into `RuntimeConfig` (or into whatever builds the effective model specs).

### 2. Model specs: two sets

- **Standard**: Keep `DEFAULT_MODEL_DOWNLOAD_SPECS` and `DEFAULT_REQUIRED_MODEL_TYPES` as today (safetensors checkpoint, same repos).
- **GGUF**: Add `GGUF_MODEL_DOWNLOAD_SPECS` (and optionally `GGUF_REQUIRED_MODEL_TYPES`) in `model_download_specs.py`:
  - **checkpoint**: Single file, e.g. `ltx-2.3-22b-dev-Q4_K_M.gguf` (or one chosen default), repo `unsloth/LTX-2.3-GGUF`, `expected_size_bytes` set to a realistic value for that quantization.
  - **upsampler**: Can stay Lightricks/LTX-2.3 (`ltx-2.3-spatial-upscaler-x2-1.0.safetensors`) unless Unsloth provides a different one.
  - **text_encoder**: Per Unsloth page, they use Gemma GGUF + mmproj + embedding connectors; we can either keep current Gemma folder spec or add a GGUF-specific set (e.g. download from unsloth/LTX-2.3-GGUF text_encoders/ and optionally unsloth/gemma-3-12b-it-qat-GGUF). For minimal first step, we can keep the same text_encoder spec and only switch checkpoint (and optionally VAE paths if we add a “gguf vae” slot later).
  - **zit**: Unchanged (same folder).
- **Runtime**: `RuntimeConfig` gets `use_gguf: bool`. When building the app, the effective `model_download_specs` is either `DEFAULT_MODEL_DOWNLOAD_SPECS` or `GGUF_MODEL_DOWNLOAD_SPECS` (and same for required types if they differ). All existing code that uses `config.spec_for()` and `config.model_path()` then automatically uses GGUF paths when the flag is true.

### 3. Pipeline creation (conditional)

- In `PipelinesHandler` (and any other place that constructs `LTXFastVideoPipeline` / IC-LoRA / A2V / Retake):
  - If `config.use_gguf` (or equivalent) is true, **do not** call the existing pipeline classes with the GGUF checkpoint path. Instead:
    - Either instantiate a **stub pipeline** that implements the same protocol but raises a clear error on `generate`/`warmup` (“GGUF inference not yet implemented; use ComfyUI or wait for backend”), or
    - Short-circuit `load_gpu_pipeline` for the fast model and return a state that shows “GGUF mode – inference disabled” and avoid loading.
- This prevents `safetensors.safe_open` (or equivalent) from being called on a .gguf file and keeps standard behavior and tests unchanged when `LTX_USE_GGUF` is not set.

### 4. Text encoder / API

- `LTXTextEncoder.get_model_id_from_checkpoint(checkpoint_path)` uses `safetensors.safe_open` on the checkpoint. For a GGUF file this will fail. When in GGUF mode:
  - Either pass a fixed `model_id` for the API (if we still want to use LTX API for embeddings with GGUF assets), or
  - Skip/return None and rely on local text encoder only when checkpoint is GGUF. Document that with GGUF, API encoding may be unavailable unless we add a separate branch that doesn’t read the checkpoint file for model_id.

### 5. Downloads and layout

- Download handler already uses `spec_for()` and `model_path()`; no change needed except that the specs come from the GGUF set when the flag is on. Ensure `downloading_path()` and `_move_to_final` work for a single-file checkpoint (GGUF) the same way as for the current single-file distilled checkpoint.
- Optional: document that with GGUF mode, the app may need to download additional files (VAE, embedding connectors) from the same Unsloth repo if we want to mirror the full ComfyUI layout; that can be a follow-up (e.g. extra model types or a single “gguf_assets” folder).

### 6. Tests

- **conftest**: Build `RuntimeConfig` with `use_gguf=False` so all existing tests keep using the default specs and paths.
- **GGUF tests**: Add a small test that with `use_gguf=True` and GGUF specs, `config.model_path("checkpoint")` ends with `.gguf` and `spec_for("checkpoint").repo_id` is the Unsloth repo. Optionally test that pipeline creation with GGUF does not call ltx_pipelines with the GGUF path (e.g. stub returns or raises the expected message).
- **No mocks**: Keep using fakes/service bundle; for GGUF path, create fake model files with a `.gguf` name when testing status/download flow if needed.

### 7. Documentation

- In `AGENTS.md` or a short “GGUF” section: describe `LTX_USE_GGUF=1`, that it switches to Unsloth GGUF download and paths, and that in-app inference for GGUF is not yet implemented (use ComfyUI with the downloaded models or wait for a backend). List the Unsloth LTX-2.3-GGUF page and ComfyUI-GGUF.

---

## File change summary

| Area | File(s) | Change |
|------|---------|--------|
| Runtime flag | `ltx2_server.py` | Read `LTX_USE_GGUF`, pass into config. |
| Config | `runtime_config/runtime_config.py` | Add `use_gguf: bool` to `RuntimeConfig`. |
| Specs | `runtime_config/model_download_specs.py` | Add `GGUF_MODEL_DOWNLOAD_SPECS` (and optional `GGUF_REQUIRED_MODEL_TYPES`); function or property to return active specs from `use_gguf`. |
| Wiring | `ltx2_server.py` (or app_factory) | Build `model_download_specs` from GGUF or default based on `use_gguf`. |
| Pipelines | `handlers/pipelines_handler.py` | When `use_gguf`, use stub pipeline or avoid loading real DistilledPipeline with checkpoint path. |
| Text encoder | `services/text_encoder/ltx_text_encoder.py` | When checkpoint is GGUF (or when `use_gguf`), skip or fix `get_model_id_from_checkpoint` (e.g. return None or fixed id). |
| Tests | `tests/conftest.py`, `tests/test_model_download_specs.py`, new `tests/test_gguf_config.py` | Use `use_gguf=False` in fixtures; add GGUF path/spec tests. |
| Docs | `AGENTS.md` or `backend/docs/GGUF_INTEGRATION_PLAN.md` | Document ENV and current limitation. |

---

## Out of scope for first PR

- Implementing actual GGUF inference (ComfyUI API, or a native GGUF runner).
- Changing frontend (beyond any minimal “GGUF mode” indicator if desired later).
- Adding multiple GGUF quantization choices in the UI; ENV is enough for the experimental fork.

---

## Order of work

1. Add `use_gguf` to `RuntimeConfig` and read `LTX_USE_GGUF` in server bootstrap.
2. Add `GGUF_MODEL_DOWNLOAD_SPECS` and wire active specs in config/build.
3. Update `PipelinesHandler` (and related) to use stub or skip real pipeline when `use_gguf`.
4. Adjust text encoder so GGUF checkpoint path doesn’t break API encoding (return None or fixed id).
5. Tests and docs.

This keeps the standard path and all tests intact while adding a clear, ENV-driven GGUF path and a safe placeholder for inference until a backend is available.
