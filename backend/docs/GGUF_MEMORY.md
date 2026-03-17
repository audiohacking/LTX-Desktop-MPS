# Why ComfyUI Can Run the Same GGUF Models Without OOM (Reference Only)

We do **not** use ComfyUI at runtime. This doc explains why a ComfyUI workflow using the same LTX GGUF models can run on the same machine without OOM or crashes, so we can align our in-process GGUF path where possible.

## ComfyUI Workflow (Reference)

A typical Unsloth LTX ComfyUI workflow uses:

- **UnetLoaderGGUF**: loads `ltx-2.3-22b-dev-Q4_K_M.gguf` into a MODEL.
- **DualCLIPLoaderGGUF**: loads Gemma as **GGUF** (`gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`) plus embeddings connectors (safetensors).
- **VAELoaderKJ**: loads video and audio VAEs separately.
- **LoraLoaderModelOnly**: applies distilled LoRA on top of the base model.

So ComfyUI keeps UNet, CLIP, and VAE as **separate nodes** and loads them in a way that avoids materializing the full dequantized model on GPU at once.

## Why ComfyUI Avoids OOM

1. **Load to RAM first**  
   ComfyUI-GGUF (and similar loaders) load the GGUF into **system RAM** first, then transfer to VRAM. That avoids a huge single allocation on the GPU during load.

2. **Quantized form in VRAM (or mmap)**  
   The GGUF plugin can keep weights in **quantized form** in memory and dequantize **on the fly** during the forward pass. That keeps VRAM usage close to the size of the GGUF file (~14 GB for Q4_K_M) instead of the full bfloat16 size (~2× that).

3. **No single “full state dict” on GPU**  
   There is no step that builds one giant dequantized state dict and moves it all to the device in one go.

4. **PYTORCH_MPS_HIGH_WATERMARK_RATIO**  
   ComfyUI typically does not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, so it stays under PyTorch’s default MPS limit and does not risk system-wide OOM.

## What Our In-Process GGUF Path Does

1. **Full dequantization**  
   We read the GGUF and **dequantize every tensor** (e.g. Q4_K_M → bfloat16) into a full state dict so the existing safetensors-based `DistilledPipeline` can load it. That state dict is roughly **2× the size** of the quantized file.

2. **CPU-only conversion (aligned with ComfyUI)**  
   We now run the **entire** GGUF → merged-safetensors conversion on **CPU** only. No tensors are moved to MPS during this step, so we avoid filling GPU memory during conversion (same idea as “load to RAM first”).

3. **Pipeline then loads the merged file**  
   After conversion, `DistilledPipeline` loads the merged safetensors file. That load is done by `ltx_pipelines` and may move the whole model to device in one go, so peak VRAM is still the full dequantized model size when the pipeline loads.

4. **MPS memory cap**  
   We set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9` on macOS so MPS does not exceed 90% of the recommended working set, avoiding system freezes (we do **not** use `0.0`).

## Summary

| Aspect              | ComfyUI (reference)        | Our in-process GGUF path              |
|---------------------|----------------------------|----------------------------------------|
| Where weights live  | RAM first; then VRAM or mmap| CPU during conversion; then pipeline  |
| Quantized in VRAM?   | Often yes (on-the-fly)     | No; we dequantize to bf16 for pipeline |
| Peak during “load”  | Lower (RAM / quantized)     | Conversion on CPU; pipeline load on GPU|
| MPS limit           | Default (no 0.0)            | 0.9 to avoid system crash              |

So ComfyUI can run the same models on the same machine because it never materializes the full dequantized model on the GPU at once. We get closer by doing GGUF conversion entirely on CPU; further gains would require the pipeline to load in a more memory-efficient way (e.g. layer-by-layer or keeping weights on CPU and moving only when needed), which would be a change in `ltx_pipelines` or a GGUF-native inference path.

## Text encoder (Gemma) in GGUF mode

**In GGUF mode we use the same setup as the reference ComfyUI workflow:** a **quantized Gemma GGUF** (e.g. `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf`) and **ltx-2.3-22b-dev_embeddings_connectors.safetensors** in a `text_encoders/` directory. We do **not** load the full ~25 GB Gemma folder. The backend resolves paths via `resolve_gguf_text_encoder_paths()` and, when present, runs `encode_text_gguf()` (Gemma GGUF via llama-cpp-python + connectors from safetensors) instead of loading the full Gemma. See `backend/services/gguf_text_encoder/` and [DEV.md](../../DEV.md).
