"""Load LTX transformer state dict from a GGUF file (in-process, no ComfyUI).

Supports F32, F16, BF16 and common quantized types (Q4_K_M, Q8_0, etc.) by
dequantizing at load time so the existing safetensors-based pipeline can run.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _tensor_from_gguf_reader(reader: object, tensor: object, dtype: torch.dtype) -> torch.Tensor:
    """Read one tensor from a gguf reader and return a torch tensor (dequantized if needed)."""
    import gguf

    name = getattr(tensor, "name", None)
    tensor_type = getattr(tensor, "tensor_type", None)
    shape = getattr(tensor, "shape", None)
    data = getattr(tensor, "data", None)
    if data is None and hasattr(reader, "read_tensor"):
        data = reader.read_tensor(tensor)
    if shape is None and hasattr(tensor, "n_elements"):
        shape = (tensor.n_elements,)
    if shape is not None and not isinstance(shape, tuple):
        shape = tuple(shape) if hasattr(shape, "__iter__") else (int(shape),)

    if data is None:
        raise ValueError(f"Could not read tensor data for {name}")

    if hasattr(data, "__array__"):
        # Copy so the array is writable; PyTorch rejects non-writable numpy buffers.
        data = torch.from_numpy(np.asarray(data).copy())
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    q = getattr(gguf, "GGMLQuantizationType", None)
    if q is not None and tensor_type is not None:
        if tensor_type in (getattr(q, "F32", None),):
            return data.view(torch.float32).to(dtype)
        if tensor_type in (getattr(q, "F16", None),):
            return data.view(torch.float16).to(dtype)
        if tensor_type in (getattr(q, "BF16", None),):
            return data.view(torch.bfloat16).to(dtype)
        block_size, type_size = getattr(gguf, "GGML_QUANT_SIZES", {}).get(tensor_type, (0, 0))
        if block_size and type_size:
            dequant = _dequantize_blocks(data, tensor_type, block_size, type_size, shape, dtype)
            if dequant is not None:
                return dequant

    if data.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return data.to(dtype)
    raise ValueError(f"Unsupported GGUF tensor type for {name}: {tensor_type}")


def _dequantize_blocks(
    data: torch.Tensor,
    tensor_type: int,
    block_size: int,
    type_size: int,
    shape: tuple[int, ...] | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Dequantize quantized blocks; returns None if type not supported."""
    try:
        from diffusers.quantizers.gguf.utils import dequantize_functions
        import gguf
    except ImportError:
        return None
    try:
        qt = gguf.GGMLQuantizationType(tensor_type)
    except (ValueError, TypeError):
        return None
    fn = dequantize_functions.get(qt)
    if fn is None:
        return None
    n_blocks = data.numel() // type_size
    blocks = data.view(torch.uint8).reshape((n_blocks, type_size))
    out = fn(blocks, block_size, type_size, dtype=dtype)
    if out is None:
        return None
    n_elems = n_blocks * block_size
    flat = out.reshape(n_elems)
    if shape is not None and len(shape) > 0:
        try:
            return flat.reshape(tuple(int(s) for s in shape)).to(dtype)
        except (ValueError, RuntimeError):
            pass
    return flat.to(dtype)


def load_transformer_state_dict_from_gguf(
    gguf_path: str | Path,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load the LTX transformer state dict from a .gguf file.

    Tensor names are returned as in the file (typically ComfyUI naming).
    The pipeline's ModelLedger applies LTXV_MODEL_COMFY_RENAMING_MAP when loading.
    """
    import gguf

    path = Path(gguf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    reader = gguf.GGUFReader(str(path))
    state_dict: dict[str, torch.Tensor] = {}

    for tensor in reader.tensors:
        name = getattr(tensor, "name", None)
        if name is None:
            continue
        try:
            t = _tensor_from_gguf_reader(reader, tensor, dtype)
            if t is not None:
                state_dict[name] = t.to(device)
        except Exception as e:
            logger.warning("Skip tensor %s: %s", name, e)
            continue

    return state_dict


def create_merged_checkpoint_from_gguf(
    gguf_path: str | Path,
    vae_base_path: str | Path | None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Create a temporary safetensors file with transformer (from GGUF) + VAE/audio (from vae_base_path).

    Returns the path to the temp file. Caller should unlink when done.
    """
    import safetensors.torch

    path = Path(gguf_path).resolve()
    state = dict(load_transformer_state_dict_from_gguf(path, device=device, dtype=dtype))

    if vae_base_path:
        vae_dir = Path(vae_base_path).expanduser().resolve()
        if vae_dir.is_dir():
            for sf in sorted(vae_dir.glob("*.safetensors")):
                try:
                    loaded = safetensors.torch.load_file(str(sf), device="cpu")
                    for k, v in loaded.items():
                        state[k] = v.to(dtype) if v.is_floating_point() else v
                except Exception as e:
                    logger.warning("Skip VAE file %s: %s", sf, e)

    fd, out_path = tempfile.mkstemp(suffix=".safetensors", prefix="ltx_gguf_")
    import os
    os.close(fd)
    safetensors.torch.save_file(state, out_path)
    return out_path
