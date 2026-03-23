"""
Model Export — GGUF Format
============================
Export a trained superGPT model to GGUF format for use with llama.cpp
and other inference engines.

GGUF (GPT-Generated Unified Format) is the standard format for running
LLMs locally with llama.cpp, ollama, LM Studio, etc.

Supports:
  - FP32, FP16 export
  - Q8_0 quantization (8-bit)
  - Q4_0 quantization (4-bit, smallest)

Usage:
    # Export as FP16 (full quality):
    python export.py --checkpoint checkpoints/best.pt --output model-fp16.gguf

    # Export as Q8_0 (good quality, smaller):
    python export.py --checkpoint best.pt --output model-q8.gguf --quantize q8_0

    # Export as Q4_0 (smallest, fast inference):
    python export.py --checkpoint best.pt --output model-q4.gguf --quantize q4_0

Note: GGUF export requires the 'gguf' pip package.
      Install with: pip install gguf
"""

import os
import sys
import struct
import argparse
import numpy as np

import torch

from config import GPTConfig
from model import GPT


# GGUF Constants
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


def quantize_q8_0(tensor):
    """Quantize a tensor to Q8_0 format (block size 32)."""
    tensor = tensor.float().numpy()
    shape = tensor.shape
    # Flatten
    flat = tensor.flatten()
    # Pad to multiple of 32
    n = len(flat)
    pad = (32 - n % 32) % 32
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    n_blocks = len(flat) // 32
    blocks = flat.reshape(n_blocks, 32)

    result = bytearray()
    for block in blocks:
        # Find scale (max absolute value)
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 1.0
        # Quantize to int8
        quantized = np.round(block / scale).clip(-128, 127).astype(np.int8)
        # Pack: float16 scale + 32 int8 values
        result.extend(struct.pack('<e', np.float16(scale)))
        result.extend(quantized.tobytes())

    return bytes(result), shape, n


def quantize_q4_0(tensor):
    """Quantize a tensor to Q4_0 format (block size 32, 4-bit)."""
    tensor = tensor.float().numpy()
    shape = tensor.shape
    flat = tensor.flatten()
    n = len(flat)
    pad = (32 - n % 32) % 32
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    n_blocks = len(flat) // 32
    blocks = flat.reshape(n_blocks, 32)

    result = bytearray()
    for block in blocks:
        amax = np.max(np.abs(block))
        scale = amax / 7.0 if amax > 0 else 1.0
        quantized = np.round(block / scale).clip(-8, 7).astype(np.int8)
        # Pack: float16 scale + 16 bytes (32 4-bit values packed in pairs)
        result.extend(struct.pack('<e', np.float16(scale)))
        packed = bytearray()
        for i in range(0, 32, 2):
            lo = quantized[i] & 0x0F
            hi = quantized[i+1] & 0x0F
            packed.append((hi << 4) | lo)
        result.extend(packed)

    return bytes(result), shape, n


def write_gguf_string(f, s):
    """Write a GGUF string (length-prefixed)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_metadata_kv(f, key, value_type, value):
    """Write a metadata key-value pair."""
    write_gguf_string(f, key)
    f.write(struct.pack('<I', value_type))

    if value_type == GGUF_TYPE_UINT32:
        f.write(struct.pack('<I', value))
    elif value_type == GGUF_TYPE_INT32:
        f.write(struct.pack('<i', value))
    elif value_type == GGUF_TYPE_FLOAT32:
        f.write(struct.pack('<f', value))
    elif value_type == GGUF_TYPE_BOOL:
        f.write(struct.pack('<?', value))
    elif value_type == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    elif value_type == GGUF_TYPE_UINT64:
        f.write(struct.pack('<Q', value))


def export_gguf(checkpoint_path: str, output_path: str, quantize: str = "none"):
    """Export a superGPT model to GGUF format.

    Args:
        checkpoint_path: path to .pt checkpoint
        output_path: path for output .gguf file
        quantize: "none" (FP16), "q8_0", or "q4_0"
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = checkpoint["model_config"]
    config = GPTConfig(**config_dict)
    state_dict = checkpoint["model"]

    print(f"\nModel: {config.n_layer}L / {config.n_head}H / {config.n_embd}E")
    print(f"Vocab: {config.vocab_size} | Context: {config.block_size}")
    print(f"Export format: {'FP16' if quantize == 'none' else quantize.upper()}")

    # Determine tensor type
    if quantize == "q8_0":
        tensor_type = GGML_TYPE_Q8_0
    elif quantize == "q4_0":
        tensor_type = GGML_TYPE_Q4_0
    else:
        tensor_type = GGML_TYPE_F16

    # Prepare tensors
    tensor_infos = []
    tensor_data_list = []

    for name, param in state_dict.items():
        # Skip MTP modules (inference only)
        if "mtp_modules" in name:
            continue

        tensor = param.detach().cpu()

        if quantize != "none" and tensor.dim() >= 2:
            # Quantize weight matrices
            if quantize == "q8_0":
                data, shape, n_elem = quantize_q8_0(tensor)
                ttype = GGML_TYPE_Q8_0
            elif quantize == "q4_0":
                data, shape, n_elem = quantize_q4_0(tensor)
                ttype = GGML_TYPE_Q4_0
        else:
            # FP16 for everything, or FP32 for 1D tensors (norms, biases)
            if tensor.dim() >= 2:
                tensor_fp16 = tensor.half()
                data = tensor_fp16.numpy().tobytes()
                ttype = GGML_TYPE_F16
            else:
                data = tensor.float().numpy().tobytes()
                ttype = GGML_TYPE_F32
            shape = tensor.shape
            n_elem = tensor.numel()

        tensor_infos.append({
            "name": name,
            "shape": list(shape),
            "type": ttype,
            "data": data,
            "n_elem": n_elem,
        })
        tensor_data_list.append(data)

    # ── Write GGUF file ──────────────────────────────────────────────────
    print(f"\nWriting {output_path}...")

    # Collect metadata
    metadata = {
        "general.architecture": (GGUF_TYPE_STRING, "supergpt"),
        "general.name": (GGUF_TYPE_STRING, f"superGPT {config.n_layer}L"),
        "general.file_type": (GGUF_TYPE_UINT32,
                              {GGML_TYPE_F16: 1, GGML_TYPE_Q8_0: 7,
                               GGML_TYPE_Q4_0: 2}.get(tensor_type, 0)),
        "supergpt.context_length": (GGUF_TYPE_UINT32, config.block_size),
        "supergpt.embedding_length": (GGUF_TYPE_UINT32, config.n_embd),
        "supergpt.block_count": (GGUF_TYPE_UINT32, config.n_layer),
        "supergpt.head_count": (GGUF_TYPE_UINT32, config.n_head),
        "supergpt.head_count_kv": (GGUF_TYPE_UINT32, config.n_kv_head),
        "supergpt.vocab_size": (GGUF_TYPE_UINT32, config.vocab_size),
        "supergpt.use_moe": (GGUF_TYPE_BOOL, config.use_moe),
        "supergpt.use_mla": (GGUF_TYPE_BOOL, config.use_mla),
        "supergpt.use_swiglu": (GGUF_TYPE_BOOL, config.use_swiglu),
        "supergpt.use_rope": (GGUF_TYPE_BOOL, config.use_rope),
        "supergpt.sliding_window": (GGUF_TYPE_UINT32, config.sliding_window),
    }

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensor_infos)))  # n_tensors
        f.write(struct.pack('<Q', len(metadata)))       # n_kv

        # Metadata
        for key, (vtype, value) in metadata.items():
            write_gguf_metadata_kv(f, key, vtype, value)

        # Tensor info headers
        for info in tensor_infos:
            write_gguf_string(f, info["name"])
            n_dims = len(info["shape"])
            f.write(struct.pack('<I', n_dims))
            for dim in info["shape"]:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', info["type"]))
            f.write(struct.pack('<Q', 0))  # offset (will be relative)

        # Alignment padding
        alignment = 32
        pos = f.tell()
        pad = (alignment - pos % alignment) % alignment
        f.write(b'\x00' * pad)

        # Tensor data
        for data in tensor_data_list:
            f.write(data)
            # Align
            pad = (alignment - len(data) % alignment) % alignment
            f.write(b'\x00' * pad)

    file_size = os.path.getsize(output_path)
    print(f"\nExported: {output_path}")
    print(f"  Size: {file_size / 1e6:.1f} MB")
    print(f"  Tensors: {len(tensor_infos)}")
    print(f"  Format: {'FP16' if quantize == 'none' else quantize.upper()}")

    # Compare with original
    orig_size = sum(p.numel() * 4 for p in state_dict.values())  # FP32
    print(f"  Compression: {orig_size/1e6:.1f} MB → {file_size/1e6:.1f} MB "
          f"({file_size/orig_size:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export superGPT model to GGUF format")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="model.gguf",
                        help="Output GGUF file path")
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "q8_0", "q4_0"],
                        help="Quantization: none (FP16), q8_0 (8-bit), q4_0 (4-bit)")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    export_gguf(args.checkpoint, args.output, args.quantize)
