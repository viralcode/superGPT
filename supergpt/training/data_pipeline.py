"""
Production Data Pipeline for superGPT
========================================
Tokenize large datasets and write fixed-size shard files to disk.
Uses multiprocessing for fast tokenization and pre-allocated numpy
buffers for constant memory usage — never accumulates tokens in RAM.

Based on Karpathy's llm.c fineweb.py pattern (the proven approach).

Two modes:
  1. Standalone prep: tokenize dataset → shard files on disk
  2. Streaming: iterate through HF dataset on-the-fly during training

Usage (standalone — prepare data first, train many times):
    python -m supergpt.training.data_pipeline \
        --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer Qwen/Qwen2.5-0.5B \
        --output-dir data \
        --max-tokens 100000000

Usage (in training script — stream on-the-fly):
    from supergpt.training.data_pipeline import create_streaming_dataloader
    loader = create_streaming_dataloader("HuggingFaceFW/fineweb-edu", ...)
"""

import os
import sys
import math
import time
import hashlib
import argparse
import pickle
import glob
from typing import Optional, List, Iterator, Callable
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


# =============================================================================
#  Standalone Data Preparation (Karpathy's shard pattern)
# =============================================================================

def tokenize_doc(doc):
    """Tokenize a single document. Used by multiprocessing Pool.

    Returns a numpy array of uint32 tokens.
    Global tokenizer is initialized per-worker via pool initializer.
    """
    text = doc.get("text", "")
    if not text or len(text.strip()) < 50:
        return np.array([], dtype=np.uint32)

    tokens = _worker_tokenizer.encode(text, add_special_tokens=False)
    # Filter token 0 (padding artifact in some tokenizers)
    tokens = [t for t in tokens if t != 0]
    if not tokens:
        return np.array([], dtype=np.uint32)

    tokens_np = np.array(tokens, dtype=np.uint32)
    return tokens_np


def _init_worker(tokenizer_name):
    """Initialize tokenizer in each worker process (called once per worker)."""
    global _worker_tokenizer
    from transformers import AutoTokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )


def prepare_data(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: Optional[str] = "sample-10BT",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "data",
    max_tokens: int = 100_000_000,
    shard_size: int = 10_000_000,  # 10M tokens per shard (~40MB on disk)
    text_field: str = "text",
    num_workers: int = 0,  # 0 = auto
):
    """Prepare tokenized shard files from a HuggingFace dataset.

    Memory-safe: uses a pre-allocated numpy buffer of shard_size tokens.
    When the buffer fills, write it to disk and reset. Peak RAM usage
    is ~shard_size * 4 bytes (40MB for 10M tokens) regardless of dataset size.

    Output structure:
        data/
          val_000000.bin    (first shard = validation)
          train_000001.bin  (rest = training)
          train_000002.bin
          ...
          meta.pkl          (tokenizer info, token counts)
    """
    import multiprocessing as mp
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Get vocab size
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    del tokenizer  # Free memory before spawning workers

    print(f"{'='*60}")
    print(f"  superGPT Data Pipeline")
    print(f"  Dataset: {dataset_name}")
    print(f"  Tokenizer: {tokenizer_name} (vocab={vocab_size:,})")
    print(f"  Target: {max_tokens:,} tokens")
    print(f"  Shard size: {shard_size:,} tokens ({shard_size * 4 / 1e6:.0f}MB each)")
    print(f"{'='*60}")

    # Always stream — never download full dataset to disk.
    # sample-10BT is 30GB+ and would fill a 50GB RunPod disk.
    # Streaming uses zero disk for source data.
    print(f"\nLoading dataset...")
    kwargs = {"split": "train", "streaming": True}
    print("  Mode: streaming (zero disk for source data)")

    if subset:
        ds = load_dataset(dataset_name, subset, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)

    # Setup tokenizer (single process — streaming datasets can't use mp.Pool)
    from transformers import AutoTokenizer as _AT
    _tokenizer = _AT.from_pretrained(tokenizer_name, trust_remote_code=True)
    print(f"  Tokenizer loaded")

    # Pre-allocate shard buffer (constant memory!)
    shard_buffer = np.empty((shard_size,), dtype=np.uint32)
    token_count = 0
    shard_index = 0
    total_tokens = 0
    n_docs = 0
    t0 = time.time()

    print(f"\nTokenizing...")

    for sample in ds:
        text = sample.get(text_field, "")
        if not text or len(text.strip()) < 50:
            continue

        # Tokenize
        ids = _tokenizer.encode(text, add_special_tokens=False)
        ids = [t for t in ids if t != 0]
        if not ids:
            continue

        tokens = np.array(ids, dtype=np.uint32)
        n_docs += 1

        # Will these tokens fit in the current shard?
        if token_count + len(tokens) < shard_size:
            shard_buffer[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
        else:
            # Fill current shard, write to disk, start new one
            remainder = shard_size - token_count
            shard_buffer[token_count:token_count + remainder] = tokens[:remainder]

            # First shard = validation, rest = training
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"{split}_{shard_index:06d}.bin")
            shard_buffer.tofile(filename)

            elapsed = time.time() - t0
            total_tokens += shard_size
            tps = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  Shard {shard_index:>3d} | {split:>5s} | "
                  f"{total_tokens:>12,} tokens | {tps:,.0f} tok/s | "
                  f"{n_docs:,} docs")

            shard_index += 1

            # Start new shard with leftover tokens
            leftover = len(tokens) - remainder
            shard_buffer[:leftover] = tokens[remainder:]
            token_count = leftover

        # Check token limit
        if total_tokens + token_count >= max_tokens:
            print(f"  Reached target: {max_tokens:,} tokens")
            break

    # Write final partial shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(output_dir, f"{split}_{shard_index:06d}.bin")
        shard_buffer[:token_count].tofile(filename)
        total_tokens += token_count
        print(f"  Shard {shard_index:>3d} | {split:>5s} | "
              f"{total_tokens:>12,} tokens (final)")

    # Also create single train.bin and val.bin for backwards compat
    _merge_shards(output_dir)

    # Save metadata
    meta = {
        "tokenizer_type": "huggingface",
        "tokenizer_name": tokenizer_name,
        "vocab_size": vocab_size,
        "dataset": dataset_name,
        "subset": subset,
        "total_tokens": total_tokens,
        "shard_size": shard_size,
        "num_shards": shard_index + 1,
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done! {total_tokens:,} tokens in {shard_index + 1} shards")
    print(f"  Time: {elapsed:.0f}s ({total_tokens/elapsed:,.0f} tok/s)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")


def _merge_shards(output_dir: str):
    """Merge shard files into single train.bin / val.bin for compatibility.

    Uses chunked reads/writes to avoid RAM spike.
    """
    CHUNK = 5_000_000  # Read/write 5M tokens at a time (~20MB)

    for split in ["train", "val"]:
        shard_files = sorted(glob.glob(os.path.join(output_dir, f"{split}_*.bin")))
        if not shard_files:
            continue

        out_path = os.path.join(output_dir, f"{split}.bin")
        with open(out_path, "wb") as out_f:
            for shard_file in shard_files:
                data = np.fromfile(shard_file, dtype=np.uint32)
                # Write in chunks
                for i in range(0, len(data), CHUNK):
                    data[i:i + CHUNK].tofile(out_f)
                del data

        total_size = os.path.getsize(out_path)
        print(f"  Merged {split}.bin: {total_size / 1e6:.1f}MB "
              f"({total_size // 4:,} tokens)")


# =============================================================================
#  Streaming Dataset (for on-the-fly training without disk prep)
# =============================================================================

class StreamingDataset(IterableDataset):
    """Stream tokenized sequences from HuggingFace datasets.

    For when you want to train directly from HF without disk prep.
    Uses constant memory — no accumulation.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        block_size: int = 256,
        text_field: str = "text",
        split: str = "train",
        max_tokens: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.text_field = text_field
        self.split = split
        self.max_tokens = max_tokens
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        """Yield (input_ids, labels) pairs of shape (block_size,)."""
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=True
        )

        kwargs = {"streaming": True, "split": self.split}
        if self.subset:
            ds = load_dataset(self.dataset_name, self.subset, **kwargs)
        else:
            ds = load_dataset(self.dataset_name, **kwargs)

        # Distributed sharding
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)

        # Multi-worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds = ds.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        # Fixed-size token buffer for sequence packing (constant memory)
        token_buffer = np.empty(self.block_size * 10, dtype=np.int64)
        buf_len = 0
        total_tokens = 0

        for sample in ds:
            text = sample.get(self.text_field, "")
            if not text or len(text.strip()) < 50:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            ids = [t for t in ids if t != 0]
            if not ids:
                continue

            # Add to buffer (resize if needed)
            needed = buf_len + len(ids)
            if needed > len(token_buffer):
                new_buf = np.empty(max(needed * 2, len(token_buffer) * 2), dtype=np.int64)
                new_buf[:buf_len] = token_buffer[:buf_len]
                token_buffer = new_buf

            token_buffer[buf_len:buf_len + len(ids)] = ids
            buf_len += len(ids)
            total_tokens += len(ids)

            # Yield packed sequences
            while buf_len >= self.block_size + 1:
                chunk = token_buffer[:self.block_size + 1].copy()
                # Shift buffer
                remaining = buf_len - self.block_size
                token_buffer[:remaining] = token_buffer[self.block_size:buf_len]
                buf_len = remaining

                yield {
                    "input_ids": torch.from_numpy(chunk[:-1]),
                    "labels": torch.from_numpy(chunk[1:]),
                }

            if self.max_tokens and total_tokens >= self.max_tokens:
                break


# =============================================================================
#  Dataloader Factory
# =============================================================================

def create_streaming_dataloader(
    dataset_name: str,
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    subset: Optional[str] = None,
    block_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
    text_field: str = "text",
    split: str = "train",
    max_tokens: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """Create a streaming dataloader for on-the-fly training."""
    dataset = StreamingDataset(
        dataset_name=dataset_name,
        subset=subset,
        tokenizer_name=tokenizer_name,
        block_size=block_size,
        text_field=text_field,
        split=split,
        max_tokens=max_tokens,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize dataset into shard files (Karpathy-style)"
    )
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="Dataset subset/config")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace tokenizer")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for shard files")
    parser.add_argument("--max-tokens", type=int, default=100_000_000,
                        help="Maximum tokens to process")
    parser.add_argument("--shard-size", type=int, default=10_000_000,
                        help="Tokens per shard file (default: 10M = ~40MB)")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Name of text column in dataset")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Tokenization workers (0 = auto)")

    args = parser.parse_args()
    prepare_data(
        dataset_name=args.dataset,
        subset=args.subset,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        shard_size=args.shard_size,
        text_field=args.text_field,
        num_workers=args.num_workers,
    )
