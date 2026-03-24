"""
Production Streaming Data Pipeline for superGPT
=================================================
Stream large datasets from HuggingFace directly to GPU without loading
everything into RAM. Supports:

  - Streaming tokenization from HF datasets (petabyte-scale)
  - Data quality filtering (length, repetition, language)
  - MinHash approximate deduplication
  - Distributed-aware sharding across FSDP ranks
  - Multi-worker prefetching for maximum GPU utilization

Usage:
    # In train.py — replace load_data with streaming pipeline:
    from supergpt.training.data_pipeline import create_streaming_dataloader

    dataloader = create_streaming_dataloader(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer_name="Qwen/Qwen2.5-0.5B",
        block_size=256,
        batch_size=16,
        num_workers=4,
    )

    for batch in dataloader:
        x, y = batch["input_ids"], batch["labels"]
        ...

    # Standalone data preparation with quality filtering:
    python -m supergpt.training.data_pipeline \\
        --dataset HuggingFaceFW/fineweb-edu \\
        --tokenizer Qwen/Qwen2.5-0.5B \\
        --output-dir data \\
        --max-tokens 1000000000 \\
        --quality-filter
"""

import os
import sys
import math
import time
import hashlib
import argparse
import pickle
from typing import Optional, List, Iterator, Callable
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


# =============================================================================
#  Data Quality Filter
# =============================================================================

class DataQualityFilter:
    """Filter low-quality text samples before tokenization.

    Applies a cascade of fast heuristic filters:
    1. Length filter — skip too-short or too-long texts
    2. Repetition filter — skip text with excessive n-gram repetition
    3. Language filter — basic ASCII ratio check (English proxy)
    4. Special character filter — skip text with too many special chars
    """

    def __init__(
        self,
        min_chars: int = 50,
        max_chars: int = 100_000,
        min_words: int = 10,
        max_repetition_ratio: float = 0.3,
        min_ascii_ratio: float = 0.9,
        max_special_ratio: float = 0.1,
        enabled: bool = True,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_ascii_ratio = min_ascii_ratio
        self.max_special_ratio = max_special_ratio
        self.enabled = enabled

        # Stats
        self.total = 0
        self.passed = 0
        self.filtered_reasons = defaultdict(int)

    def __call__(self, text: str) -> bool:
        """Return True if text passes all quality checks."""
        if not self.enabled:
            return True

        self.total += 1

        if not text or not text.strip():
            self.filtered_reasons["empty"] += 1
            return False

        text = text.strip()

        # Length filter
        if len(text) < self.min_chars:
            self.filtered_reasons["too_short"] += 1
            return False
        if len(text) > self.max_chars:
            self.filtered_reasons["too_long"] += 1
            return False

        # Word count
        words = text.split()
        if len(words) < self.min_words:
            self.filtered_reasons["too_few_words"] += 1
            return False

        # ASCII ratio (English proxy)
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / len(text)
        if ascii_ratio < self.min_ascii_ratio:
            self.filtered_reasons["non_english"] += 1
            return False

        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text)
        if special_ratio > self.max_special_ratio:
            self.filtered_reasons["too_many_specials"] += 1
            return False

        # Repetition filter (check 2-gram repetition)
        if len(words) >= 20:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            unique_bigrams = len(set(bigrams))
            total_bigrams = len(bigrams)
            repetition = 1.0 - (unique_bigrams / total_bigrams)
            if repetition > self.max_repetition_ratio:
                self.filtered_reasons["too_repetitive"] += 1
                return False

        self.passed += 1
        return True

    def stats(self) -> dict:
        """Return filtering statistics."""
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.passed / max(self.total, 1),
            "filtered_reasons": dict(self.filtered_reasons),
        }

    def print_stats(self):
        """Print filtering statistics."""
        s = self.stats()
        print(f"\n  Quality Filter Stats:")
        print(f"    Total:     {s['total']:,}")
        print(f"    Passed:    {s['passed']:,} ({s['pass_rate']:.1%})")
        for reason, count in sorted(s["filtered_reasons"].items(), key=lambda x: -x[1]):
            print(f"    Filtered ({reason}): {count:,}")


# =============================================================================
#  MinHash Deduplicator
# =============================================================================

class MinHashDeduplicator:
    """Approximate deduplication using MinHash signatures.

    Uses locality-sensitive hashing to detect near-duplicate documents.
    Much faster than exact dedup for large datasets.

    Algorithm:
    1. Compute shingle set (character n-grams) for each document
    2. Generate MinHash signature (K independent hash functions)
    3. Use LSH bands to find candidate duplicates
    4. Documents with Jaccard similarity > threshold are duplicates
    """

    def __init__(
        self,
        num_hashes: int = 128,
        shingle_size: int = 5,
        threshold: float = 0.8,
        num_bands: int = 16,
        enabled: bool = True,
    ):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.enabled = enabled

        # LSH buckets: band_idx -> {bucket_hash -> set of doc_ids}
        self.buckets = [defaultdict(set) for _ in range(num_bands)]
        self.seen_count = 0
        self.dedup_count = 0

        # Random hash coefficients (a*x + b mod p)
        self._prime = 2**31 - 1
        self._a = np.random.randint(1, self._prime, size=num_hashes, dtype=np.int64)
        self._b = np.random.randint(0, self._prime, size=num_hashes, dtype=np.int64)

    def _get_shingles(self, text: str) -> set:
        """Extract character n-gram shingles."""
        text = text.lower().strip()
        if len(text) < self.shingle_size:
            return {text}
        return {text[i:i + self.shingle_size] for i in range(len(text) - self.shingle_size + 1)}

    def _minhash(self, shingles: set) -> np.ndarray:
        """Compute MinHash signature for a shingle set."""
        sig = np.full(self.num_hashes, np.iinfo(np.int64).max, dtype=np.int64)

        for shingle in shingles:
            h = int(hashlib.md5(shingle.encode()).hexdigest(), 16) % self._prime
            hashes = (self._a * h + self._b) % self._prime
            sig = np.minimum(sig, hashes)

        return sig

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a near-duplicate of previously seen text.

        Also registers the text for future dedup checks.
        Returns True if duplicate (should be skipped).
        """
        if not self.enabled:
            return False

        shingles = self._get_shingles(text)
        if not shingles:
            return False

        sig = self._minhash(shingles)
        doc_id = self.seen_count
        self.seen_count += 1

        # Check LSH bands for candidate matches
        is_dup = False
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hashlib.md5(sig[start:end].tobytes()).hexdigest()

            if self.buckets[band_idx][band_hash]:
                is_dup = True

            self.buckets[band_idx][band_hash].add(doc_id)

        if is_dup:
            self.dedup_count += 1

        return is_dup

    def stats(self) -> dict:
        return {
            "seen": self.seen_count,
            "duplicates": self.dedup_count,
            "dedup_rate": self.dedup_count / max(self.seen_count, 1),
        }

    def print_stats(self):
        s = self.stats()
        print(f"\n  Deduplication Stats:")
        print(f"    Documents seen:  {s['seen']:,}")
        print(f"    Duplicates:      {s['duplicates']:,} ({s['dedup_rate']:.1%})")


# =============================================================================
#  Streaming Dataset
# =============================================================================

class StreamingDataset(IterableDataset):
    """Stream tokenized sequences from HuggingFace datasets.

    Handles:
    - On-the-fly tokenization from HF streaming datasets
    - Quality filtering and deduplication
    - Sequence packing (concatenate texts, split into fixed-length chunks)
    - Distributed-aware sharding across workers and ranks
    - Token ID 0 filtering (padding/special token artifact)

    This is the core primitive for training on petabyte-scale data
    without loading everything into RAM.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        block_size: int = 256,
        quality_filter: Optional[DataQualityFilter] = None,
        deduplicator: Optional[MinHashDeduplicator] = None,
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
        self.quality_filter = quality_filter or DataQualityFilter(enabled=False)
        self.deduplicator = deduplicator or MinHashDeduplicator(enabled=False)
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

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=True
        )

        # Load streaming dataset
        kwargs = {"streaming": True, "split": self.split}
        if self.subset:
            ds = load_dataset(self.dataset_name, self.subset, **kwargs)
        else:
            ds = load_dataset(self.dataset_name, **kwargs)

        # Distributed sharding — each rank processes different samples
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)

        # Multi-worker sharding within a single rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds = ds.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        # Shuffle with seed for reproducibility
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        # Token buffer for sequence packing
        token_buffer = []
        total_tokens = 0

        for sample in ds:
            text = sample.get(self.text_field, "")

            # Quality filter
            if not self.quality_filter(text):
                continue

            # Deduplication
            if self.deduplicator.is_duplicate(text):
                continue

            # Tokenize
            ids = tokenizer.encode(text, add_special_tokens=False)
            # Filter out token 0 (padding artifact)
            ids = [t for t in ids if t != 0]

            if not ids:
                continue

            token_buffer.extend(ids)
            total_tokens += len(ids)

            # Yield packed sequences when buffer has enough tokens
            while len(token_buffer) >= self.block_size + 1:
                chunk = token_buffer[:self.block_size + 1]
                token_buffer = token_buffer[self.block_size:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}

            # Check token limit
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
    quality_filter: bool = True,
    deduplication: bool = True,
    text_field: str = "text",
    split: str = "train",
    max_tokens: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a production streaming dataloader for training.

    Args:
        dataset_name: HuggingFace dataset (e.g. "HuggingFaceFW/fineweb-edu")
        tokenizer_name: HuggingFace tokenizer
        subset: Dataset subset/config
        block_size: Sequence length
        batch_size: Batch size
        num_workers: DataLoader workers for prefetching
        quality_filter: Enable quality filtering
        deduplication: Enable MinHash deduplication
        text_field: Name of the text column in the dataset
        split: Dataset split ("train", "validation")
        max_tokens: Maximum tokens to process (None = unlimited)
        rank: FSDP rank (for distributed sharding)
        world_size: Total FSDP world size
        seed: Random seed
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader yielding batches of {"input_ids": (B, T), "labels": (B, T)}
    """
    qf = DataQualityFilter(enabled=quality_filter)
    dedup = MinHashDeduplicator(enabled=deduplication)

    dataset = StreamingDataset(
        dataset_name=dataset_name,
        subset=subset,
        tokenizer_name=tokenizer_name,
        block_size=block_size,
        quality_filter=qf,
        deduplicator=dedup,
        text_field=text_field,
        split=split,
        max_tokens=max_tokens,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader


# =============================================================================
#  Standalone Data Preparation (with streaming + filtering)
# =============================================================================

def prepare_filtered_data(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: Optional[str] = "sample-10BT",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "data",
    max_tokens: int = 500_000_000,
    quality_filter: bool = True,
    deduplication: bool = True,
    val_fraction: float = 0.02,
    text_field: str = "text",
):
    """Prepare a high-quality training dataset with filtering and dedup.

    This is the "prepare once, train many times" approach for when you
    want to save tokenized data to disk rather than stream on-the-fly.
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")

    # Setup filters
    qf = DataQualityFilter(enabled=quality_filter)
    dedup = MinHashDeduplicator(enabled=deduplication)

    # Stream dataset
    print(f"\nStreaming dataset: {dataset_name}")
    kwargs = {"streaming": True, "split": "train"}
    if subset:
        ds = load_dataset(dataset_name, subset, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)

    # Tokenize with filtering — CHUNKED WRITES to avoid OOM
    # Instead of accumulating all tokens in a Python list (which uses ~28 bytes
    # per int = 28GB for 1B tokens), we write to disk in chunks of 10M tokens.
    print("Tokenizing with quality filtering...")
    CHUNK_SIZE = 10_000_000  # Write every 10M tokens
    chunk_buffer = []
    total_tokens = 0
    t0 = time.time()

    # Temporary file for streaming writes
    tmp_path = os.path.join(output_dir, "tokens_tmp.bin")
    tmp_file = open(tmp_path, "wb")

    for i, sample in enumerate(ds):
        text = sample.get(text_field, "")

        # Quality filter
        if not qf(text):
            continue

        # Dedup
        if dedup.is_duplicate(text):
            continue

        # Tokenize
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [t for t in ids if t != 0]  # Filter padding token

        if not ids:
            continue

        chunk_buffer.extend(ids)
        total_tokens += len(ids)

        # Flush chunk to disk when buffer is large enough
        if len(chunk_buffer) >= CHUNK_SIZE:
            arr = np.array(chunk_buffer, dtype=np.uint32)
            arr.tofile(tmp_file)
            chunk_buffer = []

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed
            print(f"    {i+1:,} samples | {total_tokens:,} tokens | {tps:,.0f} tok/s")

        if total_tokens >= max_tokens:
            print(f"  Reached token limit: {max_tokens:,}")
            break

    # Flush remaining tokens
    if chunk_buffer:
        arr = np.array(chunk_buffer, dtype=np.uint32)
        arr.tofile(tmp_file)
        chunk_buffer = []

    tmp_file.close()

    # Print filter stats
    qf.print_stats()
    dedup.print_stats()

    # Split train/val using memory-mapped file (no RAM spike)
    print(f"\n  Total tokens: {total_tokens:,}")
    all_tokens = np.memmap(tmp_path, dtype=np.uint32, mode='r')
    split_idx = int(len(all_tokens) * (1 - val_fraction))

    print(f"  Train tokens: {split_idx:,}")
    print(f"  Val tokens:   {len(all_tokens) - split_idx:,}")

    # Save train/val splits
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")

    # Copy slices efficiently using memmap
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint32)
    train_tokens.tofile(train_path)
    del train_tokens

    val_tokens = np.array(all_tokens[split_idx:], dtype=np.uint32)
    val_tokens.tofile(val_path)
    del val_tokens

    # Cleanup temp file
    del all_tokens
    os.remove(tmp_path)

    meta = {
        "tokenizer_type": "huggingface",
        "tokenizer_name": tokenizer_name,
        "vocab_size": vocab_size,
        "dataset": dataset_name,
        "subset": subset,
        "quality_filtered": quality_filter,
        "deduplicated": deduplication,
        "total_tokens": total_tokens,
        "filter_stats": qf.stats(),
        "dedup_stats": dedup.stats(),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({os.path.getsize(train_path) / 1e9:.2f} GB)")
    print(f"  Val:   {val_path} ({os.path.getsize(val_path) / 1e6:.2f} MB)")
    print(f"  Meta:  {meta_path}")


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production data pipeline with quality filtering & dedup"
    )
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="Dataset subset/config")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace tokenizer")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for tokenized data")
    parser.add_argument("--max-tokens", type=int, default=500_000_000,
                        help="Maximum tokens to process")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable quality filtering")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication")
    parser.add_argument("--val-fraction", type=float, default=0.02,
                        help="Validation fraction")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Name of text column in dataset")

    args = parser.parse_args()
    prepare_filtered_data(
        dataset_name=args.dataset,
        subset=args.subset,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        quality_filter=not args.no_quality_filter,
        deduplication=not args.no_dedup,
        val_fraction=args.val_fraction,
        text_field=args.text_field,
    )
