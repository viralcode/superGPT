"""
Prepare HuggingFace dataset with Qwen tokenizer for superGPT training.

Downloads a high-quality text dataset and tokenizes it with the Qwen2.5 tokenizer
for training a model that generates coherent general-purpose text.

Usage:
    python data/prepare_hf_data.py
    python data/prepare_hf_data.py --dataset wikitext --subset wikitext-103-raw-v1
    python data/prepare_hf_data.py --max-samples 50000
"""

import os
import sys
import argparse
import pickle
import numpy as np

def prepare_hf_data(
    dataset_name="wikitext",
    subset="wikitext-2-raw-v1",
    tokenizer_name="Qwen/Qwen2.5-0.5B",
    output_dir=None,
    max_samples=None,
    val_fraction=0.1,
):
    """Download HuggingFace dataset and tokenize with Qwen tokenizer."""

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load tokenizer ────────────────────────────────────────────
    print(f"Loading tokenizer: {tokenizer_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")

    # ── Step 2: Download dataset ──────────────────────────────────────────
    print(f"\nDownloading dataset: {dataset_name}/{subset}")
    from datasets import load_dataset
    ds = load_dataset(dataset_name, subset)
    print(f"  Train samples: {len(ds['train']):,}")
    if 'validation' in ds:
        print(f"  Val samples: {len(ds['validation']):,}")
    if 'test' in ds:
        print(f"  Test samples: {len(ds['test']):,}")

    # ── Step 3: Combine and tokenize ──────────────────────────────────────
    print("\nTokenizing...")

    def tokenize_split(split_data, max_n=None):
        all_ids = []
        texts = split_data['text'] if 'text' in split_data.features else split_data[list(split_data.features.keys())[0]]
        count = 0
        for text in texts:
            if not text or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            count += 1
            if max_n and count >= max_n:
                break
            if count % 5000 == 0:
                print(f"    Processed {count:,} samples ({len(all_ids):,} tokens)")
        return all_ids

    if 'validation' in ds:
        # Dataset has its own train/val split
        train_ids = tokenize_split(ds['train'], max_samples)
        val_ids = tokenize_split(ds['validation'], max_samples // 10 if max_samples else None)
    else:
        # Create our own split
        all_ids = tokenize_split(ds['train'], max_samples)
        split_idx = int(len(all_ids) * (1 - val_fraction))
        train_ids = all_ids[:split_idx]
        val_ids = all_ids[split_idx:]

    print(f"\n  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")

    # ── Step 4: Save ──────────────────────────────────────────────────────
    train_arr = np.array(train_ids, dtype=np.uint32)
    val_arr = np.array(val_ids, dtype=np.uint32)

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    # Save meta with tokenizer info
    meta = {
        "tokenizer_type": "huggingface",
        "tokenizer_name": tokenizer_name,
        "vocab_size": vocab_size,
        "dataset": f"{dataset_name}/{subset}",
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    # Also save the raw text for reference
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({os.path.getsize(train_path) / 1e6:.2f} MB)")
    print(f"  Val:   {val_path} ({os.path.getsize(val_path) / 1e6:.2f} MB)")
    print(f"  Meta:  {meta_path}")
    print(f"\n  Vocab size: {vocab_size:,}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"\nDone! Train with:")
    print(f"  python scripts/train.py --preset small --vocab-size {vocab_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HF data with Qwen tokenizer")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1",
                        help="Dataset subset/config")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace tokenizer to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Validation fraction")

    args = parser.parse_args()
    prepare_hf_data(
        dataset_name=args.dataset,
        subset=args.subset,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_fraction=args.val_fraction,
    )
