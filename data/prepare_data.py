"""
Data Preparation & Tokenization
================================
Prepares text data for training the GPT model.

Usage:
    # Download and prepare Shakespeare (sample) with BPE tokenizer:
    python data/prepare_data.py

    # Prepare your own custom text file:
    python data/prepare_data.py --input your_data.txt

    # Use character-level tokenizer (simpler, no dependencies):
    python data/prepare_data.py --tokenizer char
"""

import os
import sys
import subprocess
import argparse
import pickle
import urllib.request
import numpy as np


def download_shakespeare(output_path: str) -> str:
    """Download the tiny Shakespeare dataset as a sample corpus."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
    return output_path


class CharTokenizer:
    """Simple character-level tokenizer.

    Maps each unique character to an integer. No external dependencies needed.
    Works great for small/medium datasets and quick experimentation.
    """
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        print(f"Character-level tokenizer: vocab_size = {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")

    def encode(self, text: str) -> list:
        """Convert text to list of integer token IDs."""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, ids: list) -> str:
        """Convert list of integer token IDs back to text."""
        return "".join([self.idx_to_char.get(i, "?") for i in ids])

    def save(self, path: str):
        """Save tokenizer metadata."""
        meta = {
            "tokenizer_type": "char",
            "vocab_size": self.vocab_size,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from saved metadata."""
        with open(path, "rb") as f:
            meta = pickle.load(f)
        tok = cls()
        tok.vocab_size = meta["vocab_size"]
        tok.char_to_idx = meta["char_to_idx"]
        tok.idx_to_char = meta["idx_to_char"]
        return tok


class TiktokenWrapper:
    """Wrapper around OpenAI's tiktoken BPE tokenizer.

    Uses the same tokenizer as GPT-4 (cl100k_base encoding).
    Automatically installs tiktoken if not present.
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError:
            print("tiktoken not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
            import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        print(f"Tiktoken BPE tokenizer ({encoding_name}): vocab_size = {self.vocab_size}")

    def encode(self, text: str) -> list:
        return self.enc.encode(text, allowed_special=set())

    def decode(self, ids: list) -> str:
        return self.enc.decode(ids)

    def save(self, path: str):
        meta = {
            "tokenizer_type": "tiktoken",
            "vocab_size": self.vocab_size,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)


def prepare_data(
    input_file: str = None,
    tokenizer_type: str = "tiktoken",
    output_dir: str = None,
    val_fraction: float = 0.1,
):
    """
    Prepare text data for GPT training.

    1. Reads text from input file (or downloads Shakespeare)
    2. Tokenizes the text
    3. Splits into train/val
    4. Saves as binary .bin files
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Get the text data ─────────────────────────────────────────
    if input_file is None:
        input_file = os.path.join(output_dir, "input.txt")
        if not os.path.exists(input_file):
            download_shakespeare(input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nDataset: {len(text):,} characters")

    # ── Step 2: Tokenize ──────────────────────────────────────────────────
    if tokenizer_type == "char":
        tokenizer = CharTokenizer()
        tokenizer.fit(text)
        dtype = np.uint16  # Char vocab is always small
    elif tokenizer_type == "tiktoken":
        tokenizer = TiktokenWrapper()
        dtype = np.uint32  # tiktoken has ~100k vocab
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_type}")

    # Encode the full text
    print("Tokenizing...")
    ids = tokenizer.encode(text)
    print(f"Total tokens: {len(ids):,}")

    # ── Step 3: Train/Val split ───────────────────────────────────────────
    n = len(ids)
    split_idx = int(n * (1 - val_fraction))
    train_ids = np.array(ids[:split_idx], dtype=dtype)
    val_ids = np.array(ids[split_idx:], dtype=dtype)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")

    # ── Step 4: Save to binary files ──────────────────────────────────────
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    tokenizer.save(meta_path)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({os.path.getsize(train_path) / 1e6:.2f} MB)")
    print(f"  Val:   {val_path} ({os.path.getsize(val_path) / 1e6:.2f} MB)")
    print(f"  Meta:  {meta_path}")
    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"\nDone! You can now train with: python train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for GPT training")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input text file. Downloads Shakespeare if not provided.")
    parser.add_argument("--tokenizer", type=str, default="tiktoken",
                        choices=["char", "tiktoken"],
                        help="Tokenizer type: 'tiktoken' (GPT-4 BPE, default) or 'char' (simple)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for processed data")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of data to use for validation (default: 0.1)")

    args = parser.parse_args()
    prepare_data(
        input_file=args.input,
        tokenizer_type=args.tokenizer,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
    )
