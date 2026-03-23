"""
GPT Text Generation / Inference
================================
Load a trained GPT model checkpoint and generate text.
Uses KV-cache for fast autoregressive generation.

Usage:
    # Generate text with default settings:
    python generate.py

    # Interactive prompt mode:
    python generate.py --interactive

    # Custom generation parameters:
    python generate.py --prompt "Once upon a time" --max-tokens 500 --temperature 0.8

    # Disable KV-cache (for debugging):
    python generate.py --no-cache
"""

import os
import sys
import time
import argparse
import pickle

import torch

from config import GPTConfig
from model import GPT


def load_model(checkpoint_path: str, device: str):
    """Load a trained GPT model from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild model config
    config_dict = checkpoint["model_config"]
    config = GPTConfig(**config_dict)

    # Create and load model
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    iter_num = checkpoint.get("iter_num", "?")
    val_loss = checkpoint.get("best_val_loss", "?")

    # Check if aligned
    alignment = checkpoint.get("alignment", None)
    if alignment:
        print(f"Loaded aligned model ({alignment['method'].upper()}, "
              f"β={alignment['beta']}) from iteration {iter_num}")
    else:
        print(f"Loaded model from iteration {iter_num} (val loss: {val_loss})")

    # Print architecture features
    features = []
    if config.n_kv_head < config.n_head:
        features.append(f"GQA {config.n_head}Q/{config.n_kv_head}KV")
    if config.use_moe:
        features.append(f"MoE {config.n_experts_active}/{config.n_experts}")
    if config.use_rope:
        features.append("RoPE")
    if config.use_swiglu:
        features.append("SwiGLU")
    if features:
        print(f"Architecture: {' | '.join(features)}")

    return model, config


def load_tokenizer(data_dir: str):
    """Load the tokenizer used during training."""
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"Error: Tokenizer metadata not found at {meta_path}")
        print("Make sure you've run data/prepare_data.py first.")
        sys.exit(1)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    tokenizer_type = meta.get("tokenizer_type", "char")

    if tokenizer_type == "char":
        from data.prepare_data import CharTokenizer
        tokenizer = CharTokenizer()
        tokenizer.vocab_size = meta["vocab_size"]
        tokenizer.char_to_idx = meta["char_to_idx"]
        tokenizer.idx_to_char = meta["idx_to_char"]
        print(f"Loaded character tokenizer (vocab_size={tokenizer.vocab_size})")
    elif tokenizer_type == "tiktoken":
        from data.prepare_data import TiktokenWrapper
        tokenizer = TiktokenWrapper()
        print(f"Loaded tiktoken tokenizer (vocab_size={tokenizer.vocab_size})")
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
    use_cache: bool = True,
):
    """Generate text from a prompt. Returns text and generation stats."""
    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate with timing
    t0 = time.time()
    with torch.no_grad():
        y = model.generate(
            x, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            use_cache=use_cache,
        )
    t1 = time.time()

    # Decode
    generated_ids = y[0].tolist()
    text = tokenizer.decode(generated_ids)

    # Stats
    n_generated = len(generated_ids) - len(prompt_ids)
    elapsed = t1 - t0
    tokens_per_sec = n_generated / elapsed if elapsed > 0 else 0

    stats = {
        "n_generated": n_generated,
        "elapsed": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }

    return text, stats


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained GPT model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing tokenizer metadata (meta.pkl)")
    parser.add_argument("--prompt", type=str, default="\n",
                        help="Starting prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0 = greedy, 1.0 = diverse)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0 = no filtering)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: enter prompts continuously")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable KV-cache (for debugging)")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load model and tokenizer
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Train a model first with: python train.py")
        sys.exit(1)

    model, config = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.data_dir)

    top_k = args.top_k if args.top_k > 0 else None
    use_cache = not args.no_cache

    if args.interactive:
        # ── Interactive mode ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  GPT Interactive Generation")
        print(f"  Temperature: {args.temperature} | Top-k: {top_k}")
        print(f"  KV-Cache: {'ON' if use_cache else 'OFF'}")
        print(f"  Type your prompt and press Enter. Type 'quit' to exit.")
        print(f"{'='*60}\n")

        while True:
            try:
                prompt = input("You> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not prompt:
                prompt = "\n"

            text, stats = generate_text(
                model, tokenizer, prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=top_k,
                device=device,
                use_cache=use_cache,
            )

            print(f"\nGPT> {text}")
            print(f"  [{stats['n_generated']} tokens, {stats['tokens_per_sec']:.1f} tok/s, "
                  f"{'cached' if use_cache else 'no-cache'}]\n")
    else:
        # ── Single generation ─────────────────────────────────────────────
        print(f"\nPrompt: {repr(args.prompt)}")
        print(f"Temperature: {args.temperature} | Top-k: {top_k} | "
              f"KV-Cache: {'ON' if use_cache else 'OFF'}")
        print(f"Generating {args.max_tokens} tokens...\n")
        print("─" * 60)

        text, stats = generate_text(
            model, tokenizer, prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
            device=device,
            use_cache=use_cache,
        )
        print(text)
        print("─" * 60)
        print(f"  {stats['n_generated']} tokens in {stats['elapsed']:.2f}s "
              f"({stats['tokens_per_sec']:.1f} tokens/sec)")


if __name__ == "__main__":
    main()
