"""
GPT Text Generation / Inference
================================
Load a trained GPT model checkpoint and generate text.
Uses KV-cache for fast autoregressive generation.

Features:
  - Top-k, Top-p (nucleus), Min-p sampling
  - Repetition penalty
  - Speculative decoding with draft model
  - KV-cache for fast generation

Usage:
    # Generate text with default settings:
    python generate.py

    # Interactive prompt mode:
    python generate.py --interactive

    # Advanced sampling:
    python generate.py --prompt "Once" --top-p 0.9 --min-p 0.05 --rep-penalty 1.2

    # Speculative decoding (2-3x faster):
    python generate.py --draft-checkpoint checkpoints/small.pt --spec-k 5

    # Disable KV-cache (for debugging):
    python generate.py --no-cache
"""

import os
import sys
import time
import argparse
import pickle

import torch

from supergpt.core.config import GPTConfig
from supergpt.core.model import GPT, SpeculativeGenerator


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
    distillation = checkpoint.get("distillation", None)
    if alignment:
        print(f"Loaded aligned model ({alignment['method'].upper()}, "
              f"\u03b2={alignment['beta']}) from iteration {iter_num}")
    elif distillation:
        teacher = distillation.get('teacher', 'unknown')
        print(f"Loaded distilled model (teacher: {teacher}) from iteration {iter_num} (val loss: {val_loss})")
    else:
        print(f"Loaded model from iteration {iter_num} (val loss: {val_loss})")

    # Print architecture features
    features = []
    if config.use_mla:
        features.append(f"MLA(kv_rank={config.kv_lora_rank})")
    elif config.n_kv_head < config.n_head:
        features.append(f"GQA {config.n_head}Q/{config.n_kv_head}KV")
    if config.sliding_window > 0:
        sw_type = "alternating" if config.alternating_layers else "full"
        features.append(f"SlidingWindow({config.sliding_window}, {sw_type})")
    if config.use_moe:
        features.append(f"MoE {config.n_experts_active}/{config.n_experts}")
    if config.use_rope:
        rope_info = "RoPE"
        if config.rope_scaling_type != "none":
            rope_info += f"-{config.rope_scaling_type.upper()}({config.rope_scaling_factor}x)"
        features.append(rope_info)
    if config.use_swiglu:
        features.append("SwiGLU")
    if features:
        print(f"Architecture: {' | '.join(features)}")

    return model, config, checkpoint


def load_tokenizer(data_dir: str, checkpoint=None):
    """Load the tokenizer used during training.

    For distilled models from HuggingFace teachers, automatically loads
    the teacher's tokenizer instead of looking at meta.pkl.
    """
    # Check if this is a distilled model with a HF teacher
    if checkpoint:
        distillation = checkpoint.get("distillation", {})
        if distillation.get("teacher_type") == "huggingface":
            teacher_name = distillation.get("teacher", "")
            if teacher_name:
                try:
                    from transformers import AutoTokenizer
                    hf_tok = AutoTokenizer.from_pretrained(
                        teacher_name, trust_remote_code=True
                    )
                    # Wrap HF tokenizer to have encode/decode interface
                    class HFTokenizerWrapper:
                        def __init__(self, hf_tok):
                            self.hf_tok = hf_tok
                            self.vocab_size = hf_tok.vocab_size
                        def encode(self, text):
                            return self.hf_tok.encode(text)
                        def decode(self, tokens):
                            return self.hf_tok.decode(tokens, skip_special_tokens=True)
                    tokenizer = HFTokenizerWrapper(hf_tok)
                    print(f"Loaded HuggingFace tokenizer from teacher: {teacher_name} "
                          f"(vocab_size={tokenizer.vocab_size})")
                    return tokenizer
                except Exception as e:
                    print(f"Warning: Could not load HF tokenizer '{teacher_name}': {e}")
                    print("Falling back to meta.pkl tokenizer.")

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
    elif tokenizer_type == "huggingface":
        tokenizer_name = meta.get("tokenizer_name", "Qwen/Qwen2.5-0.5B")
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            class HFTokenizerWrapper:
                def __init__(self, hf_tok):
                    self.hf_tok = hf_tok
                    self.vocab_size = hf_tok.vocab_size
                def encode(self, text):
                    return self.hf_tok.encode(text)
                def decode(self, tokens):
                    return self.hf_tok.decode(tokens, skip_special_tokens=True)
            tokenizer = HFTokenizerWrapper(hf_tok)
            print(f"Loaded HuggingFace tokenizer: {tokenizer_name} (vocab_size={tokenizer.vocab_size})")
        except Exception as e:
            raise ValueError(f"Could not load HF tokenizer '{tokenizer_name}': {e}")
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
    top_p: float = None,
    min_p: float = None,
    repetition_penalty: float = 1.0,
    device: str = "cpu",
    use_cache: bool = True,
    spec_generator=None,
):
    """Generate text from a prompt. Returns text and generation stats."""
    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate with timing
    t0 = time.time()
    with torch.no_grad():
        if spec_generator is not None:
            y = spec_generator.generate(
                x, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
            )
        else:
            y = model.generate(
                x, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
                top_p=top_p, min_p=min_p,
                repetition_penalty=repetition_penalty,
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
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus (top-p) sampling threshold (e.g. 0.9)")
    parser.add_argument("--min-p", type=float, default=None,
                        help="Min-p dynamic threshold (e.g. 0.05)")
    parser.add_argument("--rep-penalty", type=float, default=1.0,
                        help="Repetition penalty (>1 = less repetition)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: enter prompts continuously")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable KV-cache (for debugging)")

    # Speculative decoding
    parser.add_argument("--draft-checkpoint", type=str, default=None,
                        help="Path to draft model checkpoint for speculative decoding")
    parser.add_argument("--spec-k", type=int, default=5,
                        help="Number of speculative draft tokens (default: 5)")

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

    model, config, checkpoint = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.data_dir, checkpoint=checkpoint)

    top_k = args.top_k if args.top_k > 0 else None
    use_cache = not args.no_cache

    # Setup speculative decoding if draft model provided
    spec_generator = None
    if args.draft_checkpoint:
        if not os.path.exists(args.draft_checkpoint):
            print(f"Error: Draft checkpoint not found at {args.draft_checkpoint}")
            sys.exit(1)
        print(f"\nLoading draft model for speculative decoding (k={args.spec_k})...")
        draft_model, _ = load_model(args.draft_checkpoint, device)
        spec_generator = SpeculativeGenerator(model, draft_model, k=args.spec_k)
        print("Speculative decoding enabled \u2713")

    if args.interactive:
        # ── Interactive mode ──────────────────────────────────────────────
        sampling_info = f"T={args.temperature}"
        if top_k: sampling_info += f" | top-k={top_k}"
        if args.top_p: sampling_info += f" | top-p={args.top_p}"
        if args.min_p: sampling_info += f" | min-p={args.min_p}"
        if args.rep_penalty != 1.0: sampling_info += f" | rep={args.rep_penalty}"

        print(f"\n{'='*60}")
        print(f"  superGPT Interactive Generation")
        print(f"  Sampling: {sampling_info}")
        print(f"  KV-Cache: {'ON' if use_cache else 'OFF'}")
        if spec_generator:
            print(f"  Speculative Decoding: ON (k={args.spec_k})")
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
                top_p=args.top_p,
                min_p=args.min_p,
                repetition_penalty=args.rep_penalty,
                device=device,
                use_cache=use_cache,
                spec_generator=spec_generator,
            )

            mode = "spec" if spec_generator else ("cached" if use_cache else "no-cache")
            print(f"\nGPT> {text}")
            print(f"  [{stats['n_generated']} tokens, {stats['tokens_per_sec']:.1f} tok/s, "
                  f"{mode}]\n")
    else:
        # ── Single generation ─────────────────────────────────────────────
        print(f"\nPrompt: {repr(args.prompt)}")
        sampling_info = f"T={args.temperature}"
        if top_k: sampling_info += f" | top-k={top_k}"
        if args.top_p: sampling_info += f" | top-p={args.top_p}"
        if args.min_p: sampling_info += f" | min-p={args.min_p}"
        if args.rep_penalty != 1.0: sampling_info += f" | rep={args.rep_penalty}"
        print(f"Sampling: {sampling_info}")
        print(f"KV-Cache: {'ON' if use_cache else 'OFF'}")
        if spec_generator:
            print(f"Speculative Decoding: ON (k={args.spec_k})")
        print(f"Generating {args.max_tokens} tokens...\n")
        print("\u2500" * 60)

        text, stats = generate_text(
            model, tokenizer, prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.rep_penalty,
            device=device,
            use_cache=use_cache,
            spec_generator=spec_generator,
        )
        print(text)
        print("\u2500" * 60)
        mode = "speculative" if spec_generator else ("cached" if use_cache else "no-cache")
        print(f"  {stats['n_generated']} tokens in {stats['elapsed']:.2f}s "
              f"({stats['tokens_per_sec']:.1f} tokens/sec, {mode})")


if __name__ == "__main__":
    main()
