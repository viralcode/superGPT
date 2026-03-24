"""
LoRA Fine-tuning Script
========================
Fine-tune a pre-trained superGPT model using LoRA (Low-Rank Adaptation).
Only ~1-3% of parameters are trainable, making it practical on consumer GPUs.

Usage:
    # Fine-tune on new data with LoRA:
    python finetune.py --checkpoint checkpoints/best.pt --data data/

    # Custom LoRA settings:
    python finetune.py --checkpoint best.pt --data data/ --lora-rank 32 --lora-alpha 64

    # Fine-tune only attention layers:
    python finetune.py --checkpoint best.pt --data data/ --target q_proj,k_proj,v_proj

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

import os
import sys
import math
import time
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from supergpt.core.config import GPTConfig
from supergpt.core.model import GPT
from supergpt.training.lora import apply_lora, merge_lora, save_lora


def load_data(data_dir: str, split: str, block_size: int, batch_size: int, device: str):
    """Load a batch of data from the memory-mapped binary file."""
    data_path = os.path.join(data_dir, f"{split}.bin")

    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta.get("vocab_size", 0)
        if vocab_size > 65535 or meta.get("tokenizer_type") == "tiktoken":
            dtype = np.uint32
        else:
            dtype = np.uint16
    else:
        dtype = np.uint16

    data = np.memmap(data_path, dtype=dtype, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data_dir, block_size, batch_size, device, n_iters=50):
    """Quick evaluation on validation set."""
    model.eval()
    losses = []
    for _ in range(n_iters):
        X, Y = load_data(data_dir, "val", block_size, batch_size, device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def finetune(args):
    """Main LoRA fine-tuning loop."""

    # ── Device & dtype ────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # ── Load pre-trained model ────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    print(f"Loading pre-trained model: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])

    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # ── Apply LoRA ────────────────────────────────────────────────────────
    target_modules = None
    if args.target:
        target_modules = [t.strip() for t in args.target.split(",")]

    apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    # ── Mixed precision ──────────────────────────────────────────────────
    if "cuda" in device and torch.cuda.is_bf16_supported():
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif "cuda" in device:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = nullcontext()

    # ── Optimizer (only LoRA params) ──────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LoRA Fine-tuning")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  LoRA rank:    {args.lora_rank}")
    print(f"  LoRA alpha:   {args.lora_alpha}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max iters:    {args.max_iters}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Trainable:    {sum(p.numel() for p in trainable):,} params")
    print(f"{'='*60}\n")

    model.train()
    t0 = time.time()
    best_val_loss = float("inf")

    for iter_num in range(args.max_iters):
        # Cosine LR schedule
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * iter_num / args.max_iters))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Evaluate
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(model, args.data, config.block_size,
                              args.batch_size, device)
            print(f"  Step {iter_num:>5d} | val loss: {val_loss:.4f} | lr: {lr:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save LoRA weights
                os.makedirs(args.output_dir, exist_ok=True)
                lora_path = os.path.join(args.output_dir, "lora_best.pt")
                save_lora(model, lora_path)

        # Train step
        X, Y = load_data(args.data, "train", config.block_size,
                         args.batch_size, device)

        with ctx:
            _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    # ── Save final merged model ───────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Option 1: Save merged model (LoRA baked in)
    if args.merge:
        merge_lora(model)
        merged_path = os.path.join(args.output_dir, "finetuned_merged.pt")
        ckpt = {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "iter_num": checkpoint.get("iter_num", 0),
            "best_val_loss": best_val_loss,
            "finetune": {
                "method": "lora",
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "iters": args.max_iters,
            },
        }
        torch.save(ckpt, merged_path)
        print(f"\n  Merged checkpoint: {merged_path}")

    # Option 2: Save LoRA weights only
    lora_path = os.path.join(args.output_dir, "lora_final.pt")
    save_lora(model, lora_path)

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-tuning Complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    print(f"\nGenerate with: python generate.py --checkpoint "
          f"{os.path.join(args.output_dir, 'finetuned_merged.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for superGPT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Pre-trained model checkpoint")
    parser.add_argument("--data", type=str, default="data",
                        help="Directory with train.bin and val.bin")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA alpha scaling (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    parser.add_argument("--target", type=str, default=None,
                        help="Comma-separated target modules (default: all projections)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-iters", type=int, default=1000,
                        help="Max training iterations (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Evaluation interval (default: 100)")
    parser.add_argument("--merge", action="store_true", default=True,
                        help="Merge LoRA into base model after training (default: True)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()
    finetune(args)
