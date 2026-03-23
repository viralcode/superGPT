"""
GPT Training Script with FSDP Multi-GPU Support
=================================================
Trains the GPT model on prepared text data.
Supports single-GPU, Apple Silicon (MPS), and multi-GPU FSDP training.

Usage:
    # Single GPU / CPU training:
    python train.py --preset small

    # Multi-GPU FSDP training:
    torchrun --nproc_per_node=4 train.py --preset large --distributed

    # Resume from checkpoint:
    python train.py --resume checkpoints/latest.pt
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

from config import GPTConfig, TrainConfig, get_model_config
from model import GPT


# ── FSDP imports (optional) ──────────────────────────────────────────────────
FSDP_AVAILABLE = False
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    import torch.distributed as dist
    FSDP_AVAILABLE = True
except ImportError:
    pass


def setup_distributed():
    """Initialize distributed process group for FSDP."""
    if not FSDP_AVAILABLE:
        print("Error: FSDP requires PyTorch >= 2.0 with distributed support.")
        sys.exit(1)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (for logging)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_device(config: TrainConfig):
    """Determine the best available device."""
    if config.distributed:
        return f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    if config.device != "auto":
        return config.device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype(config: TrainConfig, device: str):
    """Determine the best dtype for mixed precision training."""
    if config.dtype != "auto":
        return config.dtype
    if "cuda" in device and torch.cuda.is_bf16_supported():
        return "bfloat16"
    elif "cuda" in device:
        return "float16"
    else:
        return "float32"


def get_lr(it: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def load_data(data_dir: str, split: str, block_size: int, batch_size: int, device: str):
    """Load a batch of data from the memory-mapped binary file."""
    data_path = os.path.join(data_dir, f"{split}.bin")

    # Detect dtype from meta.pkl
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("tokenizer_type") == "tiktoken":
            dtype = np.uint32
        else:
            dtype = np.uint16
    else:
        dtype = np.uint16

    data = np.memmap(data_path, dtype=dtype, mode="r")

    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, eval_iters):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = load_data(data_dir, split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model_config: GPTConfig, train_config: TrainConfig):
    """Main training loop with optional FSDP support."""

    # ── Distributed setup ─────────────────────────────────────────────────
    local_rank = 0
    if train_config.distributed:
        local_rank = setup_distributed()

    # ── Device & dtype ────────────────────────────────────────────────────
    device = get_device(train_config)
    dtype_str = get_dtype(train_config, device)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    ptdtype = dtype_map[dtype_str]

    # Mixed precision context
    if "cuda" in device:
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = nullcontext()

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  GPT-4 Architecture Model Training")
        print(f"{'='*60}")
        print(f"  Device:       {device}")
        print(f"  Dtype:        {dtype_str}")
        print(f"  Distributed:  {train_config.distributed}")
        print(f"  Model:        {model_config.n_layer}L / {model_config.n_head}H / {model_config.n_embd}E")
        print(f"  Block size:   {model_config.block_size}")
        print(f"  Batch size:   {train_config.batch_size}")
        print(f"  Max iters:    {train_config.max_iters}")
        print(f"  Learning rate: {train_config.learning_rate}")
        if model_config.use_mla:
            print(f"  Attention:    MLA (kv_rank={model_config.kv_lora_rank})")
        elif model_config.n_kv_head < model_config.n_head:
            print(f"  Attention:    GQA ({model_config.n_head}Q / {model_config.n_kv_head}KV)")
        if model_config.use_moe:
            routing = "aux-loss-free" if model_config.aux_loss_free else "aux-loss"
            print(f"  MoE:          {model_config.n_experts} experts, "
                  f"top-{model_config.n_experts_active} active, "
                  f"{model_config.n_shared_experts} shared ({routing})")
        if model_config.n_predict_tokens > 1:
            print(f"  MTP:          {model_config.n_predict_tokens} tokens")
        print(f"{'='*60}\n")

    # ── Load tokenizer metadata ───────────────────────────────────────────
    meta_path = os.path.join(train_config.data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model_config.vocab_size = meta["vocab_size"]
        if is_main_process():
            print(f"Vocab size from data: {model_config.vocab_size}")
    else:
        if is_main_process():
            print(f"Warning: No meta.pkl found. Using default vocab_size = {model_config.vocab_size}")

    # ── Create model ──────────────────────────────────────────────────────
    model = GPT(model_config)
    model.to(device)

    # ── Wrap with FSDP if distributed ─────────────────────────────────────
    if train_config.distributed:
        # Auto-wrap policy: wrap modules with >100K parameters
        auto_wrap_policy = size_based_auto_wrap_policy
        wrap_kwargs = {"min_num_params": 100_000}

        # Mixed precision policy for FSDP
        mp_policy = MixedPrecision(
            param_dtype=ptdtype,
            reduce_dtype=ptdtype,
            buffer_dtype=ptdtype,
        )

        model = FSDP(
            model,
            auto_wrap_policy=lambda *args, **kwargs: auto_wrap_policy(
                *args, **kwargs, **wrap_kwargs
            ),
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local_rank,
        )
        if is_main_process():
            print("Model wrapped with FSDP (FullyShardedDataParallel)")
    elif train_config.compile_model and hasattr(torch, "compile"):
        if is_main_process():
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = ("cuda" in device) and not train_config.distributed
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        fused=use_fused,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_iter = 0
    best_val_loss = float("inf")

    if train_config.resume_from and os.path.exists(train_config.resume_from):
        if is_main_process():
            print(f"Resuming from checkpoint: {train_config.resume_from}")
        checkpoint = torch.load(train_config.resume_from, map_location=device, weights_only=False)

        # Handle FSDP vs non-FSDP checkpoint loading
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter_num"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if is_main_process():
            print(f"Resumed at iteration {start_iter}, best val loss: {best_val_loss:.4f}")

    # ── GradScaler for FP16 ───────────────────────────────────────────────
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(dtype_str == "float16" and "cuda" in device)
    )

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    t0 = time.time()
    raw_model = model.module if hasattr(model, "module") else model

    for iter_num in range(start_iter, train_config.max_iters):
        # Set learning rate
        lr = get_lr(iter_num, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Evaluate ──────────────────────────────────────────────────────
        if iter_num % train_config.eval_interval == 0 and is_main_process():
            losses = estimate_loss(
                model, train_config.data_dir, model_config.block_size,
                train_config.batch_size, device, train_config.eval_iters,
            )
            print(
                f"  Step {iter_num:>6d} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"lr: {lr:.2e}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": model_config.to_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                ckpt_path = os.path.join(train_config.checkpoint_dir, "best.pt")
                torch.save(checkpoint, ckpt_path)
                print(f"  ✓ New best! Saved to {ckpt_path}")

        # ── Save periodic checkpoint ──────────────────────────────────────
        if iter_num > 0 and iter_num % train_config.save_interval == 0 and is_main_process():
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            ckpt_path = os.path.join(train_config.checkpoint_dir, "latest.pt")
            torch.save(checkpoint, ckpt_path)

        # ── Forward / Backward with gradient accumulation ─────────────────
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(train_config.gradient_accumulation_steps):
            X, Y = load_data(
                train_config.data_dir, "train",
                model_config.block_size, train_config.batch_size, device,
            )
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps

            scaler.scale(loss).backward()

        # Gradient clipping
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            if train_config.distributed:
                model.clip_grad_norm_(train_config.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Timing
        if iter_num % 100 == 0 and is_main_process():
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

    # ── Final save ────────────────────────────────────────────────────────
    if is_main_process():
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "iter_num": train_config.max_iters,
            "best_val_loss": best_val_loss,
        }
        ckpt_path = os.path.join(train_config.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")
        print(f"\nGenerate text with: python generate.py")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if train_config.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-4 architecture model")
    parser.add_argument("--preset", type=str, default="small",
                        choices=["small", "medium", "large", "xl", "gpt4", "deepseek"],
                        help="Model size preset (default: small)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory with train.bin and val.bin")
    parser.add_argument("--max-iters", type=int, default=5000,
                        help="Maximum training iterations")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable FSDP multi-GPU training (use with torchrun)")

    args = parser.parse_args()

    model_config = get_model_config(args.preset)
    train_config = TrainConfig(
        data_dir=args.data_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay_iters=args.max_iters,
        resume_from=args.resume,
        device=args.device,
        compile_model=args.compile,
        distributed=args.distributed,
    )

    train(model_config, train_config)
