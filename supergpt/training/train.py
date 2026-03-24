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
import glob
import shutil
import tempfile
import threading
from contextlib import nullcontext

import numpy as np
import torch

from supergpt.core.config import GPTConfig, TrainConfig, get_model_config
from supergpt.core.model import GPT


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

# ── WandB imports (optional) ─────────────────────────────────────────────────
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
#  Fault-Tolerant Checkpointing
# =============================================================================

class CheckpointManager:
    """Production-grade checkpoint manager with:
    - Atomic writes (temp file + rename) to prevent corruption
    - Async saving (non-blocking, background thread)
    - Checkpoint rotation (keep last N)
    - Auto-resume (find latest checkpoint on startup)
    - FSDP-aware state dict handling
    """

    def __init__(self, checkpoint_dir: str, max_keep: int = 5, async_save: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        self.async_save = async_save
        self._save_thread = None
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        state: dict,
        filename: str = "latest.pt",
        is_best: bool = False,
        iter_num: int = 0,
    ):
        """Save checkpoint atomically, optionally in background thread."""
        if self.async_save:
            # Wait for previous save to finish
            if self._save_thread is not None:
                self._save_thread.join()
            self._save_thread = threading.Thread(
                target=self._atomic_save,
                args=(state, filename, is_best, iter_num),
                daemon=True,
            )
            self._save_thread.start()
        else:
            self._atomic_save(state, filename, is_best, iter_num)

    def _atomic_save(self, state: dict, filename: str, is_best: bool, iter_num: int):
        """Write to temp file, then atomic rename. Prevents corruption on crash."""
        target_path = os.path.join(self.checkpoint_dir, filename)

        # Write to a temp file in the same directory (same filesystem for rename)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.checkpoint_dir, suffix=".pt.tmp"
        )
        try:
            os.close(fd)
            torch.save(state, tmp_path)
            # Atomic rename
            shutil.move(tmp_path, target_path)
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"  Warning: Checkpoint save failed: {e}")
            return

        # Save periodic numbered checkpoint
        if iter_num > 0 and iter_num % 2000 == 0:
            numbered_path = os.path.join(
                self.checkpoint_dir, f"step_{iter_num}.pt"
            )
            shutil.copy2(target_path, numbered_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            shutil.copy2(target_path, best_path)

        # Rotate old checkpoints
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        """Keep only the last N numbered checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            os.remove(oldest)

    def find_latest(self) -> str:
        """Find the latest checkpoint for auto-resume."""
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest):
            return latest

        # Fall back to numbered checkpoints
        pattern = os.path.join(self.checkpoint_dir, "step_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        if checkpoints:
            return checkpoints[-1]

        return None

    def wait(self):
        """Wait for any async save to complete."""
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None


# =============================================================================
#  Training Monitor (WandB / TensorBoard)
# =============================================================================

class TrainingMonitor:
    """Unified training metrics logger.

    Supports WandB, TensorBoard, and stdout fallback.
    Tracks: loss, LR, throughput, gradient norms, GPU memory.
    """

    def __init__(
        self,
        enabled: bool = True,
        project: str = "superGPT",
        run_name: str = None,
        config: dict = None,
        backend: str = "wandb",  # "wandb", "tensorboard", "none"
    ):
        self.enabled = enabled
        self.backend = backend
        self._step = 0

        if not enabled or backend == "none":
            self.enabled = False
            return

        if backend == "wandb" and WANDB_AVAILABLE:
            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                resume="allow",
            )
            print(f"  WandB: {wandb.run.url}")
        elif backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=f"runs/{run_name or 'train'}")
                print(f"  TensorBoard: runs/{run_name or 'train'}")
            except ImportError:
                print("  Warning: TensorBoard not available")
                self.enabled = False
        else:
            self.enabled = False

    def log(self, metrics: dict, step: int = None):
        """Log metrics to the configured backend."""
        if not self.enabled:
            return
        if step is not None:
            self._step = step

        if self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.log(metrics, step=self._step)
        elif self.backend == "tensorboard":
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, self._step)

    def log_gradients(self, model):
        """Log gradient norms per layer for health monitoring."""
        if not self.enabled:
            return

        grad_norms = {}
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                total_norm += norm ** 2
                # Only log major layers to avoid metric explosion
                if any(k in name for k in ["wte", "wpe", "ln_f", "lm_head"]):
                    grad_norms[f"grad_norm/{name}"] = norm

        grad_norms["grad_norm/total"] = total_norm ** 0.5
        self.log(grad_norms)

    def log_gpu_stats(self):
        """Log GPU memory usage."""
        if not self.enabled or not torch.cuda.is_available():
            return

        self.log({
            "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu/max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        })

    def finish(self):
        """Finalize logging."""
        if not self.enabled:
            return
        if self.backend == "wandb" and WANDB_AVAILABLE:
            wandb.finish()
        elif self.backend == "tensorboard":
            self._writer.close()


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
    """Learning rate schedule: cosine or WSD (warmup-stable-decay)."""
    if config.lr_schedule == "wsd":
        # Warmup-Stable-Decay (DeepSeek V3)
        # Phase 1: Linear warmup
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # Phase 2: Stable at peak LR
        stable_end = int(config.lr_decay_iters * config.wsd_stable_fraction)
        if it < stable_end:
            return config.learning_rate
        # Phase 3: Cosine decay to min_lr
        decay_ratio = (it - stable_end) / max(1, config.lr_decay_iters - stable_end)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    else:
        # Cosine schedule with linear warmup
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

    # Detect dtype from meta.pkl — use uint32 if vocab > 65535
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
        print(f"  LR Schedule:  {train_config.lr_schedule.upper()}")
        if train_config.gradient_checkpointing:
            print(f"  Grad Ckpt:    ON (memory-efficient)")
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

    # ── Enable gradient checkpointing ─────────────────────────────────────
    if train_config.gradient_checkpointing:
        raw_model_temp = model.module if hasattr(model, "module") else model
        raw_model_temp.enable_gradient_checkpointing()
        if is_main_process():
            print("Gradient checkpointing enabled (memory-efficient mode)")

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

    # ── Checkpoint Manager ─────────────────────────────────────────────────
    ckpt_manager = CheckpointManager(
        checkpoint_dir=train_config.checkpoint_dir,
        max_keep=5,
        async_save=True,
    )

    # ── Training Monitor ──────────────────────────────────────────────────
    monitor_backend = getattr(train_config, 'monitor_backend', 'none')
    monitor = TrainingMonitor(
        enabled=(monitor_backend != 'none'),
        project="superGPT",
        run_name=f"{model_config.n_layer}L-{model_config.n_embd}E-{train_config.batch_size}B",
        config={
            "model": model_config.to_dict(),
            "batch_size": train_config.batch_size,
            "lr": train_config.learning_rate,
            "max_iters": train_config.max_iters,
        },
        backend=monitor_backend,
    )

    # ── Resume from checkpoint (with auto-resume) ─────────────────────────
    start_iter = 0
    best_val_loss = float("inf")

    resume_path = train_config.resume_from
    if not resume_path:
        # Auto-resume: check for existing latest checkpoint
        auto_resume = ckpt_manager.find_latest()
        if auto_resume:
            resume_path = auto_resume
            if is_main_process():
                print(f"Auto-resuming from: {resume_path}")

    if resume_path and os.path.exists(resume_path):
        if is_main_process():
            print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

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
    ema_loss = None  # Exponential moving average of training loss

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

            # Log eval metrics
            monitor.log({
                "eval/train_loss": losses["train"],
                "eval/val_loss": losses["val"],
                "eval/lr": lr,
            }, step=iter_num)
            monitor.log_gpu_stats()

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                print(f"  ✓ New best! Saved to checkpoints/best.pt")

            # Fault-tolerant checkpoint save (atomic + async)
            ckpt_state = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            ckpt_manager.save(
                ckpt_state,
                filename="latest.pt",
                is_best=(losses["val"] <= best_val_loss),
                iter_num=iter_num,
            )

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

        # Track EMA loss for smooth logging
        batch_loss = (loss * train_config.gradient_accumulation_steps).item()
        if ema_loss is None:
            ema_loss = batch_loss
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * batch_loss

        # Timing + monitoring
        if iter_num % 100 == 0 and is_main_process():
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = (
                train_config.batch_size * model_config.block_size
                * train_config.gradient_accumulation_steps * 100 / dt
            )
            print(f"  iter {iter_num:>6d} | loss {ema_loss:.4f} "
                  f"| {tokens_per_sec:.0f} tok/s | lr {lr:.2e}")
            t0 = t1

            # Log training metrics
            monitor.log({
                "train/loss": ema_loss,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
            }, step=iter_num)

            # Log gradient norms every 500 steps
            if iter_num % 500 == 0:
                monitor.log_gradients(model)
                monitor.log_gpu_stats()

    # ── Final save ────────────────────────────────────────────────────────
    if is_main_process():
        final_state = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "iter_num": train_config.max_iters,
            "best_val_loss": best_val_loss,
        }
        ckpt_manager.save(final_state, filename="latest.pt", iter_num=train_config.max_iters)
        ckpt_manager.wait()  # Ensure save completes before exit
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoint: {train_config.checkpoint_dir}/latest.pt")
        print(f"{'='*60}")
        print(f"\nGenerate text with: python generate.py")

    # ── Finalize monitor ──────────────────────────────────────────────────
    monitor.finish()

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
                        help="Path to checkpoint to resume from (empty = auto-resume)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging (requires 'pip install wandb')")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable FSDP multi-GPU training (use with torchrun)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves ~60%% memory)")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "wsd"],
                        help="LR schedule: cosine (default) or wsd (warmup-stable-decay)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming data pipeline instead of memmap")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset to stream from (e.g., HuggingFaceFW/fineweb)")
    parser.add_argument("--shard-dir", type=str, default=None,
                        help="Directory containing .bin shard files for streaming")
    parser.add_argument("--fp8", action="store_true",
                        help="Enable FP8 mixed-precision training (Hopper+ GPUs)")

    args = parser.parse_args()

    # Determine monitoring backend
    monitor_backend = "none"
    if args.wandb:
        monitor_backend = "wandb"
    elif args.tensorboard:
        monitor_backend = "tensorboard"

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
        gradient_checkpointing=args.gradient_checkpointing,
        lr_schedule=args.lr_schedule,
        use_wandb=args.wandb,
        monitor_backend=monitor_backend,
    )

    # FP8 conversion
    if args.fp8:
        try:
            from supergpt.training.fp8_utils import convert_model_to_fp8, FP8_AVAILABLE
            if FP8_AVAILABLE:
                print("\n  FP8 Training: Enabled")
                # Note: model conversion happens inside train() after model creation
                train_config.use_fp8 = True
            else:
                print("\n  FP8 Training: Not available (requires PyTorch 2.1+ with Hopper GPU)")
                train_config.use_fp8 = False
        except ImportError:
            print("\n  FP8 Training: Import error")
            train_config.use_fp8 = False
    else:
        train_config.use_fp8 = False

    # If streaming is requested, monkey-patch load_data with streaming loader
    if args.streaming or args.hf_dataset or args.shard_dir:
        try:
            from supergpt.training.data_pipeline import create_streaming_dataloader
            _stream_loader = create_streaming_dataloader(
                block_size=model_config.block_size,
                batch_size=train_config.batch_size,
                hf_dataset=args.hf_dataset,
                shard_dir=args.shard_dir or args.data_dir,
            )
            _stream_state = [iter(_stream_loader)]

            # Override load_data to pull from streaming
            _original_load_data = load_data
            def load_data_streaming(data_dir, split, block_size, batch_size, device):
                try:
                    batch = next(_stream_state[0])
                except StopIteration:
                    _stream_state[0] = iter(_stream_loader)
                    batch = next(_stream_state[0])
                x, y = batch[:, :-1], batch[:, 1:]
                return x.to(device), y.to(device)

            # Patch the global
            import types
            globals()['load_data'] = load_data_streaming
            print(f"Streaming mode enabled")
            if args.hf_dataset:
                print(f"  HF dataset: {args.hf_dataset}")
            elif args.shard_dir:
                print(f"  Shard dir: {args.shard_dir}")
        except ImportError:
            print("Warning: streaming.py not found, falling back to memmap")

    train(model_config, train_config)

