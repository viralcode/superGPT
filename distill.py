"""
Knowledge Distillation for microGPT
======================================
Train a smaller (student) model to mimic a larger (teacher) model.
The student learns to match the teacher's output probability distribution
using KL divergence, producing a smaller model with much of the teacher's
capability.

Supports two teacher modes:
  1. microGPT checkpoint (--teacher checkpoints/large.pt)
  2. HuggingFace model  (--hf-teacher Qwen/Qwen2.5-0.5B)

This is the same technique used by:
  - DeepSeek R1 Distill (671B -> 7B/14B/70B)
  - Qwen distillation series
  - DistilBERT, TinyLLaMA

How it works:
  1. Teacher model produces soft probability distributions (logits)
  2. Student model is trained to match those distributions (KL divergence)
  3. Optionally, student also trains on ground-truth tokens (cross-entropy)
  4. Temperature scaling softens distributions for better knowledge transfer

Usage:
    # Distill from a HuggingFace model (Qwen, LLaMA, Mistral, etc.):
    python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --student-preset small

    # Distill from a larger microGPT model:
    python distill.py --teacher checkpoints/large.pt --student-preset small

    # Custom settings:
    python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --student-preset medium \\
                      --temperature 3.0 --alpha 0.7

Requirements for HuggingFace mode:
    pip install transformers datasets

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
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
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, get_model_config
from model import GPT


# ==============================================================================
#  Teacher model wrappers
# ==============================================================================

class MicroGPTTeacher:
    """Teacher wrapper for a microGPT checkpoint."""

    def __init__(self, checkpoint_path, device):
        print(f"Loading microGPT teacher: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = GPTConfig(**ckpt["model_config"])
        self.model = GPT(self.config)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Teacher: {self.n_params/1e6:.1f}M params (frozen)")

    @torch.no_grad()
    def get_logits(self, input_ids):
        """Get full-sequence logits from teacher."""
        logits, _ = self.model(input_ids)
        # If only last token returned (inference mode), force training-mode forward
        if logits.size(1) == 1 and input_ids.size(1) > 1:
            x = self.model.transformer.wte(input_ids)
            if self.model.transformer.wpe is not None:
                pos = torch.arange(input_ids.size(1), device=input_ids.device)
                x = x + self.model.transformer.wpe(pos)
            x = self.model.transformer.drop(x)
            for block in self.model.transformer.h:
                x, _ = block(x)
            x = self.model.transformer.ln_f(x)
            logits = self.model.lm_head(x)
        return logits


class HuggingFaceTeacher:
    """Teacher wrapper for any HuggingFace causal LM (Qwen, LLaMA, Mistral, etc.).

    Usage:
        teacher = HuggingFaceTeacher("Qwen/Qwen2.5-0.5B", device="cuda")
        teacher = HuggingFaceTeacher("meta-llama/Llama-3.2-1B", device="cuda")
    """

    def __init__(self, model_name, device, dtype="auto"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Error: HuggingFace mode requires the 'transformers' package.")
            print("Install with: pip install transformers")
            sys.exit(1)

        print(f"Loading HuggingFace teacher: {model_name}")
        print("  (This may download the model on first use)")

        # Determine torch dtype
        if dtype == "auto":
            if device == "cuda" and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            self.model = self.model.to(device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.vocab_size = self.model.config.vocab_size
        self.block_size = getattr(self.model.config, "max_position_embeddings", 4096)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.model_name = model_name

        print(f"  Teacher: {model_name} ({self.n_params/1e6:.1f}M params, frozen)")
        print(f"  Vocab: {self.vocab_size} | Max ctx: {self.block_size}")

    @torch.no_grad()
    def get_logits(self, input_ids):
        """Get full-sequence logits from teacher."""
        outputs = self.model(input_ids)
        return outputs.logits

    def tokenize_text(self, text, max_length=512):
        """Tokenize text using the teacher's tokenizer."""
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False,
        )
        return tokens.input_ids


# ==============================================================================
#  Data loading
# ==============================================================================

def load_data_bin(data_dir, split, block_size, batch_size, device):
    """Load data from pre-tokenized binary files (microGPT format)."""
    data_path = os.path.join(data_dir, f"{split}.bin")
    meta_path = os.path.join(data_dir, "meta.pkl")

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        dtype = np.uint32 if meta.get("tokenizer_type") == "tiktoken" else np.uint16
    else:
        dtype = np.uint16

    data = np.memmap(data_path, dtype=dtype, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def load_data_hf(data_dir, split, block_size, batch_size, device, tokenizer):
    """Load and tokenize text data using the HuggingFace teacher's tokenizer.

    If pre-tokenized .bin files exist, loads from those.
    Otherwise, reads text files and tokenizes on-the-fly.
    """
    bin_path = os.path.join(data_dir, f"{split}_hf.bin")

    # Use cached tokenized data if available
    if os.path.exists(bin_path):
        data = np.memmap(bin_path, dtype=np.int32, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    # Fallback: read text and tokenize
    text_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(text_path):
        print(f"Error: No data found. Need either {bin_path} or {text_path}")
        sys.exit(1)

    print(f"Tokenizing {text_path} with HuggingFace tokenizer (first time only)...")
    with open(text_path, "r") as f:
        text = f.read()

    # Tokenize and cache
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    tokens_np = tokens.numpy().astype(np.int32)

    # Split 90/10
    n = len(tokens_np)
    split_idx = int(n * 0.9)

    train_data = tokens_np[:split_idx]
    val_data = tokens_np[split_idx:]

    # Save cached versions
    train_path = os.path.join(data_dir, "train_hf.bin")
    val_path = os.path.join(data_dir, "val_hf.bin")
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    print(f"  Cached: {train_path} ({len(train_data)} tokens), "
          f"{val_path} ({len(val_data)} tokens)")

    # Return the requested split
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ==============================================================================
#  Distillation loss
# ==============================================================================

def distillation_loss(student_logits, teacher_logits, targets,
                      temperature=2.0, alpha=0.5):
    """Compute combined distillation + cross-entropy loss.

    Args:
        student_logits: (B, T, V_student) student model output
        teacher_logits: (B, T, V_teacher) teacher model output (detached)
        targets: (B, T) ground truth token IDs
        temperature: softens distributions (higher = softer, more knowledge)
        alpha: balance between distillation and CE (1.0 = pure distillation)

    Returns:
        loss, kd_loss_value, ce_loss_value
    """
    # Handle vocab size mismatch (HF teacher may have different vocab)
    v_student = student_logits.size(-1)
    v_teacher = teacher_logits.size(-1)

    if v_student != v_teacher:
        # Use the smaller vocab for KD loss
        v_min = min(v_student, v_teacher)
        s_logits_kd = student_logits[..., :v_min]
        t_logits_kd = teacher_logits[..., :v_min]
    else:
        s_logits_kd = student_logits
        t_logits_kd = teacher_logits

    # KL Divergence on soft targets
    student_soft = F.log_softmax(s_logits_kd / temperature, dim=-1)
    teacher_soft = F.softmax(t_logits_kd / temperature, dim=-1)

    kd_loss = F.kl_div(
        student_soft.view(-1, student_soft.size(-1)),
        teacher_soft.view(-1, teacher_soft.size(-1)),
        reduction="batchmean",
    )
    # Scale by T^2 (Hinton et al.)
    kd_loss = kd_loss * (temperature ** 2)

    # Standard cross-entropy on hard targets
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )

    # Combined loss
    loss = alpha * kd_loss + (1 - alpha) * ce_loss

    return loss, kd_loss.item(), ce_loss.item()


# ==============================================================================
#  Evaluation
# ==============================================================================

@torch.no_grad()
def evaluate(student, teacher, data_loader_fn, block_size, batch_size,
             device, temperature, alpha, n_iters=50):
    """Evaluate distillation loss on validation set."""
    student.eval()
    losses = []
    for _ in range(n_iters):
        X, Y = data_loader_fn("val", block_size, batch_size, device)

        student_logits, _ = student(X)
        teacher_logits = teacher.get_logits(X)

        loss, _, _ = distillation_loss(
            student_logits, teacher_logits, Y,
            temperature=temperature, alpha=alpha,
        )
        losses.append(loss.item())
    student.train()
    return sum(losses) / len(losses)


# ==============================================================================
#  Main distillation loop
# ==============================================================================

def distill(args):
    """Main distillation training loop."""

    # -- Device --
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # -- Load teacher --
    is_hf = args.hf_teacher is not None
    if is_hf:
        teacher = HuggingFaceTeacher(args.hf_teacher, device)
    elif args.teacher:
        teacher = MicroGPTTeacher(args.teacher, device)
    else:
        print("Error: Must provide --teacher (microGPT) or --hf-teacher (HuggingFace)")
        sys.exit(1)

    # -- Create data loader function --
    if is_hf:
        # Use HuggingFace tokenizer for data
        def load_data(split, block_size, batch_size, dev):
            return load_data_hf(args.data, split, block_size, batch_size, dev,
                               teacher.tokenizer)
    else:
        def load_data(split, block_size, batch_size, dev):
            return load_data_bin(args.data, split, block_size, batch_size, dev)

    # -- Create or load student model --
    if args.student:
        print(f"Loading student model: {args.student}")
        student_ckpt = torch.load(args.student, map_location=device, weights_only=False)
        student_config = GPTConfig(**student_ckpt["model_config"])
        student = GPT(student_config)
        student.load_state_dict(student_ckpt["model"])
    else:
        print(f"Creating student model: preset={args.student_preset}")
        student_config = get_model_config(args.student_preset)

        # For HF teacher: match vocab size
        if is_hf:
            student_config.vocab_size = teacher.vocab_size

        student = GPT(student_config)

    student.to(device)
    student.train()

    student_params = sum(p.numel() for p in student.parameters())
    compression = teacher.n_params / student_params
    print(f"  Student: {student_params/1e6:.1f}M params (trainable)")
    print(f"  Compression: {compression:.1f}x")

    block_size = min(teacher.block_size, student_config.block_size)

    # -- Mixed precision --
    if "cuda" in device and torch.cuda.is_bf16_supported():
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif "cuda" in device:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = nullcontext()

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()

    # -- Training --
    teacher_name = args.hf_teacher if is_hf else args.teacher
    print(f"\n{'='*60}")
    print(f"  Knowledge Distillation")
    print(f"{'='*60}")
    print(f"  Teacher:     {teacher_name} ({teacher.n_params/1e6:.1f}M)")
    print(f"  Student:     {student_params/1e6:.1f}M params ({compression:.1f}x smaller)")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha:       {args.alpha} (KD={args.alpha:.0%}, CE={1-args.alpha:.0%})")
    print(f"  LR:          {args.lr}")
    print(f"  Max iters:   {args.max_iters}")
    print(f"  Block size:  {block_size}")
    print(f"  Device:      {device}")
    if is_hf:
        print(f"  Mode:        HuggingFace (using teacher's tokenizer)")
    print(f"{'='*60}\n")

    t0 = time.time()
    best_val_loss = float("inf")
    ema_loss = None
    ema_kd = None
    ema_ce = None

    for iter_num in range(args.max_iters):
        # Cosine LR with warmup
        if iter_num < args.warmup_iters:
            lr = args.lr * iter_num / max(1, args.warmup_iters)
        else:
            progress = (iter_num - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # -- Evaluate --
        if iter_num % args.eval_interval == 0:
            val_loss = evaluate(
                student, teacher, load_data, block_size,
                args.batch_size, device, args.temperature, args.alpha,
            )
            t1 = time.time()
            print(f"  Step {iter_num:>5d} | val loss: {val_loss:.4f} | "
                  f"lr: {lr:.2e} | {t1-t0:.1f}s")
            t0 = t1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt = {
                    "model": student.state_dict(),
                    "model_config": student_config.to_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "distillation": {
                        "teacher": teacher_name,
                        "teacher_type": "huggingface" if is_hf else "microgpt",
                        "temperature": args.temperature,
                        "alpha": args.alpha,
                        "compression": compression,
                    },
                }
                save_path = os.path.join(args.output_dir, "distilled_best.pt")
                torch.save(ckpt, save_path)
                print(f"    Saved best: {save_path}")

            student.train()

        # -- Train step --
        X, Y = load_data("train", block_size, args.batch_size, device)

        with ctx:
            with torch.no_grad():
                teacher_logits = teacher.get_logits(X)

            student_logits, _ = student(X)

            loss, kd_l, ce_l = distillation_loss(
                student_logits, teacher_logits, Y,
                temperature=args.temperature, alpha=args.alpha,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # EMA tracking
        batch_loss = loss.item()
        if ema_loss is None:
            ema_loss = batch_loss
            ema_kd = kd_l
            ema_ce = ce_l
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * batch_loss
            ema_kd = 0.95 * ema_kd + 0.05 * kd_l
            ema_ce = 0.95 * ema_ce + 0.05 * ce_l

        if iter_num % 100 == 0 and iter_num > 0:
            print(f"    iter {iter_num:>5d} | loss {ema_loss:.4f} "
                  f"(KD={ema_kd:.4f}, CE={ema_ce:.4f}) | lr {lr:.2e}")

    # -- Final save --
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "distilled_final.pt")
    ckpt = {
        "model": student.state_dict(),
        "model_config": student_config.to_dict(),
        "iter_num": args.max_iters,
        "best_val_loss": best_val_loss,
        "distillation": {
            "teacher": teacher_name,
            "teacher_type": "huggingface" if is_hf else "microgpt",
            "temperature": args.temperature,
            "alpha": args.alpha,
            "compression": compression,
        },
    }
    torch.save(ckpt, final_path)

    print(f"\n{'='*60}")
    print(f"  Distillation Complete")
    print(f"{'='*60}")
    print(f"  Teacher:         {teacher_name} ({teacher.n_params/1e6:.1f}M)")
    print(f"  Student:         {student_params/1e6:.1f}M ({compression:.1f}x smaller)")
    print(f"  Best val loss:   {best_val_loss:.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'='*60}")
    print(f"\nGenerate with: python generate.py --checkpoint {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge distillation for microGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distill from Qwen (HuggingFace):
  python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --student-preset small --data data/

  # Distill from LLaMA:
  python distill.py --hf-teacher meta-llama/Llama-3.2-1B --student-preset medium

  # Distill from a microGPT checkpoint:
  python distill.py --teacher checkpoints/large.pt --student-preset small --data data/

  # Custom temperature and alpha:
  python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --temperature 3.0 --alpha 0.7
        """,
    )

    # Teacher (one of these two is required)
    parser.add_argument("--teacher", type=str, default=None,
                        help="Path to microGPT teacher checkpoint")
    parser.add_argument("--hf-teacher", type=str, default=None,
                        help="HuggingFace model name (e.g. Qwen/Qwen2.5-0.5B, "
                             "meta-llama/Llama-3.2-1B)")

    # Student
    parser.add_argument("--student", type=str, default=None,
                        help="Path to student checkpoint (to resume)")
    parser.add_argument("--student-preset", type=str, default="small",
                        help="Preset for new student model (default: small)")

    # Distillation
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature (default: 2.0, higher=softer)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD vs CE balance: 1.0=pure KD, 0.0=pure CE (default: 0.5)")

    # Training
    parser.add_argument("--data", type=str, default="data",
                        help="Directory with input.txt or train.bin/val.bin")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--max-iters", type=int, default=5000,
                        help="Max training iterations (default: 5000)")
    parser.add_argument("--warmup-iters", type=int, default=200,
                        help="Warmup iterations (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--eval-interval", type=int, default=250,
                        help="Evaluation interval (default: 250)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()

    if not args.teacher and not args.hf_teacher:
        print("Error: Must provide --teacher (microGPT) or --hf-teacher (HuggingFace)")
        print("Examples:")
        print("  python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --student-preset small")
        print("  python distill.py --teacher checkpoints/large.pt --student-preset small")
        sys.exit(1)

    distill(args)
