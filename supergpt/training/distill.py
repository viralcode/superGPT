"""
Knowledge Distillation Pipeline
=================================
Transfer knowledge from a large teacher model to a smaller student model.
Supports logit distillation (KL divergence) and feature distillation.

This is how DeepSeek distills R1 reasoning capabilities into V3.

Usage:
  python -m supergpt.training.distill \
    --teacher checkpoints/large.pt \
    --student-preset small \
    --data-dir data \
    --temperature 4.0 \
    --alpha 0.7
"""

import os
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from supergpt.core.config import GPTConfig, TrainConfig, get_model_config
from supergpt.core.model import GPT


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Combined distillation loss.

    L = α × KL(teacher || student) × T² + (1-α) × CE(student, labels)

    Args:
        student_logits: (B, T, V) student model output
        teacher_logits: (B, T, V) teacher model output (detached)
        labels: (B, T) ground truth token IDs
        temperature: Softmax temperature (higher = softer distribution)
        alpha: Weight between distillation and hard label loss

    Returns:
        Combined loss scalar
    """
    V = student_logits.shape[-1]

    # Soft targets from teacher (temperature-scaled)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence (soft loss)
    kl_loss = F.kl_div(
        student_log_soft.view(-1, V),
        teacher_soft.view(-1, V),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Hard label loss (standard CE)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-1,
    )

    return alpha * kl_loss + (1 - alpha) * ce_loss


class FeatureDistiller(nn.Module):
    """Optional: align intermediate representations between teacher/student.

    Projects student hidden states to match teacher hidden states,
    providing richer gradient signal than logit-only distillation.
    """

    def __init__(self, student_dim: int, teacher_dim: int, n_layers: int = 4):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(student_dim, teacher_dim, bias=False)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        student_hiddens: list,
        teacher_hiddens: list,
    ) -> torch.Tensor:
        """Compute MSE loss between projected student and teacher hiddens."""
        loss = 0.0
        n = min(len(self.projectors), len(student_hiddens), len(teacher_hiddens))
        for i in range(n):
            projected = self.projectors[i](student_hiddens[i])
            loss += F.mse_loss(projected, teacher_hiddens[i].detach())
        return loss / max(n, 1)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a model from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config_dict = checkpoint["model_config"]
    model_config = GPTConfig(**model_config_dict)
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    return model, model_config


def distill(
    teacher_path: str,
    student_preset: str = "small",
    data_dir: str = "data",
    max_iters: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.7,
    device: str = "auto",
):
    """Run knowledge distillation from teacher to student."""
    import numpy as np

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"  Knowledge Distillation")
    print(f"  Teacher: {teacher_path}")
    print(f"  Student: {student_preset}")
    print(f"  Temperature: {temperature}, Alpha: {alpha}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load teacher
    print("\nLoading teacher model...")
    teacher, teacher_config = load_model_from_checkpoint(teacher_path, device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {teacher_params/1e6:.1f}M params")

    # Create student
    print("\nCreating student model...")
    student_config = get_model_config(student_preset)
    student_config.vocab_size = teacher_config.vocab_size
    student_config.block_size = teacher_config.block_size

    student = GPT(student_config)
    student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {student_params/1e6:.1f}M params "
          f"({teacher_params/student_params:.1f}× compression)")

    # Load data — auto-detect dtype from meta.pkl
    block_size = teacher_config.block_size
    data_dtype = np.uint16  # default
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("vocab_size", 0) > 65535 or meta.get("tokenizer_type") == "tiktoken":
            data_dtype = np.uint32
    train_data = np.memmap(os.path.join(data_dir, "train.bin"),
                           dtype=data_dtype, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"),
                         dtype=data_dtype, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64))
                        for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
                        for i in ix]).to(device)
        return x, y

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    print(f"\nDistilling for {max_iters} iterations...")
    t0 = time.time()
    best_val_loss = float("inf")

    for iter_num in range(max_iters):
        student.train()
        x, y = get_batch("train")

        with torch.no_grad():
            teacher_logits, _ = teacher(x)
        student_logits, _ = student(x)

        loss = distillation_loss(
            student_logits, teacher_logits, y,
            temperature=temperature, alpha=alpha,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if iter_num % 100 == 0:
            t1 = time.time()
            print(f"  iter {iter_num:>5d} | loss {loss.item():.4f} | {t1-t0:.1f}s")

        if iter_num % 500 == 0 and iter_num > 0:
            student.eval()
            val_losses = []
            for _ in range(50):
                x, y = get_batch("val")
                with torch.no_grad():
                    teacher_logits, _ = teacher(x)
                    student_logits, _ = student(x)
                    vloss = distillation_loss(
                        student_logits, teacher_logits, y,
                        temperature=temperature, alpha=alpha,
                    )
                    val_losses.append(vloss.item())

            val_loss = sum(val_losses) / len(val_losses)
            print(f"  [eval] val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "model": student.state_dict(),
                    "model_config": student_config.to_dict(),
                    "teacher_config": teacher_config.to_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }, "checkpoints/distilled.pt")
                print(f"  ✓ New best! Saved distilled model")

    print(f"\n{'='*60}")
    print(f"  Distillation complete! Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation")
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--student-preset", type=str, default="small")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    distill(**vars(args))
