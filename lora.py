"""
LoRA — Low-Rank Adaptation for superGPT
==========================================
Implements LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

LoRA injects trainable low-rank decomposition matrices (A, B) into existing
linear layers while keeping the original weights frozen. This reduces
trainable parameters by ~100x while maintaining model quality.

Usage:
    from lora import apply_lora, merge_lora, save_lora, load_lora

    # Apply LoRA to a pre-trained model
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    apply_lora(model, rank=16, alpha=32)

    # Train only LoRA params (all others are frozen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    # After training, merge LoRA weights into base model
    merge_lora(model)

    # Or save/load just the LoRA weights (~1MB vs 100MB+)
    save_lora(model, "lora_weights.pt")
    load_lora(model, "lora_weights.pt")

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with LoRA decomposition.

    Wraps an existing nn.Linear layer and adds low-rank A, B matrices.
    Output = original(x) + (x @ A) @ B * (alpha / rank)

    Only A and B are trainable; original weights are frozen.
    """
    def __init__(self, original: nn.Linear, rank: int, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # LoRA decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Original forward
        result = self.original(x)
        # LoRA addition: (x @ A @ B) * scaling
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return result + lora_out * self.scaling

    def merge(self):
        """Merge LoRA weights into original layer permanently."""
        with torch.no_grad():
            self.original.weight.add_(
                (self.lora_A @ self.lora_B).t() * self.scaling
            )
        return self.original


def apply_lora(model, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0,
               target_modules: list = None):
    """Apply LoRA to attention and output projections in a GPT model.

    Args:
        model: GPT model instance
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        dropout: dropout on LoRA layers
        target_modules: list of module name patterns to apply LoRA to.
                       Default: attention projections (q_proj, k_proj, v_proj, c_proj,
                       wq, wkv_a, wo)
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "c_proj",  # GQA
            "wq", "wq_a", "wq_b", "wkv_a", "wkv_b", "wo",  # MLA
            "w1", "w2", "w3",  # SwiGLU FFN
            "c_fc", "c_proj",  # Standard FFN
        ]

    # Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0
    for name, module in model.named_modules():
        for attr_name in dir(module):
            target = getattr(module, attr_name, None)
            if not isinstance(target, nn.Linear):
                continue
            if not any(t in attr_name for t in target_modules):
                continue

            # Replace with LoRA version
            lora_layer = LoRALinear(target, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, attr_name, lora_layer)
            lora_count += 1

    # Make sure LoRA params are trainable
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LoRA applied: {lora_count} layers | rank={rank} | alpha={alpha}")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params:.2%})")

    return model


def merge_lora(model):
    """Merge all LoRA weights into base model and remove LoRA layers.

    After merging, the model behaves identically but without LoRA overhead.
    This is useful for deployment.
    """
    merged_count = 0
    for name, module in model.named_modules():
        for attr_name in dir(module):
            target = getattr(module, attr_name, None)
            if isinstance(target, LoRALinear):
                merged_linear = target.merge()
                merged_linear.weight.requires_grad = True  # Unfreeze for future use
                if merged_linear.bias is not None:
                    merged_linear.bias.requires_grad = True
                setattr(module, attr_name, merged_linear)
                merged_count += 1

    print(f"Merged {merged_count} LoRA layers into base model")
    return model


def save_lora(model, path: str):
    """Save only LoRA weights (much smaller than full model).

    Saves LoRA A, B matrices and scaling information.
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                "lora_A": module.lora_A.data.clone(),
                "lora_B": module.lora_B.data.clone(),
                "rank": module.rank,
                "alpha": module.alpha,
            }

    torch.save(lora_state, path)
    n_params = sum(s["lora_A"].numel() + s["lora_B"].numel()
                   for s in lora_state.values())
    print(f"Saved {len(lora_state)} LoRA layers ({n_params:,} params) to {path}")


def load_lora(model, path: str, device: str = None):
    """Load LoRA weights from a checkpoint.

    The model must have LoRA already applied (via apply_lora).
    """
    lora_state = torch.load(path, map_location=device, weights_only=True)

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            state = lora_state[name]
            module.lora_A.data.copy_(state["lora_A"])
            module.lora_B.data.copy_(state["lora_B"])
            loaded += 1

    print(f"Loaded {loaded} LoRA layers from {path}")
    return model
