"""
GPT-4 / DeepSeek V3 Architecture Configuration
=================================================
All model and training hyperparameters in one place.
Supports: GQA, MLA, MoE (DeepSeekMoE), KV-cache, RoPE, SwiGLU, FSDP, MTP.
"""

import dataclasses
from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    # Model architecture
    block_size: int = 256        # Maximum sequence length (context window)
    vocab_size: int = 65536      # Vocabulary size (will be set by tokenizer)
    n_layer: int = 6             # Number of transformer decoder blocks
    n_head: int = 6              # Number of attention heads (query heads)
    n_kv_head: int = 6           # Number of KV heads (GQA: fewer than n_head)
    n_embd: int = 384            # Embedding dimension
    dropout: float = 0.1         # Dropout rate
    bias: bool = False           # Use bias in linear layers (False = modern style)
    use_rope: bool = True        # Use Rotary Positional Embeddings
    use_swiglu: bool = True      # Use SwiGLU activation

    # ── Sliding Window Attention (Mistral / Gemma 2) ─────────────────────
    sliding_window: int = 0      # 0 = full attention, >0 = window size
    alternating_layers: bool = False  # Even=global, Odd=local (Gemma 2)
    attn_logit_cap: float = 0.0  # 0 = disabled, >0 = soft cap (Gemma 2)

    # ── Context Extension (YaRN / NTK) ───────────────────────────────────
    rope_scaling_type: str = "none"   # "none", "linear", "yarn"
    rope_scaling_factor: float = 1.0  # e.g. 4.0 = extends 4K ctx to 16K

    # ── Multi-head Latent Attention (MLA) — DeepSeek V3 ──────────────────
    use_mla: bool = False        # Enable MLA (replaces GQA/MHA attention)
    kv_lora_rank: int = 512      # Latent dimension for KV compression
    q_lora_rank: int = 0         # Query compression rank (0 = no compression)
    qk_nope_head_dim: int = 128  # Content attention dim per head
    qk_rope_head_dim: int = 64   # RoPE attention dim per head
    v_head_dim: int = 128        # Value dim per head

    # ── Mixture of Experts ───────────────────────────────────────────────
    use_moe: bool = False        # Enable Mixture of Experts
    n_experts: int = 8           # Total number of routed expert FFNs
    n_experts_active: int = 2    # Top-k experts activated per token
    n_shared_experts: int = 0    # Always-on shared experts (DeepSeek style)
    n_dense_layers: int = 0      # First N layers use dense FFN, rest use MoE

    # MoE routing strategy
    score_func: str = "softmax"  # "softmax" or "sigmoid"
    route_scale: float = 1.0     # Scaling factor for routing weights
    aux_loss_free: bool = False  # Use bias-based routing (DeepSeek V3)
    bias_update_speed: float = 0.001  # γ for bias adjustment
    moe_aux_loss_weight: float = 0.01  # Aux loss weight (when not aux_loss_free)

    # ── Multi-Token Prediction (MTP) — DeepSeek V3 ──────────────────────
    n_predict_tokens: int = 1    # Tokens to predict (1=standard, >1=MTP)
    mtp_loss_weight: float = 1.0 # Weight for MTP losses relative to main loss

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_rank: int = 0           # 0 = disabled, >0 = LoRA rank
    lora_alpha: float = 16.0     # LoRA scaling factor
    lora_dropout: float = 0.0    # Dropout on LoRA layers

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class TrainConfig:
    """Configuration for training."""
    # Data
    data_dir: str = "data"

    # Training
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 3e-5

    # System
    device: str = "auto"
    dtype: str = "auto"
    compile_model: bool = False
    gradient_checkpointing: bool = False

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "cosine" or "wsd" (warmup-stable-decay)
    wsd_stable_fraction: float = 0.8  # Fraction of training at stable LR (WSD only)

    # Distributed training (FSDP)
    distributed: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000
    resume_from: str = ""


# ── Pre-built model size presets ──────────────────────────────────────────────

PRESETS = {
    "small": GPTConfig(
        n_layer=6, n_head=6, n_kv_head=6, n_embd=384,
        block_size=256, dropout=0.1,
    ),
    "medium": GPTConfig(
        n_layer=12, n_head=12, n_kv_head=4, n_embd=768,
        block_size=512, dropout=0.1,
    ),
    "large": GPTConfig(
        n_layer=24, n_head=16, n_kv_head=4, n_embd=1024,
        block_size=2048, dropout=0.1,
    ),
    "xl": GPTConfig(
        n_layer=24, n_head=16, n_kv_head=8, n_embd=2048,
        block_size=4096, dropout=0.1,
    ),
    "gpt4": GPTConfig(
        n_layer=32, n_head=32, n_kv_head=8, n_embd=4096,
        block_size=8192, dropout=0.1,
        use_moe=True, n_experts=8, n_experts_active=2,
    ),
    "deepseek": GPTConfig(
        n_layer=27, n_head=16, n_embd=2048,
        block_size=4096, dropout=0.0, bias=False,
        # MLA
        use_mla=True, kv_lora_rank=512, q_lora_rank=1536,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        # DeepSeekMoE
        use_moe=True, n_experts=64, n_experts_active=6,
        n_shared_experts=2, n_dense_layers=1,
        score_func="sigmoid", route_scale=1.0,
        aux_loss_free=True, bias_update_speed=0.001,
        # MTP
        n_predict_tokens=3,
    ),
    "mistral": GPTConfig(
        n_layer=32, n_head=32, n_kv_head=8, n_embd=4096,
        block_size=8192, dropout=0.0, bias=False,
        sliding_window=4096,
    ),
    "gemma2": GPTConfig(
        n_layer=28, n_head=16, n_kv_head=4, n_embd=2304,
        block_size=8192, dropout=0.0, bias=False,
        sliding_window=4096, alternating_layers=True,
        attn_logit_cap=50.0,
    ),
}


def get_model_config(preset: str = "small", **overrides) -> GPTConfig:
    """Get a model config from a preset, with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. "
                         f"Choose from: {list(PRESETS.keys())}")
    config = PRESETS[preset]
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config
