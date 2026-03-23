"""
GPT-4 / DeepSeek V3 Architecture Transformer Model
=====================================================
A decoder-only transformer implementing innovations from GPT-4 and DeepSeek V3:

Attention:
  - Grouped Query Attention (GQA) — fewer KV heads for efficiency
  - Multi-head Latent Attention (MLA) — low-rank KV compression, ~10x cache reduction
  - Decoupled RoPE — separate positional dims from content dims
  - KV-Cache — incremental decoding for fast generation

Feed-Forward / MoE:
  - SwiGLU activation (GPT-4 / LLaMA)
  - Mixture of Experts with shared experts (DeepSeek V3)
  - Auxiliary-loss-free load balancing via dynamic bias routing

Training:
  - Multi-Token Prediction (MTP) — denser training signal
  - RMSNorm pre-norm architecture

References:
  - DeepSeek-V3 Technical Report (2024)
  - GPT-4, LLaMA 2/3, Mistral, nanoGPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


# ── Rotary Positional Embeddings (RoPE) ──────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings with arbitrary dim support."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        end = offset + seq_len
        if end > self.cos_cached.shape[0]:
            self._build_cache(end * 2)
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply RoPE to a single tensor (q or k)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_pair(q, k, cos, sin):
    """Apply RoPE to both q and k."""
    return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)


# ── SwiGLU Feed-Forward ─────────────────────────────────────────────────────

class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN used in GPT-4, LLaMA, DeepSeek."""
    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * dim / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class StandardFeedForward(nn.Module):
    """Standard GELU FFN (GPT-2/3 style)."""
    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.c_fc = nn.Linear(dim, hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


def make_ffn(config: GPTConfig, moe_inter_dim: int = None):
    """Create a feed-forward network based on config."""
    dim = config.n_embd
    hidden = moe_inter_dim
    if config.use_swiglu:
        return SwiGLUFeedForward(dim, hidden, config.bias, config.dropout)
    else:
        return StandardFeedForward(dim, hidden, config.bias, config.dropout)


# ══════════════════════════════════════════════════════════════════════════════
#  MIXTURE OF EXPERTS (DeepSeek V3 Style)
# ══════════════════════════════════════════════════════════════════════════════

class ExpertGate(nn.Module):
    """Gating/Router for DeepSeek-style MoE.

    Supports:
    - Softmax or sigmoid scoring
    - Auxiliary-loss-free routing via dynamic bias
    - Standard auxiliary loss routing
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.dim = config.n_embd
        self.topk = config.n_experts_active
        self.n_experts = config.n_experts
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.aux_loss_free = config.aux_loss_free
        self.bias_update_speed = config.bias_update_speed
        self.aux_loss_weight = config.moe_aux_loss_weight

        self.weight = nn.Parameter(torch.empty(config.n_experts, config.n_embd))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Dynamic bias for aux-loss-free routing (DeepSeek V3)
        if self.aux_loss_free:
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.n_experts, dtype=torch.float32)
            )

        self._aux_loss = None

    @property
    def aux_loss(self):
        return self._aux_loss

    def forward(self, x):
        """
        Args:
            x: (N, dim) flattened token embeddings
        Returns:
            weights: (N, topk) — routing weights for selected experts
            indices: (N, topk) — indices of selected experts
        """
        # Router scores
        scores = F.linear(x.float(), self.weight.float())

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:  # sigmoid
            scores = scores.sigmoid()

        original_scores = scores

        # Aux-loss-free: add bias for routing decision only
        if self.aux_loss_free:
            routing_scores = scores + self.expert_bias.unsqueeze(0)
        else:
            routing_scores = scores

        # Select top-k experts
        topk_scores, indices = torch.topk(routing_scores, self.topk, dim=-1)

        # Get actual weights from original scores (without bias)
        weights = original_scores.gather(1, indices)

        # Normalize for sigmoid scoring
        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        weights = weights * self.route_scale

        # Update bias during training (aux-loss-free)
        if self.training and self.aux_loss_free:
            with torch.no_grad():
                # Count tokens per expert
                counts = torch.zeros(self.n_experts, device=x.device)
                for k in range(self.topk):
                    counts.scatter_add_(
                        0, indices[:, k],
                        torch.ones(indices.shape[0], device=x.device)
                    )
                avg_count = counts.float().mean()
                # Decrease bias for overloaded, increase for underloaded
                self.expert_bias += self.bias_update_speed * (avg_count - counts)
            self._aux_loss = torch.tensor(0.0, device=x.device)

        # Standard aux loss (when not aux-loss-free)
        elif self.training and not self.aux_loss_free:
            expert_mask = F.one_hot(indices, self.n_experts).float()
            tokens_per_expert = expert_mask.sum(dim=(0, 1))
            N = x.shape[0]
            fraction_tokens = tokens_per_expert / (N * self.topk)
            fraction_probs = original_scores.mean(dim=0)
            self._aux_loss = self.aux_loss_weight * self.n_experts * (
                fraction_tokens * fraction_probs
            ).sum()
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device)

        return weights.type_as(x), indices


class MoELayer(nn.Module):
    """Mixture of Experts with shared experts (DeepSeek V3 style).

    - Routed experts: top-k selected per token via gating
    - Shared experts: always-on, process every token
    - Supports aux-loss-free or standard aux-loss routing
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_experts_active = config.n_experts_active
        self.n_shared_experts = config.n_shared_experts

        # Router
        self.gate = ExpertGate(config)

        # Routed experts
        self.experts = nn.ModuleList([make_ffn(config) for _ in range(config.n_experts)])

        # Shared experts (always active, DeepSeek V3 innovation)
        if config.n_shared_experts > 0:
            # Shared experts use combined hidden dim
            shared_hidden = None  # Use default sizing
            self.shared_experts = make_ffn(config)
        else:
            self.shared_experts = None

    @property
    def aux_loss(self):
        return self.gate.aux_loss

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # Route through selected experts
        weights, indices = self.gate(x_flat)

        y = torch.zeros_like(x_flat)
        for i in range(self.n_experts_active):
            expert_idx = indices[:, i]
            expert_weight = weights[:, i]
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    y[mask] += expert_weight[mask].unsqueeze(-1) * expert_out

        # Add shared expert output (always active)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x_flat)

        return y.view(B, T, C)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-HEAD LATENT ATTENTION (MLA) — DeepSeek V3
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (MLA) from DeepSeek V3.

    Compresses KV into a low-rank latent vector, reducing KV-cache by ~10x.
    Uses decoupled RoPE: separate positional key dims from content key dims.

    KV Encoding:
        [c_KV, k_rope] = W_KV_A @ h          # single projection
        c_KV = RMSNorm(c_KV)
        [k_nope, v] = W_KV_B @ c_KV          # up-project to all heads
        k_rope = RoPE(k_rope)                 # position info
        k = concat(k_nope, k_rope)            # final key

    KV-Cache stores only (c_KV, k_rope) — NOT full K,V per head.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_head
        self.dim = config.n_embd
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.dropout_rate = config.dropout

        # Query projection (optionally compressed)
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim,
                                  bias=False)

        # KV projection: single down-projection → latent + rope key
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim,
                               bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)

        # KV up-projection from latent to all heads
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        # RoPE for the decoupled positional dims
        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim,
                                          max_seq_len=config.block_size)

        # Attention scale
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None):
        """
        Args:
            x: (B, T, dim)
            past_kv: optional (past_kv_latent, past_k_rope) for KV-cache

        Returns:
            output: (B, T, dim)
            present_kv: (kv_latent, k_rope) for caching
        """
        B, T, _ = x.size()

        # ── Query ────────────────────────────────────────────────────────
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(B, T, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                                 dim=-1)

        # ── Key/Value (compressed) ───────────────────────────────────────
        kv_combined = self.wkv_a(x)
        kv_latent, k_rope_raw = kv_combined.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Normalize latent
        kv_latent = self.kv_norm(kv_latent)

        # ── RoPE on decoupled positional dims ────────────────────────────
        offset = past_kv[0].shape[1] if past_kv is not None else 0
        cos, sin = self.rotary_emb(T, offset=offset)

        # Apply RoPE to query rope dims: (B, T, n_heads, rope_dim) → transpose
        q_rope = q_rope.transpose(1, 2)  # (B, n_heads, T, rope_dim)
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)

        # Apply RoPE to key rope dims: shared across heads (B, T, 1, rope_dim)
        k_rope = k_rope_raw.unsqueeze(2)  # (B, T, 1, rope_dim)
        k_rope = k_rope.transpose(1, 2)   # (B, 1, T, rope_dim)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)

        # ── KV-Cache ─────────────────────────────────────────────────────
        # Store compressed representation: (kv_latent, k_rope)
        # This is ~10x smaller than storing full K,V per head
        if past_kv is not None:
            past_latent, past_rope = past_kv
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_rope, k_rope], dim=2)

        present_kv = (kv_latent, k_rope)

        # ── Up-project KV from latent ────────────────────────────────────
        kv_full = self.wkv_b(kv_latent)  # (B, total_T, n_heads * (nope + v))
        kv_full = kv_full.view(B, -1, self.n_heads,
                               self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_full.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ── Assemble full K ──────────────────────────────────────────────
        # k_nope: (B, total_T, n_heads, nope_dim)
        # k_rope: (B, 1, total_T, rope_dim) → expand to (B, n_heads, total_T, rope_dim)
        k_nope = k_nope.transpose(1, 2)  # (B, n_heads, total_T, nope_dim)
        k_rope_expanded = k_rope.expand(-1, self.n_heads, -1, -1)
        k = torch.cat([k_nope, k_rope_expanded], dim=-1)  # (B, n_heads, total_T, qk_dim)

        # ── Assemble full Q ──────────────────────────────────────────────
        q_nope = q_nope.transpose(1, 2)  # (B, n_heads, T, nope_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, n_heads, T, qk_dim)

        # v: (B, total_T, n_heads, v_dim) → (B, n_heads, total_T, v_dim)
        v = v.transpose(1, 2)

        # ── Attention ────────────────────────────────────────────────────
        total_T = k.shape[2]
        att = (q @ k.transpose(-2, -1)) * self.softmax_scale

        # Causal mask
        if past_kv is None and T > 1:
            causal = torch.triu(
                torch.full((T, total_T), float("-inf"), device=x.device), diagonal=1
            )
            att = att + causal.unsqueeze(0).unsqueeze(0)
        # When using cache with T=1, no masking needed

        att = att.softmax(dim=-1, dtype=torch.float32).type_as(x)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_heads, T, v_dim)

        # ── Output ───────────────────────────────────────────────────────
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_head_dim)
        y = self.resid_dropout(self.wo(y))
        return y, present_kv


# ══════════════════════════════════════════════════════════════════════════════
#  GROUPED QUERY ATTENTION (GQA) — GPT-4 / LLaMA style
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-Head Attention with GQA and KV-cache support."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.dropout_rate = config.dropout
        self.use_rope = config.use_rope

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim,
                                bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim,
                                bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim,
                                bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim,
                                              max_seq_len=config.block_size)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, D).reshape(
            B, self.n_head, T, D
        )

    def forward(self, x, past_kv=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            cos, sin = self.rotary_emb(T, offset=offset)
            q, k = apply_rotary_pos_emb_pair(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)
        k_expanded = self._repeat_kv(k)
        v_expanded = self._repeat_kv(v)

        if hasattr(F, "scaled_dot_product_attention") and past_kv is None:
            y = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded, attn_mask=None,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=True,
            )
        else:
            total_len = k_expanded.shape[2]
            att = (q @ k_expanded.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if past_kv is None:
                att = att.masked_fill(
                    self.causal_mask[:, :, :T, :T] == 0, float("-inf")
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_expanded

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Transformer block supporting MLA or GQA attention, and dense or MoE FFN."""
    def __init__(self, layer_id: int, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)

        # Attention: MLA or GQA
        if config.use_mla:
            self.attn = MultiHeadLatentAttention(config)
        else:
            self.attn = CausalSelfAttention(config)

        # FFN: Dense for first N layers, MoE for the rest (DeepSeek pattern)
        self.use_moe = False
        if config.use_moe and layer_id >= config.n_dense_layers:
            self.ffn = MoELayer(config)
            self.use_moe = True
        elif config.use_swiglu:
            self.ffn = SwiGLUFeedForward(config.n_embd, bias=config.bias,
                                         dropout=config.dropout)
        else:
            self.ffn = StandardFeedForward(config.n_embd, bias=config.bias,
                                           dropout=config.dropout)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, present_kv


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-TOKEN PREDICTION (MTP) — DeepSeek V3
# ══════════════════════════════════════════════════════════════════════════════

class MTPModule(nn.Module):
    """Multi-Token Prediction module for one additional prediction depth.

    Each MTP module:
    1. Projects the main model's hidden state
    2. Combines with the embedding of the target token at that depth
    3. Passes through a small transformer layer
    4. Predicts the next token using the shared LM head

    MTP modules are ONLY used during training — discarded at inference.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.norm = RMSNorm(config.n_embd)

        # Small transformer layer for this depth
        self.attn_norm = RMSNorm(config.n_embd)
        if config.use_mla:
            self.attn = MultiHeadLatentAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        if config.use_swiglu:
            self.ffn = SwiGLUFeedForward(config.n_embd, bias=config.bias,
                                         dropout=config.dropout)
        else:
            self.ffn = StandardFeedForward(config.n_embd, bias=config.bias,
                                           dropout=config.dropout)

    def forward(self, h, target_emb):
        """
        Args:
            h: (B, T, dim) — hidden states from main model
            target_emb: (B, T, dim) — embeddings of the target token at this depth
        Returns:
            h_out: (B, T, dim) — hidden states for predicting this depth's token
        """
        # Combine projected hidden state with target embedding
        combined = self.norm(self.proj(h) + target_emb)

        # Small transformer layer
        attn_out, _ = self.attn(self.attn_norm(combined))
        combined = combined + attn_out
        combined = combined + self.ffn(self.ffn_norm(combined))

        return combined


# ══════════════════════════════════════════════════════════════════════════════
#  GPT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):
    """GPT-4 / DeepSeek V3 Architecture Language Model.

    Features: GQA or MLA attention, MoE with shared experts,
    KV-cache, RoPE, SwiGLU, RMSNorm, Multi-Token Prediction.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.aux_losses = []

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd) if (
                not config.use_rope and not config.use_mla
            ) else None,
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                TransformerBlock(i, config) for i in range(config.n_layer)
            ]),
            ln_f=RMSNorm(config.n_embd),
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Multi-Token Prediction modules (DeepSeek V3)
        self.mtp_modules = None
        if config.n_predict_tokens > 1:
            self.mtp_modules = nn.ModuleList([
                MTPModule(config) for _ in range(config.n_predict_tokens - 1)
            ])

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w2.weight") or \
               pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        # Report
        n_params = self.get_num_params()
        features = []
        if config.use_mla:
            features.append(f"MLA(kv_rank={config.kv_lora_rank})")
        elif config.n_kv_head < config.n_head:
            features.append(f"GQA({config.n_head}Q/{config.n_kv_head}KV)")
        if config.use_moe:
            active = self.get_num_params_active()
            features.append(f"MoE({config.n_experts_active}/{config.n_experts}"
                          f"+{config.n_shared_experts}shared)")
            if config.aux_loss_free:
                features.append("aux-free")
            print(f"GPT initialized: {n_params/1e6:.1f}M total, "
                  f"{active/1e6:.1f}M active | {' | '.join(features)}")
        else:
            print(f"GPT initialized: {n_params/1e6:.1f}M params | "
                  f"{' | '.join(features) if features else 'MHA'}")
        if config.n_predict_tokens > 1:
            print(f"  MTP: predicting {config.n_predict_tokens} tokens")

    def get_num_params(self, non_embedding: bool = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_params_active(self, non_embedding: bool = True):
        n_params = self.get_num_params(non_embedding)
        if not self.config.use_moe:
            return n_params
        for block in self.transformer.h:
            if block.use_moe:
                expert_params = sum(
                    p.numel() for p in block.ffn.experts[0].parameters()
                )
                total_expert_params = expert_params * self.config.n_experts
                active_expert_params = expert_params * self.config.n_experts_active
                n_params -= (total_expert_params - active_expert_params)
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass with optional Multi-Token Prediction.

        Returns:
            logits: (B, T, vocab_size) or (B, 1, vocab_size)
            loss: combined CE + MoE aux + MTP losses
        """
        B, T = idx.size()
        assert T <= self.config.block_size

        # Embeddings
        tok_emb = self.transformer.wte(idx)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        # Collect aux losses
        self.aux_losses = []

        # Transformer blocks
        for block in self.transformer.h:
            x, _ = block(x)
            if block.use_moe and hasattr(block.ffn, 'aux_loss'):
                aux = block.ffn.aux_loss
                if aux is not None:
                    self.aux_losses.append(aux)

        # Final norm
        h = self.transformer.ln_f(x)

        if targets is not None:
            # ── Training: compute loss ────────────────────────────────────
            logits = self.lm_head(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            # MoE aux loss
            if self.aux_losses:
                loss = loss + sum(self.aux_losses)

            # Multi-Token Prediction loss
            if self.mtp_modules is not None and T > 1:
                mtp_loss_total = 0.0
                n_mtp = 0
                h_prev = h
                for d, mtp_mod in enumerate(self.mtp_modules):
                    depth = d + 1
                    if T - depth < 1:
                        break
                    # Target tokens at this depth
                    mtp_targets = targets[:, depth:]  # shifted by depth
                    # Target embeddings (for combining with hidden state)
                    target_token_ids = idx[:, depth:]  # tokens shifted by depth
                    target_emb = self.transformer.wte(target_token_ids)

                    # Truncate hidden states to match
                    h_trunc = h_prev[:, :T - depth, :]

                    # MTP forward
                    h_mtp = mtp_mod(h_trunc, target_emb[:, :T - depth, :])
                    mtp_logits = self.lm_head(h_mtp)

                    # MTP loss
                    mtp_loss = F.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.size(-1)),
                        mtp_targets[:, :T - depth].contiguous().view(-1),
                        ignore_index=-1,
                    )
                    mtp_loss_total += mtp_loss
                    n_mtp += 1

                if n_mtp > 0:
                    loss = loss + self.config.mtp_loss_weight * (mtp_loss_total / n_mtp)
        else:
            # ── Inference: only last position ─────────────────────────────
            logits = self.lm_head(h[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 use_cache=True):
        """Generate tokens with KV-cache (supports both MLA and GQA cache)."""
        past_kvs = None

        if use_cache:
            # First pass: build cache from prompt
            B, T = idx.size()
            tok_emb = self.transformer.wte(idx)
            if self.transformer.wpe is not None:
                pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
                x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
            else:
                x = self.transformer.drop(tok_emb)

            past_kvs = []
            for block in self.transformer.h:
                x, present_kv = block(x)
                past_kvs.append(present_kv)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1], :])[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Subsequent tokens: incremental with cache
            for _ in range(max_new_tokens - 1):
                tok_emb = self.transformer.wte(idx_next)
                if self.transformer.wpe is not None:
                    pos = torch.tensor([idx.size(1) - 1], dtype=torch.long,
                                       device=idx.device)
                    x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
                else:
                    x = self.transformer.drop(tok_emb)

                new_past_kvs = []
                for i, block in enumerate(self.transformer.h):
                    x, present_kv = block(x, past_kv=past_kvs[i])
                    new_past_kvs.append(present_kv)
                past_kvs = new_past_kvs

                x = self.transformer.ln_f(x)
                logits = self.lm_head(x[:, -1, :]) / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

                if idx.size(1) >= self.config.block_size:
                    break
        else:
            # Non-cached generation
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size \
                    else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

        return idx
