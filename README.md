<p align="center">
  <h1 align="center">🔥 NanoForge</h1>
  <p align="center"><strong>Train your own LLM from scratch — with GPT-4 & DeepSeek V3 architecture</strong></p>
  <p align="center">
    <em>Every frontier innovation. Zero abstraction layers. Pure PyTorch.</em>
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#presets">Presets</a> •
  <a href="#deepseek-v3-innovations">DeepSeek V3</a> •
  <a href="#alignment">Alignment</a>
</p>

---

NanoForge is a **from-scratch LLM training framework** that implements the same architectural innovations found in GPT-4, DeepSeek V3, LLaMA 3, and Mistral — in ~600 lines of readable PyTorch. Train on any text, scale from laptop to GPU cluster, then align with human preferences.

## Architecture

| Innovation | What It Does | Origin |
|-----------|-------------|--------|
| 🧠 **Multi-head Latent Attention (MLA)** | Compresses KV into low-rank latent space — **~10x smaller cache** | DeepSeek V3 |
| 🔄 **Grouped Query Attention (GQA)** | Fewer KV heads → faster inference, less memory | GPT-4, LLaMA |
| 🧩 **DeepSeekMoE** | Shared + routed experts with sigmoid gating | DeepSeek V3 |
| ⚖️ **Aux-Loss-Free Routing** | Dynamic bias replaces aux loss — **better model quality** | DeepSeek V3 |
| 🔮 **Multi-Token Prediction** | Predicts N+1, N+2... tokens — denser training gradients | DeepSeek V3 |
| ⚡ **KV-Cache** | Incremental decoding — O(1) per token instead of O(n) | Universal |
| 📐 **Decoupled RoPE** | Separates positional info from semantic attention | DeepSeek V3 |
| 🔥 **SwiGLU + RMSNorm** | Modern FFN activation + stable normalization | GPT-4, LLaMA |
| 🎯 **DPO Alignment** | Align with human preferences — no reward model needed | LLaMA 3, Zephyr |
| 🌐 **FSDP** | Multi-GPU training via FullyShardedDataParallel | PyTorch |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/viralcode/nanoforge.git
cd nanoforge
pip install torch numpy

# Prepare data (downloads Shakespeare sample, uses GPT-4's BPE tokenizer)
python data/prepare_data.py

# Train a small model (works on CPU/laptop)
python train.py --preset small

# Generate text
python generate.py --prompt "To be or not to be" --interactive
```

### Train on Your Own Data

```bash
python data/prepare_data.py --input your_textfile.txt
python train.py --preset medium
```

## Presets

| Preset | Params | Attention | MoE | Context | Best For |
|--------|--------|-----------|-----|---------|----------|
| `small` | ~35M | MHA | — | 256 | CPU / laptop testing |
| `medium` | ~125M | GQA 12Q/4KV | — | 512 | Single GPU |
| `large` | ~333M | GQA 16Q/4KV | — | 2048 | Good GPU (A100/4090) |
| `xl` | ~1.3B | GQA 16Q/8KV | — | 4096 | Multi-GPU |
| `gpt4` | ~100B | GQA 32Q/8KV | 8×top-2 | 8192 | GPU cluster |
| `deepseek` | variable | **MLA** | 64×top-6 + 2 shared | 4096 | GPU cluster (DeepSeek V3 arch) |

```bash
# Scale up as your hardware allows
python train.py --preset small          # Laptop
python train.py --preset medium         # 1× GPU
python train.py --preset large          # 1× A100

# Multi-GPU with FSDP
torchrun --nproc_per_node=4 train.py --preset xl --distributed
torchrun --nproc_per_node=8 train.py --preset deepseek --distributed
```

## DeepSeek V3 Innovations

NanoForge implements the three core innovations from the [DeepSeek V3 paper](https://arxiv.org/abs/2412.19437):

### Multi-head Latent Attention (MLA)

Standard attention caches full K,V per head. MLA compresses them into a tiny latent vector:

```
Input → W_KV_A → [c_KV (latent), k_rope (position)]
                       ↓
                    RMSNorm
                       ↓
                W_KV_B → [k_content, v]  (per head, on-the-fly)

Cache stores only (c_KV, k_rope) → ~10x smaller than GQA
```

### Aux-Loss-Free MoE with Shared Experts

- **Shared experts** process every token (stable base knowledge)
- **Routed experts** are selected top-k per token (specialization)
- Dynamic bias `b_i` adjusts routing without polluting the loss function

### Multi-Token Prediction (MTP)

Each training step predicts multiple future tokens (not just the next one), providing denser gradients. MTP modules are **discarded at inference** — zero overhead.

## Alignment

Align your model with human preferences using Direct Preference Optimization (DPO):

```bash
# Create preference data (JSONL):
# {"prompt": "...", "chosen": "good response", "rejected": "bad response"}

python align.py --checkpoint checkpoints/best.pt --data preferences.jsonl
python generate.py --checkpoint checkpoints/aligned.pt --interactive
```

## Project Structure

```
nanoforge/
├── model.py            # MLA, GQA, DeepSeekMoE, MTP, KV-cache, RoPE, SwiGLU
├── config.py           # All hyperparameters + presets (small → deepseek)
├── train.py            # Training loop (AdamW, cosine LR, FSDP, mixed precision)
├── generate.py         # Text generation with KV-cache
├── align.py            # DPO alignment from preference pairs
├── data/
│   └── prepare_data.py # Tokenization (tiktoken BPE or character-level)
└── requirements.txt
```

## What This Is (and Isn't)

**This is**: A research-grade LLM training framework implementing the same architecture as GPT-4 and DeepSeek V3. Every innovation is implemented from scratch in readable PyTorch — no hidden abstractions.

**This isn't**: A pretrained model. The architecture is frontier-level, but producing a ChatGPT-quality model requires training on trillions of tokens across thousands of GPUs. This gives you the blueprint; you provide the compute.

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA, DeepSeekMoE, Multi-Token Prediction
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — MoE, GQA
- [LLaMA 2](https://arxiv.org/abs/2307.09288) — GQA, SwiGLU, RMSNorm, RoPE
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Alignment
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Inspiration

## License

MIT
