<p align="center">
  <h1 align="center">microGPT</h1>
  <p align="center"><strong>Train your own LLM from scratch — with every frontier innovation</strong></p>
  <p align="center">
    <em>GPT-4 • DeepSeek V3 • Gemma 2 • Mistral • LLaMA 3 — Zero abstraction. Pure PyTorch.</em>
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#presets">Presets</a> •
  <a href="#generation">Generation</a> •
  <a href="#lora-fine-tuning">LoRA</a> •
  <a href="#alignment">Alignment</a> •
  <a href="#export">Export</a>
</p>

---

microGPT is a **from-scratch LLM training framework** implementing every major innovation from GPT-4 through DeepSeek V3, Gemma 2, and Mistral — in readable PyTorch. Train on any text, scale from laptop to GPU cluster, fine-tune with LoRA, align with DPO, export to GGUF.

## Architecture

| Innovation | What It Does | Origin |
|-----------|-------------|--------|
| 🧠 **Multi-head Latent Attention (MLA)** | Compresses KV into low-rank latent — **~10x smaller cache** | DeepSeek V3 |
| 🔄 **Grouped Query Attention (GQA)** | Fewer KV heads → faster, less memory | GPT-4, LLaMA |
| 🪟 **Sliding Window Attention** | O(n·w) attention — handles very long sequences | Mistral |
| 🔄 **Alternating Global/Local Layers** | Even=full attention, odd=windowed — best of both | Gemma 2 |
| 🛡️ **Logit Soft-Capping** | Prevents attention logit explosion | Gemma 2 |
| ⚡ **Flash Attention** | 2-4x faster via PyTorch SDPA backend | FlashAttention-2 |
| 🧩 **DeepSeekMoE** | Shared + routed experts with sigmoid gating | DeepSeek V3 |
| ⚖️ **Aux-Loss-Free Routing** | Dynamic bias replaces aux loss | DeepSeek V3 |
| 🔮 **Multi-Token Prediction** | Predicts N+1, N+2... — denser gradients | DeepSeek V3 |
| 📐 **Decoupled RoPE** | Separates position from content attention | DeepSeek V3 |
| 🌐 **YaRN Context Extension** | Extend context window without retraining | LLaMA 3.1, Qwen |
| 🔥 **SwiGLU + RMSNorm** | Modern FFN + stable normalization | GPT-4, LLaMA |
| 💾 **KV-Cache** | O(1) per token incremental decoding | Universal |
| 🎯 **DPO Alignment** | Align with preferences — no reward model | LLaMA 3, Zephyr |
| 🔧 **LoRA Fine-tuning** | 100x fewer params to train | Microsoft |
| 🏎️ **Speculative Decoding** | 2-3x faster inference with draft model | Google/DeepMind |
| 🎲 **Top-p / Min-p / Rep Penalty** | Advanced sampling strategies | All frontier models |
| ✅ **Gradient Checkpointing** | ~60% memory reduction | Universal |
| 📈 **WSD LR Schedule** | Warmup-Stable-Decay for better convergence | DeepSeek V3 |
| 📦 **GGUF Export** | Run your model in llama.cpp / Ollama | llama.cpp |
| 🧬 **Knowledge Distillation** | Transfer knowledge from large to small model | DeepSeek R1, Qwen |
| 🌐 **FSDP** | Multi-GPU training | PyTorch |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/viralcode/microGPT.git
cd microGPT
pip install torch numpy

# Prepare data (included Shakespeare dataset, or use your own)
python data/prepare_data.py

# Train a small model (works on CPU/laptop)
python train.py --preset small

# Generate text
python generate.py --prompt "To be or not to be" --interactive
```

## Example Output

Trained on Shakespeare (~1MB of text) with the `small` preset on a MacBook:

```
$ python generate.py --prompt "To be or not to be" --top-p 0.9

To be or not to be ta'en of the tomb:
I'll pay not to see your honour's love.

LADY CAPULET:
You would have you sorrow to my heart did lie.

Nurse:
And that's the prince still tell you have said
And you for your mistre
```

```
$ python generate.py --prompt "ROMEO:" --top-p 0.9 --min-p 0.05 --rep-penalty 1.1

ROMEO:
Ay, so much lengthen'd with such a happy great father.

JULIET:
I would you call thee that he is so,
So many some content to the balm of Edward;
That had not fly thee to shake the noble duke.
```

> **10.6M params • val loss 1.479 • 49 tokens/sec on CPU • trained in ~45 min**

### Shakespeare Training Results

The included Shakespeare dataset (`data/input.txt`, ~1MB) was trained with the `small` preset on a MacBook (CPU only):

| Metric | Value |
|--------|-------|
| **Model** | `small` preset — 6 layers, 6 heads, 384 dim |
| **Parameters** | 10.6M |
| **Training data** | Tiny Shakespeare (1.1MB, ~300K tokens) |
| **Tokenizer** | Character-level (vocab_size=65) |
| **Batch size** | 32 |
| **Max iterations** | 5,000 |
| **Best val loss** | **1.479** (at iteration 1,500) |
| **Training time** | ~45 min on CPU (Apple M-series) |
| **Inference speed** | 49 tokens/sec with KV-cache |

The model learns Shakespeare's writing style, character names (ROMEO, JULIET, LADY CAPULET), dialogue structure, and poetic phrasing — all from just ~1MB of text.

### Training Options

```bash
# Basic training
python train.py --preset small --max-iters 5000

# Memory-efficient (saves ~60% VRAM)
python train.py --preset large --gradient-checkpointing

# DeepSeek V3's learning rate schedule
python train.py --preset medium --lr-schedule wsd

# Custom learning rate and batch size
python train.py --preset medium --lr 1e-4 --batch-size 128

# Resume from checkpoint
python train.py --preset small --resume checkpoints/latest.pt

# Multi-GPU with FSDP
torchrun --nproc_per_node=4 train.py --preset xl --distributed

# Compile for maximum speed (PyTorch 2.0+)
python train.py --preset medium --compile
```

### Train on Your Own Data

```bash
python data/prepare_data.py --input your_textfile.txt
python train.py --preset medium
```

## Presets

| Preset | Params | Attention | MoE | Special | Best For |
|--------|--------|-----------|-----|---------|----------|
| `small` | ~35M | MHA | — | — | CPU / laptop |
| `medium` | ~125M | GQA 12Q/4KV | — | — | Single GPU |
| `large` | ~333M | GQA 16Q/4KV | — | — | A100/4090 |
| `xl` | ~1.3B | GQA 16Q/8KV | — | — | Multi-GPU |
| `gpt4` | ~100B | GQA 32Q/8KV | 8×top-2 | — | GPU cluster |
| `deepseek` | variable | **MLA** | 64×top-6+2shared | aux-free, MTP | GPU cluster |
| `mistral` | ~7B | GQA 32Q/8KV | — | **sliding window 4K** | GPU cluster |
| `gemma2` | ~2.7B | GQA 16Q/4KV | — | **alternating layers, logit cap** | GPU cluster |

```bash
# Scale up as your hardware allows
python train.py --preset small                    # Laptop
python train.py --preset medium                   # 1× GPU
python train.py --preset large --gradient-checkpointing  # Memory-efficient

# Training options
python train.py --preset medium --lr-schedule wsd  # DeepSeek V3 LR schedule
python train.py --preset large --gradient-checkpointing --compile  # Max efficiency

# Multi-GPU with FSDP
torchrun --nproc_per_node=4 train.py --preset xl --distributed
torchrun --nproc_per_node=8 train.py --preset deepseek --distributed
```

## Generation

```bash
# Standard generation
python generate.py --prompt "Once upon a time" --interactive

# Advanced sampling
python generate.py --prompt "Once" --top-p 0.9 --min-p 0.05 --rep-penalty 1.2

# Speculative decoding (2-3x faster!)
# Train a small draft model first, then:
python generate.py --draft-checkpoint checkpoints/small.pt --spec-k 5
```

### Sampling Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| Top-k | `--top-k 50` | Keep top-k highest probability tokens |
| Top-p (nucleus) | `--top-p 0.9` | Keep tokens until cumulative probability reaches p |
| Min-p | `--min-p 0.05` | Filter tokens below 5% of the max probability |
| Repetition penalty | `--rep-penalty 1.2` | Reduce probability of repeated tokens |
| Temperature | `--temperature 0.8` | Control randomness (0=greedy, 1=diverse) |

## LoRA Fine-tuning

Fine-tune with only ~1-3% trainable parameters:

```bash
# Fine-tune a pre-trained model
python finetune.py --checkpoint checkpoints/best.pt --data data/ --lora-rank 16

# Custom LoRA settings
python finetune.py --checkpoint best.pt --data data/ --lora-rank 32 --lora-alpha 64

# Generate with fine-tuned model
python generate.py --checkpoint checkpoints/finetuned_merged.pt --interactive
```

## Alignment

Align your model with human preferences using DPO:

```bash
# Create preference data (JSONL):
# {"prompt": "...", "chosen": "good response", "rejected": "bad response"}

python align.py --checkpoint checkpoints/best.pt --data preferences.jsonl
python generate.py --checkpoint checkpoints/aligned.pt --interactive
```

## Export

Export to GGUF format for use with llama.cpp, Ollama, LM Studio:

```bash
# FP16 (full quality)
python export.py --checkpoint best.pt --output model-fp16.gguf

# Q8_0 (8-bit quantized, good quality, smaller)
python export.py --checkpoint best.pt --output model-q8.gguf --quantize q8_0

# Q4_0 (4-bit quantized, smallest, fastest)
python export.py --checkpoint best.pt --output model-q4.gguf --quantize q4_0
```

## Context Extension (YaRN)

Extend your model's context window at inference without retraining:

```python
from config import GPTConfig

config = GPTConfig(
    ...,
    rope_scaling_type="yarn",   # or "linear"
    rope_scaling_factor=4.0,    # 4x context: 4K → 16K
)
```

## Knowledge Distillation

Transfer knowledge from a large teacher model to a smaller student model. Supports both HuggingFace models (Qwen, LLaMA, Mistral) and microGPT checkpoints.

```bash
# Distill from Qwen (requires: pip install transformers)
python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --student-preset small --data data/

# Distill from LLaMA
python distill.py --hf-teacher meta-llama/Llama-3.2-1B --student-preset medium

# Distill from a larger microGPT model
python distill.py --teacher checkpoints/large.pt --student-preset small --data data/

# Custom temperature and balance
python distill.py --hf-teacher Qwen/Qwen2.5-0.5B --temperature 3.0 --alpha 0.7

# Generate with the distilled model
python generate.py --checkpoint checkpoints/distilled_best.pt --interactive
```

**Recommended HuggingFace teachers:**

| Model | Size | Best For |
|-------|------|----------|
| `Qwen/Qwen2.5-0.5B` | 500M | Quick experiments, CPU-friendly |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Good quality, single GPU |
| `meta-llama/Llama-3.2-1B` | 1B | Strong baseline |
| `mistralai/Mistral-7B-v0.3` | 7B | High quality, needs GPU |

## Project Structure

```
microGPT/
├── model.py            # MLA, GQA, sliding window, Flash Attn, MoE, MTP, KV-cache,
│                       # RoPE+YaRN, SwiGLU, speculative decoding, grad checkpointing
├── config.py           # All hyperparameters + presets (small → gemma2)
├── train.py            # Training (AdamW, cosine/WSD LR, FSDP, grad ckpt, mixed prec)
├── generate.py         # Generation (top-k/p, min-p, rep penalty, speculative decoding)
├── align.py            # DPO alignment from preference pairs
├── distill.py          # Knowledge distillation (teacher → student)
├── lora.py             # LoRA: apply, merge, save, load
├── finetune.py         # LoRA fine-tuning script
├── export.py           # GGUF export (FP16, Q8_0, Q4_0)
├── data/
│   └── prepare_data.py # Tokenization (tiktoken BPE or character-level)
└── requirements.txt
```

## What This Is (and Isn't)

**This is**: The most comprehensive from-scratch LLM framework, implementing every major innovation from GPT-4 through the latest frontier models. Every feature is implemented in readable PyTorch — no hidden abstractions.

**This isn't**: A pretrained model. The architecture is frontier-level, but producing a ChatGPT-quality model requires trillions of tokens and thousands of GPUs. This gives you the complete blueprint; you provide the compute.

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA, DeepSeekMoE, MTP, WSD schedule
- [Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118) — Alternating attention, logit soft-capping
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Sliding window attention
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — MoE, GQA
- [LLaMA 2](https://arxiv.org/abs/2307.09288) — GQA, SwiGLU, RMSNorm, RoPE
- [YaRN](https://arxiv.org/abs/2309.00071) — Context extension via RoPE scaling
- [LoRA](https://arxiv.org/abs/2106.09685) — Low-rank adaptation
- [DPO](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [Speculative Decoding](https://arxiv.org/abs/2302.01318) — Draft-verify acceleration
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Inspiration

## License

MIT
