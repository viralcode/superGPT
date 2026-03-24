"""
superGPT Neural Network Visualizer
====================================
Real-time web dashboard for visualizing neural network internals.

Features:
  - Architecture diagram: layers, attention heads, MoE routing
  - Weight distributions: histograms per layer
  - Attention heatmaps: attention correlations
  - Training metrics: loss curves, learning rate
  - Activation flow: watch data flow through the network
  - Network canvas: live animated visualization
  - Checkpoint inspector: model size, config, metadata

Usage:
    python -m supergpt.tools.visualize --checkpoint checkpoints/best.pt
    python -m supergpt.tools.visualize --checkpoint checkpoints/best.pt --port 8050
"""

import os
import sys
import json
import math
import argparse
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from supergpt.core.config import GPTConfig
from supergpt.core.model import GPT


# ==============================================================================
#  Model Analysis
# ==============================================================================

def analyze_model(checkpoint_path, device="cpu"):
    """Load and analyze a model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    analysis = {
        "config": ckpt["model_config"],
        "checkpoint": {
            "path": checkpoint_path,
            "iter_num": ckpt.get("iter_num", "unknown"),
            "best_val_loss": ckpt.get("best_val_loss", None),
            "distillation": ckpt.get("distillation", None),
            "alignment": ckpt.get("alignment", None),
        },
        "architecture": get_architecture_info(model, config),
        "weights": get_weight_stats(model),
        "layer_details": get_layer_details(model, config),
    }

    return model, config, analysis


def get_architecture_info(model, config):
    """Extract architecture overview."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Detect attention type
    attn_type = "GQA"
    if config.use_mla:
        attn_type = "MLA (Multi-head Latent)"
    elif getattr(config, 'use_nsa', False):
        attn_type = "NSA (Native Sparse)"
    elif config.n_kv_head == config.n_head:
        attn_type = "MHA (Multi-Head)"

    # Detect FFN type
    ffn_type = "SwiGLU" if config.use_swiglu else "GELU"
    if config.use_moe:
        ffn_type += f" + MoE({config.n_experts}E, top-{config.n_experts_active})"

    return {
        "total_params": total_params,
        "trainable_params": trainable,
        "n_layers": config.n_layer,
        "n_heads": config.n_head,
        "n_kv_heads": config.n_kv_head,
        "n_embd": config.n_embd,
        "block_size": config.block_size,
        "vocab_size": config.vocab_size,
        "attention_type": attn_type,
        "ffn_type": ffn_type,
        "rope": config.use_rope,
        "mtp": config.n_predict_tokens > 1,
    }


def get_weight_stats(model):
    """Compute weight statistics for each parameter."""
    stats = []
    for name, param in model.named_parameters():
        data = param.data.float()
        stats.append({
            "name": name,
            "shape": list(param.shape),
            "numel": param.numel(),
            "mean": data.mean().item(),
            "std": data.std().item(),
            "min": data.min().item(),
            "max": data.max().item(),
            "abs_mean": data.abs().mean().item(),
            "norm": data.norm().item(),
            # Histogram data (20 bins)
            "histogram": compute_histogram(data, bins=20),
        })
    return stats


def compute_histogram(tensor, bins=20):
    """Compute histogram of tensor values."""
    data = tensor.flatten().cpu()
    hist = torch.histc(data, bins=bins)
    min_val = data.min().item()
    max_val = data.max().item()
    bin_edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]
    return {
        "counts": hist.tolist(),
        "edges": bin_edges,
    }


def get_layer_details(model, config):
    """Get per-layer information for the architecture diagram."""
    layers = []

    # Embedding
    layers.append({
        "type": "embedding",
        "name": "Token Embedding",
        "params": config.vocab_size * config.n_embd,
        "shape": [config.vocab_size, config.n_embd],
    })
    layers.append({
        "type": "embedding",
        "name": "Position Embedding",
        "params": config.block_size * config.n_embd if not config.use_rope else 0,
        "shape": [config.block_size, config.n_embd] if not config.use_rope else "RoPE",
    })

    # Transformer blocks
    for i in range(config.n_layer):
        block = {
            "type": "transformer_block",
            "name": f"Block {i}",
            "layer_id": i,
            "components": [],
        }

        # Attention
        attn_type = "GQA"
        if config.use_mla:
            attn_type = "MLA"
        elif getattr(config, 'use_nsa', False):
            attn_type = "NSA"
        block["components"].append({
            "type": "attention",
            "name": attn_type,
            "n_heads": config.n_head,
            "n_kv_heads": config.n_kv_head,
        })

        # FFN
        if config.use_moe and i >= config.n_dense_layers:
            block["components"].append({
                "type": "moe",
                "n_experts": config.n_experts,
                "n_active": config.n_experts_active,
                "n_shared": config.n_shared_experts,
            })
        else:
            block["components"].append({
                "type": "ffn",
                "name": "SwiGLU" if config.use_swiglu else "GELU",
            })

        layers.append(block)

    # Output
    layers.append({
        "type": "output",
        "name": "LM Head",
        "params": config.vocab_size * config.n_embd,
    })

    return layers


@torch.no_grad()
def get_attention_patterns(model, text, config, device="cpu", max_tokens=64):
    """Generate attention patterns for a given text input."""
    # Simple tokenization
    vocab_size = config.vocab_size
    if vocab_size > 256:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            tokens = tok.encode(text)[:max_tokens]
            token_strs = [tok.decode([t]) for t in tokens]
        except Exception:
            tokens = [min(ord(c), vocab_size - 1) for c in text[:max_tokens]]
            token_strs = list(text[:max_tokens])
    else:
        tokens = [min(ord(c), vocab_size - 1) for c in text[:max_tokens]]
        token_strs = list(text[:max_tokens])

    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # Hook to capture attention weights
    attention_maps = {}

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            # For CausalSelfAttention, capture attention weights
            if hasattr(module, 'attn_dropout'):
                # We'll compute attention from q, k
                pass
        return hook_fn

    # Forward pass — capture hidden states
    hidden_states = []
    def capture_hidden(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach().cpu())
        else:
            hidden_states.append(output.detach().cpu())

    hooks = []
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        h = block.register_forward_hook(capture_hidden)
        hooks.append(h)

    model(x)

    for h in hooks:
        h.remove()

    # Compute approximate attention from hidden states
    patterns = []
    for i, hs in enumerate(hidden_states):
        # Self-correlation as proxy for attention
        hs_norm = F.normalize(hs[0], dim=-1)  # (T, C)
        attn_approx = (hs_norm @ hs_norm.T).tolist()  # (T, T)
        patterns.append({
            "layer": i,
            "attention": attn_approx,
        })

    return {
        "tokens": token_strs,
        "patterns": patterns,
    }


@torch.no_grad()
def get_activation_flow(model, text, config, device="cpu"):
    """Capture activation magnitudes through each layer."""
    vocab_size = config.vocab_size
    tokens = [min(ord(c), vocab_size - 1) for c in text[:32]]
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    activations = []

    def capture(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations.append({
                "name": name,
                "mean": out.float().mean().item(),
                "std": out.float().std().item(),
                "max": out.float().abs().max().item(),
                "norm": out.float().norm().item(),
            })
        return hook

    hooks = []
    # Embedding
    hooks.append(model.transformer.wte.register_forward_hook(capture("embedding")))
    # Each block
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(capture(f"block_{i}")))
    # Final norm
    hooks.append(model.transformer.ln_f.register_forward_hook(capture("final_norm")))

    model(x)

    for h in hooks:
        h.remove()

    return activations


# ==============================================================================
#  HTML Dashboard
# ==============================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>superGPT Visualizer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #000000;
  --bg-card: #09090b;
  --bg-card-hover: #18181b;
  --border: #27272a;
  --text: #ffffff;
  --text-dim: #a1a1aa;
  --text-muted: #52525b;
  --accent: #ffffff;
  --accent-dim: #e4e4e7;
  --highlight: #ffffff;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

.header {
  padding: 32px 40px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 100;
}

.header h1 {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.5px;
  color: var(--text);
  margin-bottom: 4px;
}

.header .subtitle {
  font-size: 13px;
  color: var(--text-dim);
  font-weight: 400;
}

.header-stats { display: flex; gap: 32px; }
.header-stat { text-align: left; }
.header-stat .label {
  font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}
.header-stat .value {
  font-size: 18px; font-weight: 500; font-family: 'JetBrains Mono', monospace; color: var(--text);
}

.tabs {
  display: flex; gap: 8px; padding: 0 40px; border-bottom: 1px solid var(--border); background: var(--bg);
}
.tab {
  padding: 16px 4px; margin-right: 24px; font-size: 13px; font-weight: 500; color: var(--text-dim); cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; user-select: none; letter-spacing: 0.2px;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--text); border-bottom-color: var(--text); }

.main { padding: 40px; max-width: 1600px; margin: 0 auto; }
.section { display: none; transform: translateY(10px); opacity: 0; transition: all 0.4s ease; }
.section.active { display: block; transform: translateY(0); opacity: 1; }

.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 32px; margin-bottom: 24px; }
.card h3 { font-size: 11px; font-weight: 600; color: var(--text); margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px; }

.grid { display: grid; gap: 24px; }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

.stat-box { background: transparent; border: 1px solid var(--border); border-radius: 8px; padding: 24px; text-align: left; }
.stat-box .stat-value { font-size: 24px; font-weight: 400; font-family: 'JetBrains Mono', monospace; color: var(--text); margin-bottom: 8px; }
.stat-box .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }

.arch-diagram { display: flex; flex-direction: column; align-items: center; gap: 16px; padding: 16px 0; }
.arch-node { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 20px 32px; text-align: center; min-width: 320px; position: relative; }
.arch-node .node-title { font-weight: 500; font-size: 13px; color: var(--text); margin-bottom: 6px; }
.arch-node .node-detail { font-size: 12px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
.arch-connector { width: 1px; height: 24px; background: var(--border); }
.block-group { border: 1px solid var(--border); background: var(--bg-card); border-radius: 8px; padding: 24px; display: flex; flex-direction: column; align-items: center; gap: 16px; position: relative; min-width: 380px; }
.block-group .block-label { position: absolute; top: -10px; left: 32px; background: var(--bg-card); padding: 0 8px; font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid var(--border); border-radius: 4px; }
.block-repeat { font-size: 12px; color: var(--text-dim); padding: 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; width: 100%; text-align: center; }
.block-repeat .repeat-badge { color: var(--text); font-family: 'JetBrains Mono', monospace; margin: 0 4px; }

.histogram { display: flex; align-items: flex-end; gap: 1px; height: 60px; padding: 8px 0; }
.histogram .bar { flex: 1; background: var(--text-muted); min-width: 2px; }
.histogram:hover .bar { background: var(--text-dim); }
.weight-table { width: 100%; border-collapse: collapse; }
.weight-table th { text-align: left; padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--text-dim); font-weight: 400; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.weight-table td { padding: 16px; border-bottom: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); }
.weight-name { color: var(--text); }
.weight-shape { color: var(--text-dim); }

.heatmap-container { overflow-x: auto; padding: 24px 0; }
.heatmap { display: inline-block; background: var(--bg); border: 1px solid var(--border); padding: 16px; border-radius: 8px; }
.heatmap-row { display: flex; gap: 1px; margin-bottom: 1px; }
.heatmap-cell { width: 28px; height: 28px; transition: opacity 0.2s; }
.heatmap-cell:hover { opacity: 0.8; border:1px solid #fff;}
.heatmap-labels { display: flex; gap: 1px; margin-bottom: 8px; margin-left: 36px; }
.heatmap-label { width: 28px; text-align: center; font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }

.flow-chart { display: flex; align-items: flex-end; gap: 4px; height: 240px; padding: 24px 0 40px 0; }
.flow-bar { flex: 1; position: relative; background: var(--border); min-width: 30px; border-top: 1px solid var(--text-muted); transition: all 0.2s; }
.flow-bar:hover { background: var(--text-muted); border-top-color: var(--text); }
.flow-bar .flow-label { position: absolute; bottom: -32px; left: 50%; transform: translateX(-50%) rotate(-45deg); font-size: 10px; color: var(--text-dim); white-space: nowrap; transform-origin: left; font-family: 'JetBrains Mono', monospace; }
.flow-bar .flow-value { position: absolute; top: -24px; left: 50%; transform: translateX(-50%); font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--text); }

.layer-selector { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
.layer-btn { padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border); background: transparent; color: var(--text-dim); cursor: pointer; font-size: 12px; font-family: 'JetBrains Mono', monospace; transition: all 0.2s; }
.layer-btn:hover { border-color: var(--text-muted); color: var(--text); }
.layer-btn.active { background: var(--text); color: var(--bg); border-color: var(--text); }

.input-group { display: flex; gap: 12px; margin-bottom: 24px; }
.input-group input { flex: 1; padding: 12px 16px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 13px; outline: none; transition: border-color 0.2s; }
.input-group input:focus { border-color: var(--text-muted); }
.input-group button { padding: 12px 24px; border-radius: 6px; border: 1px solid var(--text); background: var(--text); color: var(--bg); font-weight: 500; font-size: 13px; cursor: pointer; transition: opacity 0.2s; text-transform: uppercase; letter-spacing: 1px;}
.input-group button:hover { opacity: 0.9; }

.config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.config-item { display: flex; justify-content: space-between; padding: 12px 16px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; font-size: 12px; }
.config-key { color: var(--text-dim); }
.config-val { color: var(--text); font-family: 'JetBrains Mono', monospace; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>superGPT Visualizer</h1>
    <div class="subtitle">Architecture & Weights Inspector</div>
  </div>
  <div class="header-stats" id="headerStats"></div>
</div>

<div class="tabs" id="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="architecture">Architecture</div>
  <div class="tab" data-tab="weights">Weights</div>
  <div class="tab" data-tab="attention">Attention</div>
  <div class="tab" data-tab="activations">Activations</div>
  <div class="tab" data-tab="network">Network</div>
  <div class="tab" data-tab="config">Configuration</div>
</div>

<div class="main">
  <!-- Overview -->
  <div class="section active" id="sec-overview">
    <div class="grid grid-4" id="overviewStats"></div>
    <div class="grid grid-2" style="margin-top: 24px;">
      <div class="card">
        <h3>Model Specifications</h3>
        <div id="modelSummary"></div>
      </div>
      <div class="card">
        <h3>Checkpoint State</h3>
        <div id="checkpointInfo"></div>
      </div>
    </div>
    <div class="card">
      <h3>Parameter Distribution</h3>
      <div id="paramDist"></div>
    </div>
  </div>

  <!-- Architecture -->
  <div class="section" id="sec-architecture">
    <div class="card">
      <h3>Network Topology</h3>
      <div class="arch-diagram" id="archDiagram"></div>
    </div>
  </div>

  <!-- Weights -->
  <div class="section" id="sec-weights">
    <div class="card">
      <h3>Parameter Statistics</h3>
      <div style="overflow-x: auto;">
        <table class="weight-table" id="weightTable">
          <thead>
            <tr>
              <th>Tensor</th>
              <th>Shape</th>
              <th>Count</th>
              <th>Mean</th>
              <th>Std Dev</th>
              <th>Distribution</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Attention -->
  <div class="section" id="sec-attention">
    <div class="card">
      <h3>Self-Attention Maps</h3>
      <div class="input-group">
        <input type="text" id="attnInput" value="The quick brown fox jumps over the lazy dog">
        <button onclick="requestAttention()">Compute</button>
      </div>
      <div class="layer-selector" id="layerSelector"></div>
      <div class="heatmap-container" id="heatmapContainer">
        <p style="color: var(--text-dim); text-align: center; padding: 40px; font-size: 13px;">
          Submit text to compute attention mapping
        </p>
      </div>
    </div>
  </div>

  <!-- Activations -->
  <div class="section" id="sec-activations">
    <div class="card">
      <h3>Forward Propagation Magnitudes</h3>
      <div class="input-group">
        <input type="text" id="actInput" value="Initialize sequence">
        <button onclick="requestActivations()">Trace</button>
      </div>
      <div id="activationFlow">
        <p style="color: var(--text-dim); text-align: center; padding: 40px; font-size: 13px;">
          Submit text to trace layer activations
        </p>
      </div>
    </div>
  </div>

  <!-- Neural Network 3D Visualization -->
  <div class="section" id="sec-network">
    <div class="card" style="padding:0;overflow:hidden;position:relative;border:none;">
      <div style="position:absolute;top:24px;left:32px;z-index:10;">
        <h3 style="margin:0;letter-spacing:1px;color:var(--text);">Live Network Topology</h3>
        <p style="font-size:11px;color:var(--text-dim);margin-top:6px;font-family:'JetBrains Mono',monospace;">REAL-TIME ACTIVATION SIMULATION</p>
      </div>
      <div style="position:absolute;top:24px;right:32px;z-index:10;display:flex;gap:12px;">
        <button onclick="toggleAnimation()" id="animToggle" style="padding:8px 16px;border-radius:4px;border:1px solid var(--border);background:var(--bg);color:var(--text);cursor:pointer;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:500;">Pause</button>
        <button onclick="pulseNetwork()" style="padding:8px 16px;border-radius:4px;border:1px solid var(--text);background:var(--text);color:var(--bg);cursor:pointer;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Trigger Pulse</button>
      </div>
      <canvas id="networkCanvas" style="width:100%;height:600px;display:block;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;"></canvas>
    </div>
    <div class="grid grid-3" style="margin-top:24px;">
      <div class="stat-box" style="text-align:center;">
        <div style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Input Layer</div>
        <div style="font-size:18px;color:var(--text);font-family:'JetBrains Mono',monospace;" id="nnInputSize">—</div>
      </div>
      <div class="stat-box" style="text-align:center;">
        <div style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Hidden Blocks</div>
        <div style="font-size:18px;color:var(--text);font-family:'JetBrains Mono',monospace;" id="nnHiddenLayers">—</div>
      </div>
      <div class="stat-box" style="text-align:center;">
        <div style="font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Output Layer</div>
        <div style="font-size:18px;color:var(--text);font-family:'JetBrains Mono',monospace;" id="nnOutputSize">—</div>
      </div>
    </div>
  </div>

  <!-- Config -->
  <div class="section" id="sec-config">
    <div class="card">
      <h3>Configuration Parameters</h3>
      <div class="config-grid" id="configGrid"></div>
    </div>
  </div>
</div>

<script>
let DATA = null;

async function loadData() {
  const resp = await fetch('/api/analysis');
  DATA = await resp.json();
  renderAll();
}

function renderAll() {
  renderHeader();
  renderOverview();
  renderArchitecture();
  renderWeights();
  renderConfig();
}

function formatNum(n) {
  if (n >= 1e9) return (n/1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toString();
}

function renderHeader() {
  const arch = DATA.architecture;
  const ckpt = DATA.checkpoint;
  document.getElementById('headerStats').innerHTML = `
    <div class="header-stat">
      <div class="label">Parameters</div>
      <div class="value">${formatNum(arch.total_params)}</div>
    </div>
    <div class="header-stat">
      <div class="label">Layers</div>
      <div class="value">${arch.n_layers}</div>
    </div>
    <div class="header-stat">
      <div class="label">Val Loss</div>
      <div class="value">${ckpt.best_val_loss ? ckpt.best_val_loss.toFixed(4) : '—'}</div>
    </div>
  `;
}

function renderOverview() {
  const arch = DATA.architecture;
  const ckpt = DATA.checkpoint;

  document.getElementById('overviewStats').innerHTML = `
    <div class="stat-box">
      <div class="stat-value">${formatNum(arch.total_params)}</div>
      <div class="stat-label">Total Params</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${arch.n_layers}L × ${arch.n_heads}H</div>
      <div class="stat-label">Layers × Heads</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${arch.n_embd}</div>
      <div class="stat-label">Hidden Dim</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${arch.block_size}</div>
      <div class="stat-label">Context Length</div>
    </div>
  `;

  document.getElementById('modelSummary').innerHTML = `
    <div class="config-grid">
      <div class="config-item"><span class="config-key">Attention</span><span class="config-val">${arch.attention_type}</span></div>
      <div class="config-item"><span class="config-key">FFN</span><span class="config-val">${arch.ffn_type}</span></div>
      <div class="config-item"><span class="config-key">KV Heads</span><span class="config-val">${arch.n_kv_heads}</span></div>
      <div class="config-item"><span class="config-key">Vocab Size</span><span class="config-val">${formatNum(arch.vocab_size)}</span></div>
      <div class="config-item"><span class="config-key">RoPE</span><span class="config-val">${arch.rope ? 'True' : 'False'}</span></div>
      <div class="config-item"><span class="config-key">MTP</span><span class="config-val">${arch.mtp ? 'True' : 'False'}</span></div>
    </div>
  `;

  let ckptHtml = `
    <div class="config-grid">
      <div class="config-item"><span class="config-key">Step</span><span class="config-val">${ckpt.iter_num}</span></div>
      <div class="config-item"><span class="config-key">Val Loss</span><span class="config-val">${ckpt.best_val_loss ? ckpt.best_val_loss.toFixed(4) : '—'}</span></div>
  `;
  if (ckpt.distillation) {
    ckptHtml += `
      <div class="config-item"><span class="config-key">Teacher</span><span class="config-val">${ckpt.distillation.teacher || '—'}</span></div>
      <div class="config-item"><span class="config-key">Compression</span><span class="config-val">${ckpt.distillation.compression ? ckpt.distillation.compression.toFixed(1) + 'x' : '—'}</span></div>
    `;
  }
  if (ckpt.alignment) {
    ckptHtml += `
      <div class="config-item"><span class="config-key">Alignment</span><span class="config-val">${ckpt.alignment.method || '—'}</span></div>
    `;
  }
  ckptHtml += '</div>';
  document.getElementById('checkpointInfo').innerHTML = ckptHtml;

  const weights = DATA.weights;
  let paramsByType = { embedding: 0, attention: 0, ffn: 0, norm: 0, other: 0 };
  weights.forEach(w => {
    if (w.name.includes('wte') || w.name.includes('wpe') || w.name.includes('lm_head'))
      paramsByType.embedding += w.numel;
    else if (w.name.includes('attn') || w.name.includes('proj') || w.name.includes('wq') || w.name.includes('wkv'))
      paramsByType.attention += w.numel;
    else if (w.name.includes('ffn') || w.name.includes('w1') || w.name.includes('w2') || w.name.includes('w3') || w.name.includes('c_fc'))
      paramsByType.ffn += w.numel;
    else if (w.name.includes('ln') || w.name.includes('norm'))
      paramsByType.norm += w.numel;
    else
      paramsByType.other += w.numel;
  });

  const total = Object.values(paramsByType).reduce((a,b) => a+b, 0);
  const colors = { embedding: '#ffffff', attention: '#d4d4d8', ffn: '#a1a1aa', norm: '#71717a', other: '#3f3f46' };
  let distHtml = '<div style="display:flex;height:24px;border-radius:4px;overflow:hidden;margin-bottom:16px;">';
  for (const [k, v] of Object.entries(paramsByType)) {
    const pct = (v / total * 100);
    if (pct > 0.5) {
      distHtml += `<div style="width:${pct}%;background:${colors[k]};"></div>`;
    }
  }
  distHtml += '</div><div style="display:flex;gap:24px;flex-wrap:wrap;">';
  for (const [k, v] of Object.entries(paramsByType)) {
    distHtml += `<span style="font-size:12px;display:flex;align-items:center;gap:8px;color:var(--text-dim);"><span style="width:8px;height:8px;border-radius:2px;background:${colors[k]};"></span>${k}: <span style="font-family:'JetBrains Mono',monospace;color:var(--text);">${formatNum(v)}</span></span>`;
  }
  distHtml += '</div>';
  document.getElementById('paramDist').innerHTML = distHtml;
}

function renderArchitecture() {
  const arch = DATA.architecture;
  let html = '';

  html += `<div class="arch-node">
    <div class="node-title">Token Embedding</div>
    <div class="node-detail">${formatNum(arch.vocab_size)} tokens → ${arch.n_embd}d</div>
  </div>`;
  html += '<div class="arch-connector"></div>';

  if (arch.rope) {
    html += `<div class="arch-node">
      <div class="node-title">Rotary Position Encoding (RoPE)</div>
      <div class="node-detail">Computed continuously without learned parameters</div>
    </div>`;
  } else {
    html += `<div class="arch-node">
      <div class="node-title">Position Embedding</div>
      <div class="node-detail">${arch.block_size} positions → ${arch.n_embd}d</div>
    </div>`;
  }
  html += '<div class="arch-connector"></div>';

  html += '<div class="block-group">';
  html += `<div class="block-label">Transformer Sequence</div>`;

  html += `<div class="arch-node">
    <div class="node-title">${arch.attention_type}</div>
    <div class="node-detail">${arch.n_heads} heads, ${arch.n_kv_heads} KV heads, ${arch.n_embd}d</div>
  </div>`;
  html += '<div class="arch-connector"></div>';

  if (arch.ffn_type.includes('MoE')) {
    html += `<div class="arch-node">
      <div class="node-title">${arch.ffn_type}</div>
      <div class="node-detail">Dynamic Expert Router</div>
    </div>`;
  } else {
    html += `<div class="arch-node">
      <div class="node-title">${arch.ffn_type} Feed-Forward</div>
      <div class="node-detail">${arch.n_embd}d → ${arch.n_embd * 4}d → ${arch.n_embd}d</div>
    </div>`;
  }

  html += `<div class="block-repeat">
    Repeated <span class="repeat-badge">×${arch.n_layers}</span> sequentially
  </div>`;
  html += '</div>';

  html += '<div class="arch-connector"></div>';
  html += `<div class="arch-node">
    <div class="node-title">Linear Head</div>
    <div class="node-detail">${arch.n_embd}d → ${formatNum(arch.vocab_size)} logits</div>
  </div>`;

  document.getElementById('archDiagram').innerHTML = html;
}

function renderWeights() {
  const tbody = document.querySelector('#weightTable tbody');
  let html = '';

  DATA.weights.forEach(w => {
    const maxCount = Math.max(...w.histogram.counts);
    let histHtml = '<div class="histogram">';
    w.histogram.counts.forEach(c => {
      const h = Math.max(2, (c / maxCount) * 100);
      histHtml += `<div class="bar" style="height:${h}%"></div>`;
    });
    histHtml += '</div>';

    html += `<tr>
      <td class="weight-name">${w.name}</td>
      <td class="weight-shape">${JSON.stringify(w.shape).replace(/,/g, ', ')}</td>
      <td>${formatNum(w.numel)}</td>
      <td>${w.mean.toFixed(4)}</td>
      <td>${w.std.toFixed(4)}</td>
      <td style="width:120px;padding:8px 16px;">${histHtml}</td>
    </tr>`;
  });

  tbody.innerHTML = html;
}

function renderConfig() {
  const config = DATA.config;
  let html = '';
  for (const [key, val] of Object.entries(config)) {
    html += `<div class="config-item">
      <span class="config-key">${key}</span>
      <span class="config-val">${JSON.stringify(val)}</span>
    </div>`;
  }
  document.getElementById('configGrid').innerHTML = html;
}

async function requestAttention() {
  const text = document.getElementById('attnInput').value;
  const resp = await fetch('/api/attention?text=' + encodeURIComponent(text));
  const data = await resp.json();
  renderAttention(data);
}

function renderAttention(data) {
  let selectorHtml = '';
  data.patterns.forEach((p, i) => {
    selectorHtml += `<button class="layer-btn ${i === 0 ? 'active' : ''}"
      onclick="showLayer(${i})" id="lbtn-${i}">Layer ${i}</button>`;
  });
  document.getElementById('layerSelector').innerHTML = selectorHtml;

  window._attnData = data;
  showLayer(0);
}

function showLayer(idx) {
  document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('lbtn-' + idx)?.classList.add('active');

  const data = window._attnData;
  const pattern = data.patterns[idx];
  const tokens = data.tokens;

  let html = '<div class="heatmap">';
  html += '<div class="heatmap-labels">';
  tokens.forEach(t => {
    const display = t.replace(' ', '·').replace('\\n', '↵');
    html += `<div class="heatmap-label">${display}</div>`;
  });
  html += '</div>';

  pattern.attention.forEach((row, i) => {
    html += '<div class="heatmap-row">';
    html += `<div class="heatmap-label" style="text-align:right;margin-right:8px;line-height:28px;">${tokens[i].replace(' ', '·').replace('\\n', '↵')}</div>`;
    row.forEach((val, j) => {
      const intensity = Math.min(Math.abs(val), 1);
      const colorVal = Math.floor(255 * intensity);
      html += `<div class="heatmap-cell" style="background:rgb(${colorVal},${colorVal},${colorVal})"
        title="${tokens[i]}→${tokens[j]}: ${val.toFixed(3)}"></div>`;
    });
    html += '</div>';
  });
  html += '</div>';

  document.getElementById('heatmapContainer').innerHTML = html;
}

async function requestActivations() {
  const text = document.getElementById('actInput').value;
  const resp = await fetch('/api/activations?text=' + encodeURIComponent(text));
  const data = await resp.json();
  renderActivations(data);
}

function renderActivations(data) {
  const maxNorm = Math.max(...data.map(d => d.norm));

  let html = '<div class="flow-chart">';
  data.forEach((d, i) => {
    const h = (d.norm / maxNorm) * 100;
    html += `<div class="flow-bar" style="height:${Math.max(h, 5)}%;">
      <div class="flow-value">${d.norm.toFixed(1)}</div>
      <div class="flow-label">${d.name}</div>
    </div>`;
  });
  html += '</div>';
  document.getElementById('activationFlow').innerHTML = html;
}

// Tab switching
document.getElementById('tabs').addEventListener('click', e => {
  if (!e.target.classList.contains('tab')) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  e.target.classList.add('active');
  const targetId = 'sec-' + e.target.dataset.tab;
  const section = document.getElementById(targetId);
  section.classList.add('active');
  
  if (e.target.dataset.tab === 'network') {
    setTimeout(renderNetwork, 100);
  }
});

let nnAnimating = true;
let nnAnimFrame = null;
let nnParticles = [];
let nnPulseWave = -1;

function toggleAnimation() {
  nnAnimating = !nnAnimating;
  const btn = document.getElementById('animToggle');
  btn.textContent = nnAnimating ? 'Pause' : 'Play';
  btn.style.background = nnAnimating ? 'var(--bg)' : 'var(--text)';
  btn.style.color = nnAnimating ? 'var(--text)' : 'var(--bg)';
  if (nnAnimating) renderNetwork();
}

function pulseNetwork() {
  nnPulseWave = 0;
}

function renderNetwork() {
  if (!DATA) return;
  const canvas = document.getElementById('networkCanvas');
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);

  const W = rect.width;
  const H = rect.height;
  const arch = DATA.architecture;

  document.getElementById('nnInputSize').textContent = arch.n_embd;
  document.getElementById('nnHiddenLayers').textContent = arch.n_layers + ' × ' + arch.n_heads;
  document.getElementById('nnOutputSize').textContent = formatNum(arch.vocab_size);

  const nodesPerLayer = [];
  const layerLabels = [];

  const inputNodes = Math.min(12, arch.n_embd);
  nodesPerLayer.push(inputNodes);
  layerLabels.push('EMB');

  for (let i = 0; i < Math.min(arch.n_layers, 8); i++) {
    nodesPerLayer.push(Math.min(arch.n_heads, 12));
    layerLabels.push('ATT ' + i);
    nodesPerLayer.push(Math.min(8, Math.floor(arch.n_embd * 4 / 100)));
    layerLabels.push('FFN ' + i);
  }

  nodesPerLayer.push(Math.min(10, arch.vocab_size));
  layerLabels.push('OUT');

  const numLayers = nodesPerLayer.length;
  const padX = 60;
  const padY = 60;
  const layerSpacing = (W - padX * 2) / (numLayers - 1);
  const t = Date.now() / 1000;

  const nodes = [];
  for (let l = 0; l < numLayers; l++) {
    const n = nodesPerLayer[l];
    const x = padX + l * layerSpacing;
    const nodeSpacing = Math.min(44, (H - padY * 2) / (n + 1));
    const startY = H / 2 - (n - 1) * nodeSpacing / 2;

    const layerNodes = [];
    for (let i = 0; i < n; i++) {
      const y = startY + i * nodeSpacing;
      const phase = (l * 0.7 + i * 0.3 + t * 1.5) % (Math.PI * 2);
      const pulse = 0.5 + 0.5 * Math.sin(phase);

      let wavePulse = 0;
      if (nnPulseWave >= 0) {
        const dist = Math.abs(l - nnPulseWave);
        if (dist < 1.5) wavePulse = Math.max(0, 1 - dist);
      }
      layerNodes.push({ x, y, pulse, wavePulse });
    }
    nodes.push(layerNodes);
  }

  ctx.clearRect(0, 0, W, H);

  // Connections
  for (let l = 0; l < numLayers - 1; l++) {
    const src = nodes[l];
    const dst = nodes[l + 1];
    for (let i = 0; i < src.length; i++) {
      for (let j = 0; j < dst.length; j++) {
        if (Math.abs(i / src.length - j / dst.length) > 0.6 && src.length > 3) continue;

        const baseAlpha = 0.05 + 0.05 * Math.sin(t * 0.8 + i * 0.5 + j * 0.3);
        const waveAlpha = (src[i].wavePulse + dst[j].wavePulse) * 0.3;

        ctx.beginPath();
        ctx.moveTo(src[i].x, src[i].y);
        ctx.lineTo(dst[j].x, dst[j].y);
        ctx.strokeStyle = `rgba(255, 255, 255, ${baseAlpha + waveAlpha})`;
        ctx.lineWidth = 0.5 + waveAlpha * 1.5;
        ctx.stroke();
      }
    }
  }

  // Particles
  if (Math.random() < 0.2) {
    const srcL = Math.floor(Math.random() * (numLayers - 1));
    const srcN = Math.floor(Math.random() * nodes[srcL].length);
    const dstN = Math.floor(Math.random() * nodes[srcL + 1].length);
    const src = nodes[srcL][srcN];
    const dst = nodes[srcL + 1][dstN];
    nnParticles.push({
      sx: src.x, sy: src.y,
      dx: dst.x, dy: dst.y,
      progress: 0,
      speed: 0.01 + Math.random() * 0.015,
      size: 1.5 + Math.random() * 1.0,
    });
  }

  nnParticles = nnParticles.filter(p => p.progress <= 1);
  for (const p of nnParticles) {
    p.progress += p.speed;
    const x = p.sx + (p.dx - p.sx) * p.progress;
    const y = p.sy + (p.dy - p.sy) * p.progress;

    ctx.beginPath();
    ctx.arc(x, y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }

  // Nodes
  for (let l = 0; l < numLayers; l++) {
    for (const node of nodes[l]) {
      const r = 3 + node.pulse * 1.5 + node.wavePulse * 3;
      const brightness = 0.3 + node.pulse * 0.5 + node.wavePulse * 0.5;

      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 255, 255, ${brightness})`;
      ctx.fill();
    }
  }

  // Labels
  ctx.textAlign = 'center';
  for (let l = 0; l < numLayers; l++) {
    const x = padX + l * layerSpacing;
    ctx.fillStyle = '#a1a1aa';
    ctx.font = '500 10px JetBrains Mono, monospace';
    ctx.fillText(layerLabels[l], x, H - 24);
  }

  if (nnPulseWave >= 0) {
    nnPulseWave += 0.15;
    if (nnPulseWave > numLayers + 2) nnPulseWave = -1;
  }

  if (nnAnimating) {
    nnAnimFrame = requestAnimationFrame(renderNetwork);
  }
}

// Init
loadData();
</script>
</body>
</html>"""



# ==============================================================================
#  HTTP Server with API endpoints
# ==============================================================================

class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the dashboard and API endpoints."""

    model = None
    config = None
    analysis = None

    def do_GET(self):
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())

        elif self.path == '/api/analysis':
            self.send_json(self.analysis)

        elif self.path.startswith('/api/attention'):
            try:
                text = self.path.split('text=')[-1] if 'text=' in self.path else "Hello world"
                from urllib.parse import unquote
                text = unquote(text)
                data = get_attention_patterns(
                    self.__class__.model, text,
                    self.__class__.config, device="cpu",
                )
                self.send_json(data)
            except Exception as e:
                self.send_json({"error": str(e), "tokens": [], "patterns": []})

        elif self.path.startswith('/api/activations'):
            try:
                text = self.path.split('text=')[-1] if 'text=' in self.path else "Hello"
                from urllib.parse import unquote
                text = unquote(text)
                data = get_activation_flow(
                    self.__class__.model, text,
                    self.__class__.config, device="cpu",
                )
                self.send_json(data)
            except Exception as e:
                self.send_json({"error": str(e)})

        else:
            self.send_error(404)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress log noise


def main():
    parser = argparse.ArgumentParser(
        description="superGPT Neural Network Visualizer",
        epilog="Open http://localhost:8050 in your browser after starting.",
    )
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=8050, help="Server port (default: 8050)")
    parser.add_argument("--device", default="cpu", help="Device for inference (default: cpu)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ⚡ superGPT Neural Network Visualizer")
    print(f"{'='*60}")
    print(f"  Loading: {args.checkpoint}")

    model, config, analysis = analyze_model(args.checkpoint, args.device)

    # Set class-level references
    VisualizerHandler.model = model
    VisualizerHandler.config = config
    VisualizerHandler.analysis = analysis

    arch = analysis["architecture"]
    print(f"  Model: {arch['total_params']/1e6:.1f}M params, {arch['n_layers']}L, {arch['attention_type']}")
    print(f"  Val Loss: {analysis['checkpoint'].get('best_val_loss', 'N/A')}")
    print(f"{'='*60}")
    print(f"\n  🌐 Dashboard: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(('0.0.0.0', args.port), VisualizerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
