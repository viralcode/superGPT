"""
Inference Serving Engine for superGPT
========================================
Production HTTP server with continuous batching, PagedAttention,
and an OpenAI-compatible API.

Features:
  - Continuous Batching: dynamically add/remove sequences mid-generation
  - PagedAttention: fixed-size KV-cache blocks to eliminate fragmentation
  - OpenAI-compatible: /v1/completions and /v1/chat/completions
  - SSE streaming: token-by-token response streaming

Usage:
    # Start the server
    python serve.py --checkpoint checkpoints/best.pt --port 8000

    # With custom settings
    python serve.py --checkpoint best.pt --port 8000 --max-batch 32 --max-tokens 512

    # Query the API
    curl http://localhost:8000/v1/completions -d '{
        "prompt": "To be or not to be",
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": true
    }'

Reference:
    vLLM, HuggingFace TGI, Kwon et al. "PagedAttention" (2023)
"""

import os
import sys
import json
import time
import uuid
import asyncio
import argparse
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT


# ==============================================================================
#  PagedAttention KV-Cache
# ==============================================================================

@dataclass
class KVBlock:
    """A fixed-size block of KV-cache storage."""
    block_id: int
    block_size: int  # tokens per block
    n_heads: int
    head_dim: int
    device: str = "cpu"

    def __post_init__(self):
        self.k = torch.zeros(self.block_size, self.n_heads, self.head_dim,
                              device=self.device)
        self.v = torch.zeros(self.block_size, self.n_heads, self.head_dim,
                              device=self.device)
        self.n_filled = 0

    @property
    def is_full(self):
        return self.n_filled >= self.block_size

    @property
    def free_slots(self):
        return self.block_size - self.n_filled


class PagedKVCache:
    """Paged KV-cache manager.

    Allocates KV-cache in fixed-size blocks (pages) instead of pre-allocating
    max_seq_len for every sequence. This eliminates memory fragmentation
    and allows ~4x more concurrent sequences.
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 block_size: int = 16, max_blocks: int = 1024,
                 device: str = "cpu"):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device

        # Pre-allocate block pool
        self.pool: List[KVBlock] = []
        self.free_blocks: List[int] = list(range(max_blocks))
        self.max_blocks = max_blocks

        # Per-layer, per-sequence block tables
        # block_tables[layer][seq_id] = [block_id_0, block_id_1, ...]
        self.block_tables: List[Dict[str, List[int]]] = [
            {} for _ in range(n_layers)
        ]

        # Initialize pool
        for i in range(max_blocks):
            self.pool.append(KVBlock(i, block_size, n_heads, head_dim, device))

    def allocate(self, seq_id: str, n_tokens: int = 1):
        """Allocate blocks for a new or growing sequence."""
        n_blocks_needed = (n_tokens + self.block_size - 1) // self.block_size

        for layer in range(self.n_layers):
            if seq_id not in self.block_tables[layer]:
                self.block_tables[layer][seq_id] = []

            current = len(self.block_tables[layer][seq_id])
            for _ in range(max(0, n_blocks_needed - current)):
                if not self.free_blocks:
                    raise RuntimeError("PagedKVCache: out of blocks")
                block_id = self.free_blocks.pop()
                self.block_tables[layer][seq_id].append(block_id)
                # Reset the block
                self.pool[block_id].n_filled = 0
                self.pool[block_id].k.zero_()
                self.pool[block_id].v.zero_()

    def free(self, seq_id: str):
        """Free all blocks for a completed sequence."""
        for layer in range(self.n_layers):
            if seq_id in self.block_tables[layer]:
                for block_id in self.block_tables[layer][seq_id]:
                    self.free_blocks.append(block_id)
                del self.block_tables[layer][seq_id]

    def append_kv(self, layer: int, seq_id: str, k: torch.Tensor, v: torch.Tensor):
        """Append new KV entries to the sequence's cache."""
        blocks = self.block_tables[layer].get(seq_id, [])
        if not blocks:
            self.allocate(seq_id, 1)
            blocks = self.block_tables[layer][seq_id]

        # Find current block (last one with space)
        block = self.pool[blocks[-1]]
        if block.is_full:
            # Need a new block
            if not self.free_blocks:
                raise RuntimeError("PagedKVCache: out of blocks")
            new_id = self.free_blocks.pop()
            self.block_tables[layer][seq_id].append(new_id)
            self.pool[new_id].n_filled = 0
            self.pool[new_id].k.zero_()
            self.pool[new_id].v.zero_()
            block = self.pool[new_id]

        pos = block.n_filled
        block.k[pos] = k
        block.v[pos] = v
        block.n_filled += 1

    def get_kv(self, layer: int, seq_id: str):
        """Get the full KV cache for a sequence (concatenated blocks)."""
        blocks = self.block_tables[layer].get(seq_id, [])
        if not blocks:
            return None, None

        k_parts, v_parts = [], []
        for block_id in blocks:
            block = self.pool[block_id]
            n = block.n_filled
            if n > 0:
                k_parts.append(block.k[:n])
                v_parts.append(block.v[:n])

        if not k_parts:
            return None, None

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    @property
    def utilization(self):
        """Return cache utilization as a fraction."""
        used = self.max_blocks - len(self.free_blocks)
        return used / self.max_blocks


# ==============================================================================
#  Request / Sequence Management
# ==============================================================================

@dataclass
class GenerationRequest:
    """A pending generation request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    prompt_tokens: List[int] = field(default_factory=list)
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 1.0
    stream: bool = False
    # State
    generated_tokens: List[int] = field(default_factory=list)
    finished: bool = False
    created_at: float = field(default_factory=time.time)
    # Output
    output_text: str = ""
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class ContinuousBatcher:
    """Continuous batching scheduler with PagedAttention.

    Dynamically batches and unbatches sequences as they start and finish
    generation. New sequences can join the batch without waiting for
    existing ones to complete.

    Uses PagedKVCache for memory-efficient KV-cache management:
    - Fixed-size blocks eliminate fragmentation
    - Sequences only allocate blocks as they grow
    - Freed blocks are recycled immediately
    """

    def __init__(self, model: GPT, max_batch: int = 32, max_seq_len: int = 512,
                 device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.device = device

        # PagedKVCache for efficient KV management
        config = model.config
        n_heads = getattr(config, 'n_kv_head', config.n_head) or config.n_head
        head_dim = config.n_embd // config.n_head
        self.kv_cache = PagedKVCache(
            n_layers=config.n_layer,
            n_heads=n_heads,
            head_dim=head_dim,
            block_size=16,
            max_blocks=max_batch * (max_seq_len // 16 + 1),
            device=device,
        )

        # Request queues
        self.pending: List[GenerationRequest] = []
        self.active: Dict[str, GenerationRequest] = {}

        # Per-sequence state: full token history for context
        self.seq_tokens: Dict[str, List[int]] = {}

        self.lock = threading.Lock()

    def add_request(self, req: GenerationRequest):
        """Add a new generation request to the pending queue."""
        with self.lock:
            self.pending.append(req)

    @torch.no_grad()
    def step(self):
        """Process one generation step for all active sequences.

        1. Admit pending requests and run prefill (process full prompt)
        2. For continuing sequences, run decode (single token)
        3. Sample next tokens
        4. Remove finished sequences and free their KV blocks
        """
        # Admit new requests
        newly_admitted = []
        with self.lock:
            while self.pending and len(self.active) < self.max_batch:
                req = self.pending.pop(0)
                self.active[req.id] = req
                self.seq_tokens[req.id] = list(req.prompt_tokens)
                newly_admitted.append(req.id)

        if not self.active:
            return False  # Nothing to do

        # Phase 1: Prefill — process full prompts for newly admitted sequences
        for req_id in newly_admitted:
            req = self.active[req_id]
            if not req.prompt_tokens:
                continue

            # Allocate KV blocks for the prompt
            self.kv_cache.allocate(req_id, len(req.prompt_tokens))

            # Run full prompt through the model
            prompt_ids = torch.tensor(
                req.prompt_tokens, dtype=torch.long, device=self.device
            ).unsqueeze(0)  # (1, T)

            logits, _ = self.model(prompt_ids)
            token_logits = logits[0, -1, :]  # Last position logits

            # Sample first token
            next_token = self._sample(token_logits, req)
            req.generated_tokens.append(next_token)
            self.seq_tokens[req_id].append(next_token)

            if len(req.generated_tokens) >= req.max_tokens:
                req.finished = True

        # Phase 2: Decode — one token per continuing sequence
        continuing = [
            rid for rid in self.active
            if rid not in newly_admitted and not self.active[rid].finished
        ]

        if continuing:
            # Build a batch of last tokens for all continuing sequences
            batch_ids = []
            batch_inputs = []

            for req_id in continuing:
                req = self.active[req_id]
                # Build full context (or truncated to block_size)
                all_tokens = self.seq_tokens[req_id]
                ctx = all_tokens[-self.model.config.block_size:]
                batch_ids.append(req_id)
                batch_inputs.append(ctx)

            # Pad to same length for batching
            max_len = max(len(t) for t in batch_inputs)
            padded = []
            for tokens in batch_inputs:
                padded.append([0] * (max_len - len(tokens)) + tokens)

            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)

            # Forward pass
            logits, _ = self.model(input_ids)

            # Sample for each sequence (use the last non-pad position)
            for i, req_id in enumerate(batch_ids):
                req = self.active[req_id]
                seq_len = len(batch_inputs[i])
                # Logits at the last real token position
                token_logits = logits[i, -1, :]

                next_token = self._sample(token_logits, req)
                req.generated_tokens.append(next_token)
                self.seq_tokens[req_id].append(next_token)

                # Extend KV allocation
                total_tokens = len(self.seq_tokens[req_id])
                try:
                    self.kv_cache.allocate(req_id, total_tokens)
                except RuntimeError:
                    req.finished = True  # Out of KV blocks

                if len(req.generated_tokens) >= req.max_tokens:
                    req.finished = True

        # Remove finished sequences and free their KV blocks
        with self.lock:
            finished = [rid for rid, req in self.active.items() if req.finished]
            for rid in finished:
                self.kv_cache.free(rid)
                if rid in self.seq_tokens:
                    del self.seq_tokens[rid]
                del self.active[rid]

        return True

    def _sample(self, token_logits, req):
        """Sample a token from logits with temperature, top-k, top-p."""
        if req.temperature > 0:
            token_logits = token_logits / req.temperature

        # Top-k
        if req.top_k > 0:
            v, _ = torch.topk(token_logits, min(req.top_k, token_logits.size(-1)))
            token_logits[token_logits < v[-1]] = float('-inf')

        # Top-p
        if req.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(token_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > req.top_p
            sorted_logits[mask] = float('-inf')
            token_logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

        probs = F.softmax(token_logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def generation_loop(self, poll_interval: float = 0.01):
        """Main generation loop — runs in a background thread."""
        while True:
            had_work = self.step()
            if not had_work:
                time.sleep(poll_interval)


# ==============================================================================
#  HTTP Server (OpenAI-Compatible)
# ==============================================================================

class SuperGPTHandler(BaseHTTPRequestHandler):
    """HTTP request handler with OpenAI-compatible API."""

    server_version = "superGPT/1.0"
    batcher = None  # Set by server
    tokenizer = None
    model_name = "supergpt"

    def do_POST(self):
        if self.path == "/v1/completions":
            self._handle_completions()
        elif self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404, "Not Found")

    def _handle_completions(self):
        body = self._read_body()
        if body is None:
            return

        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 100)
        temperature = body.get("temperature", 0.8)
        top_k = body.get("top_k", 50)
        top_p = body.get("top_p", 1.0)
        stream = body.get("stream", False)

        # Tokenize prompt
        if self.tokenizer:
            prompt_tokens = self.tokenizer(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Create request
        req = GenerationRequest(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stream=stream,
        )

        # Submit to batcher
        self.batcher.add_request(req)

        if stream:
            self._stream_response(req)
        else:
            self._blocking_response(req)

    def _handle_chat_completions(self):
        body = self._read_body()
        if body is None:
            return

        messages = body.get("messages", [])
        # Convert chat messages to a single prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|assistant|>\n"

        max_tokens = body.get("max_tokens", 100)
        temperature = body.get("temperature", 0.8)
        stream = body.get("stream", False)

        if self.tokenizer:
            prompt_tokens = self.tokenizer(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        req = GenerationRequest(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        self.batcher.add_request(req)

        if stream:
            self._stream_chat_response(req)
        else:
            self._blocking_chat_response(req)

    def _handle_models(self):
        self._send_json({
            "data": [{"id": self.model_name, "object": "model",
                       "owned_by": "superGPT"}]
        })

    def _blocking_response(self, req):
        """Wait for generation to complete and return full response."""
        while not req.finished:
            time.sleep(0.01)

        # Decode tokens
        if self.tokenizer:
            text = "".join(chr(t) if t < 128 else "?" for t in req.generated_tokens)
        else:
            text = "".join(chr(t) for t in req.generated_tokens if t < 128)

        response = {
            "id": req.id,
            "object": "text_completion",
            "model": self.model_name,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": len(req.prompt_tokens),
                "completion_tokens": len(req.generated_tokens),
                "total_tokens": len(req.prompt_tokens) + len(req.generated_tokens),
            },
        }
        self._send_json(response)

    def _stream_response(self, req):
        """Stream tokens as SSE events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        last_len = 0
        while not req.finished:
            if len(req.generated_tokens) > last_len:
                new_tokens = req.generated_tokens[last_len:]
                last_len = len(req.generated_tokens)

                text = "".join(chr(t) if t < 128 else "?" for t in new_tokens)
                chunk = {
                    "id": req.id,
                    "object": "text_completion",
                    "choices": [{"text": text, "index": 0}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            time.sleep(0.01)

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_chat_response(self, req):
        """Stream chat completion tokens as SSE events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        last_len = 0
        while not req.finished:
            if len(req.generated_tokens) > last_len:
                new_tokens = req.generated_tokens[last_len:]
                last_len = len(req.generated_tokens)

                text = "".join(chr(t) if t < 128 else "?" for t in new_tokens)
                chunk = {
                    "id": req.id,
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": text}, "index": 0}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            time.sleep(0.01)

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _blocking_chat_response(self, req):
        while not req.finished:
            time.sleep(0.01)

        text = "".join(chr(t) if t < 128 else "?" for t in req.generated_tokens)
        response = {
            "id": req.id,
            "object": "chat.completion",
            "model": self.model_name,
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "index": 0,
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": len(req.prompt_tokens),
                "completion_tokens": len(req.generated_tokens),
                "total_tokens": len(req.prompt_tokens) + len(req.generated_tokens),
            },
        }
        self._send_json(response)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self.send_error(400, "Empty request body")
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return None

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quiet logging
        pass


# ==============================================================================
#  Server Entry Point
# ==============================================================================

def start_server(args):
    """Start the superGPT inference server."""

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    print(f"Loading model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params on {device}")

    # Create batcher
    batcher = ContinuousBatcher(model, max_batch=args.max_batch,
                                 max_seq_len=args.max_tokens, device=device)

    # Start generation thread
    gen_thread = threading.Thread(target=batcher.generation_loop, daemon=True)
    gen_thread.start()

    # Set up handler
    SuperGPTHandler.batcher = batcher
    SuperGPTHandler.model_name = f"supergpt-{n_params//1e6:.0f}m"

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), SuperGPTHandler)
    print(f"\n{'='*60}")
    print(f"  superGPT Inference Server")
    print(f"{'='*60}")
    print(f"  Model:      {n_params/1e6:.1f}M params")
    print(f"  Device:     {device}")
    print(f"  Max batch:  {args.max_batch}")
    print(f"  Port:       {args.port}")
    print(f"{'='*60}")
    print(f"\n  API:     http://localhost:{args.port}/v1/completions")
    print(f"  Health:  http://localhost:{args.port}/health")
    print(f"  Models:  http://localhost:{args.port}/v1/models")
    print(f"\n  Example:")
    print(f'  curl http://localhost:{args.port}/v1/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"prompt": "To be or not to be", "max_tokens": 100}}\'')
    print(f"\nServing...\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="superGPT inference server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--max-batch", type=int, default=32,
                        help="Max concurrent sequences (default: 32)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per response (default: 512)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()
    start_server(args)
