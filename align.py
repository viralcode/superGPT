"""
DPO Alignment Script (Direct Preference Optimization)
======================================================
Fine-tunes a pre-trained GPT model using DPO — the modern alternative to RLHF.
Used by LLaMA 3, Mistral, Zephyr, and other production LLMs.

DPO directly optimizes a policy from preference pairs, without needing a
separate reward model or PPO. It's simpler, more stable, and achieves
comparable results to full RLHF.

Usage:
    # Align a trained model with preference data:
    python align.py --checkpoint checkpoints/best.pt --data preferences.jsonl

    # Adjust alignment strength (beta):
    python align.py --checkpoint checkpoints/best.pt --data prefs.jsonl --beta 0.5

Preference Data Format (JSONL):
    Each line is a JSON object with:
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France is a country in Europe."
    }

The model learns to prefer "chosen" over "rejected" responses.

Reference: Rafailov et al., "Direct Preference Optimization:
Your Language Model is Secretly a Reward Model" (2023)
"""

import os
import sys
import json
import copy
import argparse
import pickle
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT


def load_model(checkpoint_path: str, device: str):
    """Load a pre-trained GPT model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = checkpoint["model_config"]
    config = GPTConfig(**config_dict)

    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    return model, config, checkpoint


def load_tokenizer(data_dir: str):
    """Load the tokenizer used during training."""
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"Error: Tokenizer metadata not found at {meta_path}")
        sys.exit(1)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    tokenizer_type = meta.get("tokenizer_type", "char")

    if tokenizer_type == "char":
        from data.prepare_data import CharTokenizer
        tokenizer = CharTokenizer()
        tokenizer.vocab_size = meta["vocab_size"]
        tokenizer.char_to_idx = meta["char_to_idx"]
        tokenizer.idx_to_char = meta["idx_to_char"]
    elif tokenizer_type == "tiktoken":
        from data.prepare_data import TiktokenWrapper
        tokenizer = TiktokenWrapper()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return tokenizer


def load_preferences(data_path: str):
    """Load preference pairs from a JSONL file."""
    preferences = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            assert "prompt" in item, "Each preference must have a 'prompt' field"
            assert "chosen" in item, "Each preference must have a 'chosen' field"
            assert "rejected" in item, "Each preference must have a 'rejected' field"
            preferences.append(item)

    print(f"Loaded {len(preferences)} preference pairs")
    return preferences


def compute_log_probs(model, input_ids, target_ids, device):
    """Compute per-token log probabilities for a sequence.

    Args:
        model: GPT model
        input_ids: input token IDs (1D tensor)
        target_ids: target token IDs (1D tensor, shifted by 1)
        device: computation device

    Returns:
        total log probability of the target sequence
    """
    input_ids = input_ids.unsqueeze(0).to(device)
    target_ids = target_ids.unsqueeze(0).to(device)

    logits, _ = model(input_ids)
    # logits shape: (1, T, vocab_size) when targets not provided via model()
    # But we passed no targets, so it only returns last token logits.
    # We need full logits, so use forward with targets=None but full sequence
    logits = model.lm_head(model.transformer.ln_f(
        _forward_blocks(model, input_ids)
    ))

    # Compute log probs for each target token
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log probs of the actual target tokens
    target_log_probs = log_probs.squeeze(0).gather(1, target_ids.squeeze(0).unsqueeze(-1)).squeeze(-1)

    return target_log_probs.sum()


def _forward_blocks(model, idx):
    """Run input through transformer blocks (without LM head)."""
    B, T = idx.size()
    tok_emb = model.transformer.wte(idx)

    if model.transformer.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = model.transformer.drop(tok_emb + model.transformer.wpe(pos))
    else:
        x = model.transformer.drop(tok_emb)

    for block in model.transformer.h:
        x, _ = block(x)

    return x


def dpo_loss(
    policy_model,
    ref_model,
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float,
    device: str,
    max_length: int,
):
    """Compute DPO loss for a single preference pair.

    DPO Loss = -log(σ(β * (log π(y_w|x) - log π_ref(y_w|x)
                          - log π(y_l|x) + log π_ref(y_l|x))))

    Where:
        π = policy model (being trained)
        π_ref = reference model (frozen copy of initial policy)
        y_w = chosen/winning response
        y_l = rejected/losing response
        x = prompt
        β = temperature parameter (controls alignment strength)
    """
    # Tokenize
    prompt_ids = tokenizer.encode(prompt)
    chosen_ids = tokenizer.encode(prompt + chosen)
    rejected_ids = tokenizer.encode(prompt + rejected)

    # Truncate to max length
    chosen_ids = chosen_ids[:max_length]
    rejected_ids = rejected_ids[:max_length]

    prompt_len = len(prompt_ids)

    # Create input/target pairs (teacher forcing)
    chosen_input = torch.tensor(chosen_ids[:-1], dtype=torch.long)
    chosen_target = torch.tensor(chosen_ids[1:], dtype=torch.long)
    rejected_input = torch.tensor(rejected_ids[:-1], dtype=torch.long)
    rejected_target = torch.tensor(rejected_ids[1:], dtype=torch.long)

    # Only compute log probs over the response tokens (not the prompt)
    # Policy model log probs
    with torch.no_grad():
        ref_chosen_logprob = _sequence_log_prob(
            ref_model, chosen_input, chosen_target, prompt_len, device
        )
        ref_rejected_logprob = _sequence_log_prob(
            ref_model, rejected_input, rejected_target, prompt_len, device
        )

    policy_chosen_logprob = _sequence_log_prob(
        policy_model, chosen_input, chosen_target, prompt_len, device
    )
    policy_rejected_logprob = _sequence_log_prob(
        policy_model, rejected_input, rejected_target, prompt_len, device
    )

    # DPO loss
    chosen_reward = beta * (policy_chosen_logprob - ref_chosen_logprob)
    rejected_reward = beta * (policy_rejected_logprob - ref_rejected_logprob)

    loss = -F.logsigmoid(chosen_reward - rejected_reward)

    # Metrics
    with torch.no_grad():
        reward_margin = (chosen_reward - rejected_reward).item()
        accuracy = float(chosen_reward > rejected_reward)

    return loss, reward_margin, accuracy


def _sequence_log_prob(model, input_ids, target_ids, prompt_len, device):
    """Compute sum of log probs over response tokens only."""
    input_ids = input_ids.unsqueeze(0).to(device)
    target_ids = target_ids.unsqueeze(0).to(device)

    # Full forward pass to get all logits
    x = _forward_blocks(model, input_ids)
    logits = model.lm_head(model.transformer.ln_f(x))

    # Log probs for each position
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather target token log probs
    token_log_probs = log_probs.squeeze(0).gather(
        1, target_ids.squeeze(0).unsqueeze(-1)
    ).squeeze(-1)

    # Only sum over response tokens (skip prompt)
    response_start = max(0, prompt_len - 1)  # -1 because input is shifted
    response_log_prob = token_log_probs[response_start:].sum()

    return response_log_prob


def align(args):
    """Main DPO alignment loop."""

    # ── Setup ─────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    policy_model, config, original_ckpt = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.data_dir)

    # Create frozen reference model (copy of initial policy)
    print("Creating frozen reference model...")
    ref_model = GPT(config)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load preference data
    if not os.path.exists(args.data):
        print(f"Error: Preference data not found at {args.data}")
        print(f"\nExpected JSONL format:")
        print(f'  {{"prompt": "...", "chosen": "...", "rejected": "..."}}')
        sys.exit(1)

    preferences = load_preferences(args.data)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Mixed precision
    if "cuda" in device:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DPO Alignment")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    print(f"  Beta:        {args.beta}")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Preferences: {len(preferences)}")
    print(f"{'='*60}\n")

    policy_model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        total_margin = 0
        total_accuracy = 0

        for i, pref in enumerate(preferences):
            with ctx:
                loss, margin, accuracy = dpo_loss(
                    policy_model, ref_model, tokenizer,
                    pref["prompt"], pref["chosen"], pref["rejected"],
                    beta=args.beta, device=device,
                    max_length=config.block_size,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_margin += margin
            total_accuracy += accuracy

            if (i + 1) % 10 == 0 or i == len(preferences) - 1:
                avg_loss = total_loss / (i + 1)
                avg_margin = total_margin / (i + 1)
                avg_acc = total_accuracy / (i + 1)
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {i+1}/{len(preferences)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Margin: {avg_margin:.4f} | "
                    f"Acc: {avg_acc:.1%}"
                )

    # ── Save aligned model ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "aligned.pt")

    checkpoint = {
        "model": policy_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": config.to_dict(),
        "iter_num": original_ckpt.get("iter_num", 0),
        "best_val_loss": original_ckpt.get("best_val_loss", float("inf")),
        "alignment": {
            "method": "dpo",
            "beta": args.beta,
            "epochs": args.epochs,
            "n_preferences": len(preferences),
        },
    }
    torch.save(checkpoint, ckpt_path)

    print(f"\n{'='*60}")
    print(f"  DPO Alignment Complete!")
    print(f"  Aligned checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    print(f"\nGenerate with: python generate.py --checkpoint {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Alignment for GPT model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to preference data (JSONL)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing tokenizer metadata")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO temperature (alignment strength, default: 0.1)")
    parser.add_argument("--lr", type=float, default=5e-7,
                        help="Learning rate for alignment (default: 5e-7)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of alignment epochs (default: 3)")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save aligned checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")

    args = parser.parse_args()
    align(args)
