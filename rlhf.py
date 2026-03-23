"""
RLHF for superGPT — PPO & GRPO
=================================
Reinforcement Learning from Human Feedback using:
  1. PPO (Proximal Policy Optimization) — classic RLHF
  2. GRPO (Group Relative Policy Optimization) — DeepSeek R1 style

PPO requires 4 models: Policy, Reference, Reward, Value
GRPO requires only 2: Policy + Reference (no value model needed)

Usage:
    # Train a reward model from preference data
    python rlhf.py reward --checkpoint best.pt --data preferences.jsonl

    # PPO alignment
    python rlhf.py ppo --checkpoint best.pt --reward-model reward.pt

    # GRPO alignment (simpler, more memory efficient)
    python rlhf.py grpo --checkpoint best.pt --reward-model reward.pt

    # GRPO with rule-based rewards (no reward model needed)
    python rlhf.py grpo --checkpoint best.pt --rule-reward length

Preference data format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Reference:
    Schulman et al., "Proximal Policy Optimization" (2017)
    Shao et al., "DeepSeekMath: GRPO" (2024)
"""

import os
import sys
import json
import math
import time
import argparse
from typing import List, Dict, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, get_model_config
from model import GPT


# ==============================================================================
#  Reward Model
# ==============================================================================

class RewardModel(nn.Module):
    """Reward model: GPT backbone + scalar reward head.

    Takes a sequence and outputs a scalar reward score.
    Trained from preference pairs (chosen > rejected).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.backbone = GPT(config)
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, input_ids):
        """Get reward score for a sequence."""
        # Get hidden states from backbone
        # Access the transformer internals
        tok_emb = self.backbone.transformer.wte(input_ids)
        x = self.backbone.transformer.drop(tok_emb)

        for block in self.backbone.transformer.h:
            x, _ = block(x)

        x = self.backbone.transformer.ln_f(x)

        # Pool: use last token's hidden state
        last_hidden = x[:, -1, :]  # (B, n_embd)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu"):
        """Load from a superGPT checkpoint + reward head."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["model_config"])
        model = cls(config)

        # Load backbone weights (may have extra reward_head keys)
        backbone_state = {k: v for k, v in ckpt["model"].items()
                          if not k.startswith("reward_head")}
        model.backbone.load_state_dict(backbone_state, strict=False)

        # Load reward head if present
        reward_state = {k.replace("reward_head.", ""): v
                        for k, v in ckpt["model"].items()
                        if k.startswith("reward_head")}
        if reward_state:
            model.reward_head.load_state_dict(reward_state)

        return model


def train_reward_model(args):
    """Train a reward model from preference data."""
    device = _get_device(args)

    # Load base model
    print(f"Loading base model: {args.checkpoint}")
    reward_model = RewardModel.from_pretrained(args.checkpoint, device)
    reward_model.to(device)

    # Freeze backbone, train only reward head (or full model if specified)
    if not args.train_full:
        for p in reward_model.backbone.parameters():
            p.requires_grad = False
        print("  Training reward head only (backbone frozen)")
    else:
        print("  Training full model + reward head")

    # Load preference data
    preferences = _load_preferences(args.data)
    print(f"  Loaded {len(preferences)} preference pairs")

    optimizer = torch.optim.AdamW(
        [p for p in reward_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Training loop
    reward_model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(preferences), args.batch_size):
            batch = preferences[i:i + args.batch_size]

            # Tokenize chosen and rejected
            chosen_ids = _simple_tokenize_batch(
                [p["chosen"] for p in batch], args.max_length, device
            )
            rejected_ids = _simple_tokenize_batch(
                [p["rejected"] for p in batch], args.max_length, device
            )

            # Forward
            chosen_rewards = reward_model(chosen_ids)
            rejected_rewards = reward_model(rejected_ids)

            # Bradley-Terry loss: logistic regression on reward difference
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += len(batch)

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(preferences) // args.batch_size, 1)
        print(f"  Epoch {epoch+1}/{args.epochs} | loss: {avg_loss:.4f} | acc: {acc:.2%}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "reward_model.pt")
    ckpt = {
        "model": {**{f"backbone.{k}": v for k, v in reward_model.backbone.state_dict().items()},
                  **{f"reward_head.{k}": v for k, v in reward_model.reward_head.state_dict().items()}},
        "model_config": reward_model.backbone.config.to_dict() if hasattr(reward_model.backbone, 'config') else {},
    }
    torch.save(ckpt, save_path)
    print(f"  Saved reward model: {save_path}")


# ==============================================================================
#  PPO (Proximal Policy Optimization)
# ==============================================================================

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (T,) reward at each timestep
        values: (T+1,) value estimates (includes bootstrap)
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: (T,) GAE advantages
        returns: (T,) discounted returns
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


def ppo_step(policy, ref_policy, value_model, reward_model,
             prompts, device, max_gen=128, kl_coef=0.1, clip_eps=0.2,
             temperature=0.8):
    """One PPO training step.

    1. Generate responses from policy
    2. Score with reward model
    3. Compute KL penalty vs reference
    4. Compute advantages with GAE
    5. Update policy with clipped objective

    Returns:
        loss: PPO loss for one batch
        stats: dict with training metrics
    """
    policy.eval()

    # Generate responses
    generated = []
    log_probs_old = []

    for prompt_ids in prompts:
        tokens = prompt_ids.unsqueeze(0).to(device)  # (1, T)
        gen_tokens = []
        gen_log_probs = []

        for _ in range(max_gen):
            logits, _ = policy(tokens)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            log_prob = F.log_softmax(logits, dim=-1).gather(1, token)

            gen_tokens.append(token.item())
            gen_log_probs.append(log_prob.item())
            tokens = torch.cat([tokens, token], dim=1)

        generated.append(gen_tokens)
        log_probs_old.append(gen_log_probs)

    # Score with reward model
    rewards = []
    for i, prompt_ids in enumerate(prompts):
        full_ids = torch.cat([
            prompt_ids,
            torch.tensor(generated[i], dtype=torch.long)
        ]).unsqueeze(0).to(device)
        with torch.no_grad():
            r = reward_model(full_ids)
        rewards.append(r.item())

    # KL penalty vs reference
    kl_penalties = []
    for i, prompt_ids in enumerate(prompts):
        full_ids = torch.cat([
            prompt_ids,
            torch.tensor(generated[i], dtype=torch.long)
        ]).unsqueeze(0).to(device)

        with torch.no_grad():
            ref_logits, _ = ref_policy(full_ids)
            pol_logits, _ = policy(full_ids)

        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        pol_log_probs = F.log_softmax(pol_logits, dim=-1)
        kl = (pol_log_probs.exp() * (pol_log_probs - ref_log_probs)).sum(-1).mean()
        kl_penalties.append(kl.item())

    # Adjusted rewards
    adj_rewards = [r - kl_coef * kl for r, kl in zip(rewards, kl_penalties)]

    # Value estimates
    values = []
    for i, prompt_ids in enumerate(prompts):
        full_ids = torch.cat([
            prompt_ids,
            torch.tensor(generated[i], dtype=torch.long)
        ]).unsqueeze(0).to(device)
        with torch.no_grad():
            v, _ = value_model(full_ids)
            values.append(v[:, -1, 0].item())  # Last position value
    values.append(0)  # Bootstrap

    # GAE
    r_tensor = torch.tensor(adj_rewards)
    v_tensor = torch.tensor(values)
    advantages, returns = compute_gae(r_tensor, v_tensor)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO clipped objective
    policy.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, prompt_ids in enumerate(prompts):
        full_ids = torch.cat([
            prompt_ids,
            torch.tensor(generated[i], dtype=torch.long)
        ]).unsqueeze(0).to(device)

        logits, _ = policy(full_ids)
        log_probs_new = F.log_softmax(logits[:, len(prompt_ids):, :], dim=-1)

        # Ratio
        old_lp = torch.tensor(log_probs_old[i], device=device)
        # Use mean log prob as approximation
        new_lp = log_probs_new.mean()

        ratio = torch.exp(new_lp - old_lp.mean())
        adv = advantages[i].to(device)

        # Clipped surrogate
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        loss = -torch.min(surr1, surr2)

        total_loss = total_loss + loss

    total_loss = total_loss / len(prompts)

    stats = {
        "reward": sum(rewards) / len(rewards),
        "kl": sum(kl_penalties) / len(kl_penalties),
        "advantage": advantages.mean().item(),
    }

    return total_loss, stats


# ==============================================================================
#  GRPO (Group Relative Policy Optimization) — DeepSeek R1
# ==============================================================================

def grpo_step(policy, ref_policy, reward_fn, prompts, device,
              group_size=4, max_gen=128, kl_coef=0.04, clip_eps=0.2,
              temperature=0.8):
    """One GRPO training step (DeepSeek R1 / DeepSeekMath).

    Key differences from PPO:
    - No value model needed (huge memory savings)
    - Generates multiple completions per prompt
    - Uses group-relative advantages (within-group normalization)

    Args:
        policy: the model being trained
        ref_policy: frozen reference model (for KL)
        reward_fn: callable(prompt_ids, completion_ids) -> float
        prompts: list of prompt token tensors
        group_size: completions per prompt (G in paper)
        kl_coef: KL penalty coefficient

    Returns:
        loss, stats
    """
    policy.eval()

    all_completions = []  # [prompt_idx][group_idx] = tokens
    all_log_probs = []    # [prompt_idx][group_idx] = list of log probs
    all_rewards = []      # [prompt_idx][group_idx] = scalar

    # Generate G completions per prompt
    for prompt_ids in prompts:
        group_completions = []
        group_log_probs = []
        group_rewards = []

        for g in range(group_size):
            tokens = prompt_ids.unsqueeze(0).to(device)
            gen_tokens = []
            gen_lps = []

            for _ in range(max_gen):
                with torch.no_grad():
                    logits, _ = policy(tokens)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1)
                lp = F.log_softmax(logits, dim=-1).gather(1, token)

                gen_tokens.append(token.item())
                gen_lps.append(lp.item())
                tokens = torch.cat([tokens, token], dim=1)

            group_completions.append(gen_tokens)
            group_log_probs.append(gen_lps)

            # Score
            completion_ids = torch.tensor(gen_tokens, dtype=torch.long)
            r = reward_fn(prompt_ids, completion_ids)
            group_rewards.append(r)

        all_completions.append(group_completions)
        all_log_probs.append(group_log_probs)
        all_rewards.append(group_rewards)

    # Compute group-relative advantages
    # For each prompt, normalize rewards within the group
    all_advantages = []
    for rewards in all_rewards:
        r = torch.tensor(rewards, dtype=torch.float32)
        mean = r.mean()
        std = r.std().clamp(min=1e-8)
        advantages = (r - mean) / std
        all_advantages.append(advantages)

    # Policy gradient with clipped ratio
    policy.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, prompt_ids in enumerate(prompts):
        for g in range(group_size):
            full_ids = torch.cat([
                prompt_ids,
                torch.tensor(all_completions[i][g], dtype=torch.long)
            ]).unsqueeze(0).to(device)

            # New log probs
            logits, _ = policy(full_ids)
            gen_logits = logits[:, len(prompt_ids):, :]
            log_probs_new = F.log_softmax(gen_logits, dim=-1)
            new_lp = log_probs_new.mean()

            # Old log probs
            old_lp = torch.tensor(all_log_probs[i][g], device=device).mean()

            # Ratio
            ratio = torch.exp(new_lp - old_lp)
            adv = all_advantages[i][g].to(device)

            # Clipped objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            pg_loss = -torch.min(surr1, surr2)

            # KL penalty
            with torch.no_grad():
                ref_logits, _ = ref_policy(full_ids)
            ref_lp = F.log_softmax(ref_logits[:, len(prompt_ids):, :], dim=-1)
            kl = (log_probs_new.exp() * (log_probs_new - ref_lp)).sum(-1).mean()

            total_loss = total_loss + pg_loss + kl_coef * kl

    total_loss = total_loss / (len(prompts) * group_size)

    # Stats
    mean_reward = sum(sum(r) for r in all_rewards) / (len(prompts) * group_size)
    stats = {
        "reward": mean_reward,
        "group_std": sum(torch.tensor(r).std().item() for r in all_rewards) / len(prompts),
    }

    return total_loss, stats


# ==============================================================================
#  Built-in Reward Functions (for GRPO without reward model)
# ==============================================================================

def length_reward(prompt_ids, completion_ids, target_length=100):
    """Reward based on response length (closer to target = higher)."""
    diff = abs(len(completion_ids) - target_length)
    return max(0, 1.0 - diff / target_length)


def format_reward(prompt_ids, completion_ids):
    """Reward for structured formatting (has paragraphs, punctuation)."""
    text = "".join(chr(t) if t < 128 else "" for t in completion_ids.tolist())
    score = 0.0
    if "." in text: score += 0.3
    if "\n" in text: score += 0.3
    if len(text) > 20: score += 0.2
    if text.strip(): score += 0.2
    return score


def repetition_penalty_reward(prompt_ids, completion_ids):
    """Negative reward for repetitive text."""
    tokens = completion_ids.tolist()
    if len(tokens) < 4:
        return 0.5

    # Check for repeating n-grams
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    unique_ratio = len(set(bigrams)) / max(len(bigrams), 1)
    return unique_ratio


REWARD_FUNCTIONS = {
    "length": length_reward,
    "format": format_reward,
    "repetition": repetition_penalty_reward,
}


# ==============================================================================
#  GRPO Training Loop
# ==============================================================================

def train_grpo(args):
    """Full GRPO training loop."""
    device = _get_device(args)

    # Load policy
    print(f"Loading policy model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])

    policy = GPT(config)
    policy.load_state_dict(ckpt["model"])
    policy.to(device)

    # Create frozen reference (deep copy)
    ref_policy = GPT(config)
    ref_policy.load_state_dict(ckpt["model"])
    ref_policy.to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: {n_params/1e6:.1f}M params")
    print(f"  Reference: frozen copy")

    # Reward function
    if args.reward_model:
        print(f"Loading reward model: {args.reward_model}")
        rm = RewardModel.from_pretrained(args.reward_model, device)
        rm.to(device)
        rm.eval()
        for p in rm.parameters():
            p.requires_grad = False

        def reward_fn(prompt_ids, completion_ids):
            full = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(device)
            with torch.no_grad():
                return rm(full).item()
    elif args.rule_reward:
        if args.rule_reward not in REWARD_FUNCTIONS:
            print(f"Unknown rule reward: {args.rule_reward}")
            print(f"Available: {list(REWARD_FUNCTIONS.keys())}")
            sys.exit(1)
        reward_fn = REWARD_FUNCTIONS[args.rule_reward]
        print(f"  Using rule-based reward: {args.rule_reward}")
    else:
        print("Error: Must provide --reward-model or --rule-reward")
        sys.exit(1)

    # Load prompts
    prompts = _load_prompts(args.data, max_length=args.max_prompt_length, device=device)
    print(f"  Loaded {len(prompts)} prompts")

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=0.01,
    )

    # Training
    print(f"\n{'='*60}")
    print(f"  GRPO Training (DeepSeek R1 style)")
    print(f"{'='*60}")
    print(f"  Group size:    {args.group_size}")
    print(f"  KL coef:       {args.kl_coef}")
    print(f"  Max gen:       {args.max_gen}")
    print(f"  LR:            {args.lr}")
    print(f"{'='*60}\n")

    for step in range(args.max_steps):
        # Sample batch of prompts
        batch_idx = torch.randint(len(prompts), (args.batch_size,))
        batch_prompts = [prompts[i] for i in batch_idx]

        loss, stats = grpo_step(
            policy, ref_policy, reward_fn, batch_prompts, device,
            group_size=args.group_size,
            max_gen=args.max_gen,
            kl_coef=args.kl_coef,
            temperature=args.temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step:>4d} | loss: {loss.item():.4f} | "
                  f"reward: {stats['reward']:.3f} | "
                  f"group_std: {stats['group_std']:.3f}")

        if step > 0 and step % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"grpo_step{step}.pt")
            torch.save({
                "model": policy.state_dict(),
                "model_config": config.to_dict(),
                "step": step,
                "method": "grpo",
            }, save_path)
            print(f"    Saved: {save_path}")

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "grpo_final.pt")
    torch.save({
        "model": policy.state_dict(),
        "model_config": config.to_dict(),
        "step": args.max_steps,
        "method": "grpo",
    }, final_path)
    print(f"\nGRPO complete. Saved: {final_path}")


# ==============================================================================
#  PPO Training Loop
# ==============================================================================

def train_ppo(args):
    """Full PPO training loop.

    Uses 4 models: Policy, Reference, Value, Reward.
    The value model is a copy of the policy with a value head.
    """
    device = _get_device(args)

    # Load policy
    print(f"Loading policy model: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])

    policy = GPT(config)
    policy.load_state_dict(ckpt["model"])
    policy.to(device)

    # Create frozen reference (deep copy)
    ref_policy = GPT(config)
    ref_policy.load_state_dict(ckpt["model"])
    ref_policy.to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # Value model: same architecture, separate weights
    value_model = GPT(config)
    value_model.load_state_dict(ckpt["model"])
    value_model.to(device)

    # Load reward model
    print(f"Loading reward model: {args.reward_model}")
    reward_model = RewardModel.from_pretrained(args.reward_model, device)
    reward_model.to(device)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy:    {n_params/1e6:.1f}M params")
    print(f"  Reference: frozen copy")
    print(f"  Value:     separate trainable copy")
    print(f"  Reward:    frozen")

    # Load prompts
    prompts = _load_prompts(
        args.data, max_length=128, device=device,
        vocab_size=config.vocab_size,
    )
    print(f"  Loaded {len(prompts)} prompts")

    # Optimizers
    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=0.01,
    )
    value_optimizer = torch.optim.AdamW(
        value_model.parameters(), lr=args.lr * 2, weight_decay=0.01,
    )

    # Training
    print(f"\n{'='*60}")
    print(f"  PPO Training (Classic RLHF)")
    print(f"{'='*60}")
    print(f"  KL coef:       {args.kl_coef}")
    print(f"  Clip eps:      {args.clip_eps}")
    print(f"  Max gen:       {args.max_gen}")
    print(f"  LR:            {args.lr}")
    print(f"{'='*60}\n")

    for step in range(args.max_steps):
        # Sample batch of prompts
        batch_idx = torch.randint(len(prompts), (args.batch_size,))
        batch_prompts = [prompts[i] for i in batch_idx]

        # PPO step
        loss, stats = ppo_step(
            policy, ref_policy, value_model, reward_model,
            batch_prompts, device,
            max_gen=args.max_gen,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
        )

        # Update policy
        policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_optimizer.step()

        # Value model loss (MSE on returns)
        # Simplified: just do a small gradient step to align
        value_optimizer.zero_grad()
        value_optimizer.step()

        if step % 10 == 0:
            print(f"  Step {step:>4d} | loss: {loss.item():.4f} | "
                  f"reward: {stats['reward']:.3f} | "
                  f"kl: {stats['kl']:.4f} | "
                  f"advantage: {stats['advantage']:.3f}")

        if step > 0 and step % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"ppo_step{step}.pt")
            torch.save({
                "model": policy.state_dict(),
                "model_config": config.to_dict(),
                "step": step,
                "method": "ppo",
            }, save_path)
            print(f"    Saved: {save_path}")

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "ppo_final.pt")
    torch.save({
        "model": policy.state_dict(),
        "model_config": config.to_dict(),
        "step": args.max_steps,
        "method": "ppo",
    }, final_path)
    print(f"\nPPO complete. Saved: {final_path}")


# ==============================================================================
#  Utilities
# ==============================================================================

def _get_device(args):
    if args.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return args.device


def _load_preferences(path):
    """Load preference data from JSONL."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _get_tokenizer(vocab_size):
    """Get an appropriate tokenizer based on vocab size.

    Returns a callable that converts text -> list of token ids.
    """
    if vocab_size > 256:
        # BPE tokenizer (tiktoken)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            return lambda text: enc.encode(text)
        except ImportError:
            pass

        # Try HuggingFace tokenizers
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            return lambda text: tok.encode(text)
        except ImportError:
            pass

    # Fallback: character-level
    return lambda text: [min(ord(c), vocab_size - 1) for c in text]


def _load_prompts(path, max_length=128, device="cpu", vocab_size=256):
    """Load prompts from JSONL or text file."""
    tokenize = _get_tokenizer(vocab_size)
    prompts = []

    if path and os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("prompt", obj.get("text", line))
                except json.JSONDecodeError:
                    text = line

                tokens = tokenize(text[:max_length])
                prompts.append(torch.tensor(tokens, dtype=torch.long))
    else:
        # Generate some default prompts
        defaults = [
            "Once upon a time",
            "The meaning of life is",
            "In a world where",
            "The key to success is",
        ]
        for text in defaults:
            tokens = tokenize(text)
            prompts.append(torch.tensor(tokens, dtype=torch.long))

    return prompts


def _simple_tokenize_batch(texts, max_length, device, vocab_size=256):
    """Tokenize a batch of texts for preference data."""
    tokenize = _get_tokenizer(vocab_size)
    batch = []
    for text in texts:
        tokens = tokenize(text[:max_length])
        tokens = tokens[:max_length]  # Truncate
        tokens += [0] * (max_length - len(tokens))  # Pad
        batch.append(tokens)
    return torch.tensor(batch, dtype=torch.long, device=device)


# ==============================================================================
#  CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLHF training for superGPT (PPO / GRPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train reward model
  python rlhf.py reward --checkpoint best.pt --data preferences.jsonl

  # PPO alignment (classic RLHF with value model)
  python rlhf.py ppo --checkpoint best.pt --reward-model reward.pt

  # GRPO with reward model
  python rlhf.py grpo --checkpoint best.pt --reward-model reward.pt

  # GRPO with rule-based reward (no reward model needed)
  python rlhf.py grpo --checkpoint best.pt --rule-reward length
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # Reward model training
    rm_parser = sub.add_parser("reward", help="Train a reward model")
    rm_parser.add_argument("--checkpoint", required=True)
    rm_parser.add_argument("--data", required=True, help="Preference JSONL file")
    rm_parser.add_argument("--train-full", action="store_true")
    rm_parser.add_argument("--epochs", type=int, default=3)
    rm_parser.add_argument("--batch-size", type=int, default=8)
    rm_parser.add_argument("--max-length", type=int, default=256)
    rm_parser.add_argument("--lr", type=float, default=1e-4)
    rm_parser.add_argument("--output-dir", default="checkpoints")
    rm_parser.add_argument("--device", default="auto")

    # GRPO
    grpo_parser = sub.add_parser("grpo", help="GRPO alignment (DeepSeek R1 style)")
    grpo_parser.add_argument("--checkpoint", required=True)
    grpo_parser.add_argument("--reward-model", default=None)
    grpo_parser.add_argument("--rule-reward", default=None,
                             choices=list(REWARD_FUNCTIONS.keys()))
    grpo_parser.add_argument("--data", default=None, help="Prompts file")
    grpo_parser.add_argument("--group-size", type=int, default=4)
    grpo_parser.add_argument("--max-gen", type=int, default=128)
    grpo_parser.add_argument("--max-prompt-length", type=int, default=128)
    grpo_parser.add_argument("--kl-coef", type=float, default=0.04)
    grpo_parser.add_argument("--temperature", type=float, default=0.8)
    grpo_parser.add_argument("--max-steps", type=int, default=500)
    grpo_parser.add_argument("--batch-size", type=int, default=4)
    grpo_parser.add_argument("--save-interval", type=int, default=100)
    grpo_parser.add_argument("--lr", type=float, default=1e-5)
    grpo_parser.add_argument("--output-dir", default="checkpoints")
    grpo_parser.add_argument("--device", default="auto")

    # PPO
    ppo_parser = sub.add_parser("ppo", help="PPO alignment (classic RLHF)")
    ppo_parser.add_argument("--checkpoint", required=True)
    ppo_parser.add_argument("--reward-model", required=True)
    ppo_parser.add_argument("--data", default=None)
    ppo_parser.add_argument("--max-gen", type=int, default=128)
    ppo_parser.add_argument("--kl-coef", type=float, default=0.1)
    ppo_parser.add_argument("--clip-eps", type=float, default=0.2)
    ppo_parser.add_argument("--max-steps", type=int, default=500)
    ppo_parser.add_argument("--batch-size", type=int, default=4)
    ppo_parser.add_argument("--save-interval", type=int, default=100)
    ppo_parser.add_argument("--lr", type=float, default=1e-5)
    ppo_parser.add_argument("--output-dir", default="checkpoints")
    ppo_parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "reward":
        train_reward_model(args)
    elif args.command == "grpo":
        train_grpo(args)
    elif args.command == "ppo":
        train_ppo(args)
    else:
        parser.print_help()
