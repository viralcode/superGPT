"""
3D Parallelism for superGPT
==============================
Megatron-style Tensor Parallelism (TP) and Pipeline Parallelism (PP)
for training massive models across hundreds of GPUs.

Architecture:
  - Tensor Parallel: Split attention heads and FFN columns across GPUs in a node
  - Pipeline Parallel: Split transformer layers across stages with 1F1B schedule
  - Data Parallel: FSDP handles data parallelism (existing)
  - Combined: TP × PP × DP = Total GPU count

Usage:
    # 8 GPUs: 2-way tensor parallel × 4-way pipeline parallel
    torchrun --nproc_per_node=8 train.py --preset xl \\
        --tensor-parallel 2 --pipeline-parallel 4

    # 16 GPUs across 2 nodes: 4-way TP × 2-way PP × 2-way DP
    torchrun --nproc_per_node=8 --nnodes=2 train.py --preset gpt4 \\
        --tensor-parallel 4 --pipeline-parallel 2

Reference:
    Shoeybi et al., "Megatron-LM" (2019)
    Narayanan et al., "Efficient Large-Scale Language Model Training" (2021)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple


# ==============================================================================
#  Process Group Mesh
# ==============================================================================

class ParallelMesh:
    """3D parallelism process group mesh.

    Organizes GPUs into a 3D grid: [TP, PP, DP]
    For example, 16 GPUs with TP=4, PP=2, DP=2:
        GPU layout: [tp_rank, pp_rank, dp_rank]
    """

    def __init__(self, tp_size: int = 1, pp_size: int = 1):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")

        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = self.world_size // (tp_size * pp_size)

        assert self.world_size == tp_size * pp_size * self.dp_size, \
            f"world_size ({self.world_size}) != TP({tp_size}) × PP({pp_size}) × DP({self.dp_size})"

        # Compute local ranks
        self.tp_rank = self.global_rank % tp_size
        self.pp_rank = (self.global_rank // tp_size) % pp_size
        self.dp_rank = self.global_rank // (tp_size * pp_size)

        # Create process groups
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
        self._create_groups()

        print(f"[Rank {self.global_rank}] Mesh: TP={self.tp_rank}/{tp_size} "
              f"PP={self.pp_rank}/{pp_size} DP={self.dp_rank}/{self.dp_size}")

    def _create_groups(self):
        """Create TP, PP, and DP process groups."""
        # TP groups: GPUs that share the same PP and DP rank
        for pp in range(self.pp_size):
            for dp in range(self.dp_size):
                ranks = [dp * self.tp_size * self.pp_size + pp * self.tp_size + tp
                         for tp in range(self.tp_size)]
                group = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.tp_group = group

        # PP groups: GPUs that share the same TP and DP rank
        for tp in range(self.tp_size):
            for dp in range(self.dp_size):
                ranks = [dp * self.tp_size * self.pp_size + pp * self.tp_size + tp
                         for pp in range(self.pp_size)]
                group = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.pp_group = group

        # DP groups: GPUs that share the same TP and PP rank
        for tp in range(self.tp_size):
            for pp in range(self.pp_size):
                ranks = [dp * self.tp_size * self.pp_size + pp * self.tp_size + tp
                         for dp in range(self.dp_size)]
                group = dist.new_group(ranks)
                if self.global_rank in ranks:
                    self.dp_group = group


# ==============================================================================
#  Tensor Parallelism Primitives
# ==============================================================================

class _AllReduceFunc(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _IdentityFwdAllReduceBwd(torch.autograd.Function):
    """Identity in forward, all-reduce in backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


def all_reduce_forward(x, group):
    """All-reduce tensor across TP group (forward pass)."""
    return _AllReduceFunc.apply(x, group)


def all_reduce_backward(x, group):
    """All-reduce gradients across TP group (backward pass)."""
    return _IdentityFwdAllReduceBwd.apply(x, group)


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer.

    Splits the output dimension across TP ranks.
    Y = XA^T where A is split column-wise: A = [A_0 | A_1 | ... | A_{tp-1}]
    Each rank computes Y_i = X @ A_i^T (a slice of the output).

    Used for: Q, K, V projections, FFN up-projections (w1, w3)
    """

    def __init__(self, in_features: int, out_features: int, tp_group, tp_size: int,
                 bias: bool = False, gather_output: bool = False):
        super().__init__()
        assert out_features % tp_size == 0, \
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"

        self.tp_group = tp_group
        self.tp_size = tp_size
        self.gather_output = gather_output
        self.out_per_rank = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank)) if bias else None

        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Identity forward, all-reduce backward (for input gradients)
        x = all_reduce_backward(x, self.tp_group)
        y = nn.functional.linear(x, self.weight, self.bias)

        if self.gather_output:
            # Gather outputs from all TP ranks
            output_list = [torch.empty_like(y) for _ in range(self.tp_size)]
            dist.all_gather(output_list, y, group=self.tp_group)
            y = torch.cat(output_list, dim=-1)

        return y


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer.

    Splits the input dimension across TP ranks.
    Y = XA^T where A is split row-wise: A = [A_0; A_1; ...; A_{tp-1}]
    Each rank computes partial_Y_i = X_i @ A_i^T, then all-reduce to get Y.

    Used for: Output projections (attn out, FFN down w2)
    """

    def __init__(self, in_features: int, out_features: int, tp_group, tp_size: int,
                 bias: bool = False, input_is_parallel: bool = True):
        super().__init__()
        assert in_features % tp_size == 0, \
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"

        self.tp_group = tp_group
        self.tp_size = tp_size
        self.input_is_parallel = input_is_parallel
        self.in_per_rank = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        y = nn.functional.linear(x, self.weight)

        # All-reduce forward (sum partial results across TP ranks)
        y = all_reduce_forward(y, self.tp_group)

        if self.bias is not None:
            y = y + self.bias

        return y


# ==============================================================================
#  Pipeline Parallelism (1F1B Schedule)
# ==============================================================================

class PipelineStage(nn.Module):
    """Wraps a subset of transformer layers as a pipeline stage."""

    def __init__(self, layers: nn.ModuleList, pp_rank: int, pp_size: int,
                 embed: Optional[nn.Module] = None, ln_f: Optional[nn.Module] = None,
                 lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.layers = layers
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_first = (pp_rank == 0)
        self.is_last = (pp_rank == pp_size - 1)

        # First stage owns embedding, last stage owns head
        self.embed = embed if self.is_first else None
        self.ln_f = ln_f if self.is_last else None
        self.lm_head = lm_head if self.is_last else None

    def forward(self, x, targets=None):
        if self.is_first and self.embed is not None:
            x = self.embed(x)

        for layer in self.layers:
            x, _ = layer(x)

        if self.is_last:
            if self.ln_f is not None:
                x = self.ln_f(x)
            if self.lm_head is not None:
                logits = self.lm_head(x)
                loss = None
                if targets is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                return logits, loss

        return x, None


def send_tensor(tensor, dst_rank, group):
    """Send a tensor to dst_rank."""
    dist.send(tensor.contiguous(), dst=dst_rank, group=group)


def recv_tensor(shape, dtype, src_rank, device, group):
    """Receive a tensor from src_rank."""
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank, group=group)
    return tensor


class PipelineScheduler:
    """1F1B (one-forward-one-backward) pipeline schedule.

    Splits a batch into micro-batches and pipelines them across stages.

    Schedule for 4 stages, 4 micro-batches:
        Stage 0:  F0 F1 F2 F3 B3 B2 B1 B0
        Stage 1:     F0 F1 F2 F3 B3 B2 B1 B0
        Stage 2:        F0 F1 F2 F3 B3 B2 B1 B0
        Stage 3:           F0 F1 F2 F3 B3 B2 B1 B0
    """

    def __init__(self, stage: PipelineStage, mesh: ParallelMesh, n_micro: int = 4):
        self.stage = stage
        self.mesh = mesh
        self.n_micro = n_micro

    def run(self, input_batch, targets_batch=None):
        """Execute 1F1B schedule on input batch.

        Returns:
            total_loss averaged across micro-batches (only valid on last stage)
        """
        pp_rank = self.mesh.pp_rank
        pp_size = self.mesh.pp_size
        device = next(self.stage.parameters()).device

        # Split into micro-batches
        micro_inputs = input_batch.chunk(self.n_micro)
        micro_targets = targets_batch.chunk(self.n_micro) if targets_batch is not None else [None] * self.n_micro

        activations = []  # For backward pass
        total_loss = torch.tensor(0.0, device=device)

        # -- Warmup: forward passes to fill pipeline --
        n_warmup = min(pp_size - pp_rank - 1, self.n_micro)
        n_1f1b = self.n_micro - n_warmup

        for i in range(n_warmup):
            output = self._forward_step(micro_inputs[i], micro_targets[i],
                                         pp_rank, pp_size, device)
            activations.append(output)

        # -- Steady state: 1F1B --
        for i in range(n_1f1b):
            # Forward
            fwd_idx = n_warmup + i
            output = self._forward_step(micro_inputs[fwd_idx], micro_targets[fwd_idx],
                                         pp_rank, pp_size, device)

            # Backward (on earliest un-backwarded micro-batch)
            if activations:
                bwd_output = activations.pop(0)
                if isinstance(bwd_output, tuple) and bwd_output[1] is not None:
                    loss = bwd_output[1] / self.n_micro
                    loss.backward()
                    total_loss += loss.detach()

            activations.append(output)

        # -- Cooldown: backward passes for remaining --
        for output in activations:
            if isinstance(output, tuple) and output[1] is not None:
                loss = output[1] / self.n_micro
                loss.backward()
                total_loss += loss.detach()

        return total_loss

    def _forward_step(self, micro_input, micro_target, pp_rank, pp_size, device):
        """Forward one micro-batch through this stage."""
        if pp_rank == 0:
            x = micro_input
        else:
            # Receive from previous stage
            # Estimate shape from the stage
            x = recv_tensor(
                micro_input.shape, micro_input.dtype,
                self.mesh.global_rank - self.mesh.tp_size,
                device, self.mesh.pp_group,
            )

        output = self.stage(x, micro_target)

        if pp_rank < pp_size - 1:
            # Send to next stage
            out_tensor = output[0] if isinstance(output, tuple) else output
            send_tensor(
                out_tensor,
                self.mesh.global_rank + self.mesh.tp_size,
                self.mesh.pp_group,
            )

        return output


# ==============================================================================
#  Model Parallelization Utilities
# ==============================================================================

def parallelize_model(model, mesh: ParallelMesh):
    """Apply tensor and pipeline parallelism to a GPT model.

    Args:
        model: GPT model instance
        mesh: ParallelMesh with TP/PP configuration

    Returns:
        PipelineStage wrapping this rank's layers
    """
    n_layers = len(model.transformer.h)
    assert n_layers % mesh.pp_size == 0, \
        f"n_layers ({n_layers}) must be divisible by pp_size ({mesh.pp_size})"

    layers_per_stage = n_layers // mesh.pp_size
    start = mesh.pp_rank * layers_per_stage
    end = start + layers_per_stage

    # Extract this stage's layers
    stage_layers = nn.ModuleList(list(model.transformer.h)[start:end])

    # Build pipeline stage
    stage = PipelineStage(
        layers=stage_layers,
        pp_rank=mesh.pp_rank,
        pp_size=mesh.pp_size,
        embed=model.transformer.wte if mesh.pp_rank == 0 else None,
        ln_f=model.transformer.ln_f if mesh.pp_rank == mesh.pp_size - 1 else None,
        lm_head=model.lm_head if mesh.pp_rank == mesh.pp_size - 1 else None,
    )

    # Apply tensor parallelism to attention heads within this stage
    if mesh.tp_size > 1:
        for layer in stage.layers:
            _apply_tp_to_layer(layer, mesh)

    return stage


def _apply_tp_to_layer(layer, mesh: ParallelMesh):
    """Apply tensor parallelism to a single transformer layer.

    Replaces nn.Linear layers with ColumnParallel/RowParallel equivalents.
    Handles both GQA (q_proj/k_proj/v_proj/c_proj) and MLA attention variants.
    """
    attn = layer.attn

    # GQA attention: q_proj, k_proj, v_proj are column-parallel, c_proj is row-parallel
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        proj = getattr(attn, proj_name, None)
        if isinstance(proj, nn.Linear):
            in_f = proj.in_features
            out_f = proj.out_features
            if out_f % mesh.tp_size == 0:
                setattr(attn, proj_name, ColumnParallelLinear(
                    in_f, out_f, mesh.tp_group, mesh.tp_size,
                ))

    if hasattr(attn, 'c_proj') and isinstance(attn.c_proj, nn.Linear):
        in_f = attn.c_proj.in_features
        out_f = attn.c_proj.out_features
        if in_f % mesh.tp_size == 0:
            attn.c_proj = RowParallelLinear(
                in_f, out_f, mesh.tp_group, mesh.tp_size,
            )

    # MLA attention: wq_a, wq_b, wkv_a, wkv_b, wo
    for proj_name in ['wq_a', 'wq_b', 'wkv_a', 'wkv_b']:
        proj = getattr(attn, proj_name, None)
        if isinstance(proj, nn.Linear):
            in_f = proj.in_features
            out_f = proj.out_features
            if out_f % mesh.tp_size == 0:
                setattr(attn, proj_name, ColumnParallelLinear(
                    in_f, out_f, mesh.tp_group, mesh.tp_size,
                ))

    if hasattr(attn, 'wo') and isinstance(attn.wo, nn.Linear):
        in_f = attn.wo.in_features
        out_f = attn.wo.out_features
        if in_f % mesh.tp_size == 0:
            attn.wo = RowParallelLinear(
                in_f, out_f, mesh.tp_group, mesh.tp_size,
            )

    # Also handle fused c_attn if present (some model variants)
    if hasattr(attn, 'c_attn') and isinstance(attn.c_attn, nn.Linear):
        in_f = attn.c_attn.in_features
        out_f = attn.c_attn.out_features
        if out_f % mesh.tp_size == 0:
            attn.c_attn = ColumnParallelLinear(
                in_f, out_f, mesh.tp_group, mesh.tp_size,
            )

    # FFN: w1, w3 are column-parallel, w2 is row-parallel
    ffn = layer.ffn
    if hasattr(ffn, 'w1') and isinstance(ffn.w1, nn.Linear):
        in_f = ffn.w1.in_features
        out_f = ffn.w1.out_features
        if out_f % mesh.tp_size == 0:
            ffn.w1 = ColumnParallelLinear(in_f, out_f, mesh.tp_group, mesh.tp_size)
        if hasattr(ffn, 'w3') and isinstance(ffn.w3, nn.Linear):
            ffn.w3 = ColumnParallelLinear(in_f, out_f, mesh.tp_group, mesh.tp_size)
        if hasattr(ffn, 'w2') and isinstance(ffn.w2, nn.Linear):
            ffn.w2 = RowParallelLinear(out_f, in_f, mesh.tp_group, mesh.tp_size)


def get_parallel_args(parser):
    """Add parallelism arguments to an argument parser."""
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel degree (default: 1)")
    parser.add_argument("--pipeline-parallel", type=int, default=1,
                        help="Pipeline parallel degree (default: 1)")
    parser.add_argument("--pipeline-micro-batches", type=int, default=4,
                        help="Number of micro-batches for pipeline (default: 4)")
    return parser
