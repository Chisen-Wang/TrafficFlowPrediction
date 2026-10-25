import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cv_squared(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    if x.shape[0] <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean().square() + eps)


def _entropy_loss(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    if x.shape[0] <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return -(x * torch.log(x + eps)).sum(dim=-1).mean()


class TimeFilterMoERouter(nn.Module):
    """Top-p expert router adapted from TimeFilter's patch filtration idea.

    Each node row chooses among three edge families: physical graph edges,
    dynamically similar edges, and the residual global edges.
    """

    def __init__(self, num_nodes: int, top_p: float = 0.5):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_p = top_p
        self.gate = nn.Linear(num_nodes, 3, bias=False)
        self.noise = nn.Linear(num_nodes, 3, bias=False)
        self.softplus = nn.Softplus()

    def _noisy_top_p_gating(
        self,
        rows: torch.Tensor,
        is_training: bool,
        noise_epsilon: float = 1e-2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_logits = self.gate(rows)
        if is_training:
            raw_noise = self.noise(rows)
            noise_stddev = self.softplus(raw_noise) + noise_epsilon
            clean_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev

        probs = torch.softmax(clean_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        above_top_p = cumulative > self.top_p

        threshold_indices = above_top_p.long().argmax(dim=-1)
        threshold = F.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        remove_mask = above_top_p & ~threshold

        keep_sorted = ~remove_mask
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(-1, sorted_indices, keep_sorted)

        routed_probs = probs * keep.float()
        loss = _cv_squared(routed_probs.sum(dim=0)) + 0.1 * _entropy_loss(probs)
        return keep.bool(), loss

    def forward(self, rows: torch.Tensor, is_training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.top_p <= 0.0:
            keep = torch.zeros(*rows.shape[:-1], 3, device=rows.device, dtype=torch.bool)
            keep[..., 0] = True
            return keep, torch.zeros((), device=rows.device, dtype=rows.dtype)

        flat_rows = rows.reshape(-1, rows.size(-1))
        keep, loss = self._noisy_top_p_gating(flat_rows, is_training)
        return keep.reshape(*rows.shape[:-1], 3), loss


class TimeFilterAttentionBias(nn.Module):
    """Builds TimeFilter-style additive attention bias for node-token GPT layers."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_nodes: int,
        alpha: float = 0.1,
        top_p: float = 0.5,
        static_weight: float = 1.0,
        dynamic_weight: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.head_dim = hidden_dim // num_heads
        self.alpha = alpha
        self.static_weight = static_weight
        self.dynamic_weight = dynamic_weight

        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.router = TimeFilterMoERouter(num_nodes=num_nodes, top_p=top_p)

        self.last_bias_shape: Optional[Tuple[int, ...]] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        x = x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _dynamic_similarity(self, node_tokens: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.proj_q(node_tokens))
        key = self._split_heads(self.proj_k(node_tokens))
        sim = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        return F.gelu(sim)

    def _dynamic_allowed(self, sim: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0.0:
            return torch.ones_like(sim, dtype=torch.bool)

        drop_k = int(self.alpha * sim.size(-1))
        if drop_k <= 0:
            return torch.ones_like(sim, dtype=torch.bool)

        low_indices = torch.topk(sim, k=drop_k, dim=-1, largest=False).indices
        allowed = torch.ones_like(sim, dtype=torch.bool)
        allowed.scatter_(-1, low_indices, False)
        return allowed

    def forward(
        self,
        node_tokens: torch.Tensor,
        static_adj: torch.Tensor,
        is_training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = node_tokens.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}.")

        sim = self._dynamic_similarity(node_tokens)
        dynamic_allowed = self._dynamic_allowed(sim)

        static_adj = static_adj.to(device=node_tokens.device, dtype=sim.dtype)
        static_allowed = static_adj > 0
        eye = torch.eye(num_nodes, device=node_tokens.device, dtype=torch.bool)

        static_region = static_allowed & ~eye
        dynamic_region = dynamic_allowed & ~static_allowed.unsqueeze(0).unsqueeze(0) & ~eye
        residual_region = ~(static_region.unsqueeze(0).unsqueeze(0) | dynamic_region | eye)

        route_keep, moe_loss = self.router(sim, is_training=is_training)
        route_static = route_keep[..., 0].unsqueeze(-1) & static_region.unsqueeze(0).unsqueeze(0)
        route_dynamic = route_keep[..., 1].unsqueeze(-1) & dynamic_region
        route_residual = route_keep[..., 2].unsqueeze(-1) & residual_region
        allowed = eye.unsqueeze(0).unsqueeze(0) | route_static | route_dynamic | route_residual

        static_bias = self.static_weight * static_adj.unsqueeze(0).unsqueeze(0)
        dynamic_bias = self.dynamic_weight * sim
        additive_bias = static_bias + dynamic_bias

        mask_value = torch.finfo(additive_bias.dtype).min
        additive_bias = additive_bias.masked_fill(~allowed, mask_value)
        additive_bias = additive_bias.expand(batch_size, self.num_heads, num_nodes, num_nodes)

        self.last_bias_shape = tuple(additive_bias.shape)
        return additive_bias, moe_loss
