import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.chunk(x, chunks=2, dim=self.dim)
        if self.nonlinear:
            value = torch.tanh(value)
        return torch.sigmoid(gate) * value


class ParameterizedGatedLinearUnit(nn.Module):
    def __init__(self, d_hidden: int, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear
        self.weight = nn.Parameter(torch.zeros((d_hidden,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.chunk(x, chunks=2, dim=self.dim)
        if self.nonlinear:
            value = torch.tanh(value)
        return torch.sigmoid(gate) * value * torch.tanh(self.weight)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        d_input: Optional[int] = None,
        d_output: Optional[int] = None,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_input = d_input or d_hidden
        self.d_output = d_output or self.d_input
        self.d_static = d_static or 0
        self.add_skip = self.d_output != self.d_input
        if self.add_skip:
            self.skip_proj = nn.Linear(self.d_input, self.d_output)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_input + self.d_static, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, self.d_output * 2),
            GatedLinearUnit(nonlinear=False),
        )
        self.layer_norm = nn.LayerNorm(self.d_output)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.d_static > 0:
            if c is None:
                raise ValueError("A static context tensor is required.")
            x_in = torch.cat([x, c], dim=-1)
        else:
            x_in = x
        skip = self.skip_proj(x) if self.add_skip else x
        return self.layer_norm(self.mlp(x_in) + skip)


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        num_vars: int,
        dropout: float = 0.0,
        add_static: bool = False,
    ):
        super().__init__()
        if num_vars < 1:
            raise ValueError("VariableSelectionNetwork requires at least one variable.")
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static
        shrink = 2 ** int(math.log2(num_vars))
        self.var_hidden = max(1, d_hidden // shrink)

        self.proj = nn.Linear(d_hidden, self.var_hidden)
        self.weight_network = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_input=self.var_hidden * num_vars,
            d_output=num_vars,
            d_static=d_hidden if add_static else None,
            dropout=dropout,
        )
        self.variable_network = GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)

    def forward(
        self,
        variables: List[torch.Tensor],
        static: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(variables) != self.num_vars:
            raise ValueError(f"Expected {self.num_vars} variables, got {len(variables)}.")

        stacked = torch.stack(variables, dim=-2)
        projected = self.proj(stacked).flatten(start_dim=-2)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(projected, static)
        if mask is not None:
            weight = weight.masked_fill(~mask.bool(), -1e8)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)
        encoded = self.variable_network(stacked)
        selected = torch.sum(encoded.transpose(-2, -1) * weight, dim=-1)
        return selected, weight


class PEMSUniCACovariateAdapter(nn.Module):
    """UniCA-style covariate adapter for ST-LLM+ PEMS node tokens.

    The adapter mirrors UniCA's plug-in recipe locally: covariate homogenization,
    variable selection, gated residual transforms, and attention-based future
    fusion, without depending on the UniCA repository or GluonTS stack.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_len: int,
        output_len: int,
        num_nodes: int,
        past_dynamic_dim: int = 2,
        future_dynamic_dim: int = 4,
        static_dim: int = 3,
        num_heads: int = 1,
        dropout: float = 0.1,
        with_gate: bool = False,
        with_future: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.past_dynamic_dim = past_dynamic_dim
        self.future_dynamic_dim = future_dynamic_dim
        self.static_dim = static_dim
        self.with_gate = with_gate
        self.with_future = with_future

        self.past_projectors = nn.ModuleList(
            [nn.Linear(input_len, hidden_dim) for _ in range(past_dynamic_dim)]
        )
        self.future_projectors = nn.ModuleList(
            [nn.Linear(output_len, hidden_dim) for _ in range(future_dynamic_dim)]
        )
        self.static_projectors = nn.ModuleList(
            [nn.Linear(1, hidden_dim) for _ in range(static_dim)]
        )

        self.static_selector = VariableSelectionNetwork(
            d_hidden=hidden_dim,
            num_vars=max(static_dim, 1),
            dropout=dropout,
        )
        self.static_context = GatedResidualNetwork(d_hidden=hidden_dim, dropout=dropout)
        self.past_selector = VariableSelectionNetwork(
            d_hidden=hidden_dim,
            num_vars=1 + past_dynamic_dim,
            dropout=dropout,
            add_static=True,
        )
        self.future_selector = VariableSelectionNetwork(
            d_hidden=hidden_dim,
            num_vars=max(future_dynamic_dim, 1),
            dropout=dropout,
            add_static=True,
        )

        if with_gate:
            self.past_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                ParameterizedGatedLinearUnit(hidden_dim, nonlinear=False),
            )
            self.future_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                ParameterizedGatedLinearUnit(hidden_dim, nonlinear=False),
            )
        else:
            self.past_gate = None
            self.future_gate = None

        self.future_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.past_future_indicator = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _expand_nodes(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(2) == self.num_nodes:
            return x
        if x.size(2) == 1:
            return x.expand(-1, -1, self.num_nodes, -1)
        raise ValueError(f"Expected node dimension 1 or {self.num_nodes}, got {x.size(2)}.")

    def _static_tokens(
        self,
        covariates: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        static_real = covariates.get("static_real")
        if static_real is None:
            static_real = torch.zeros(self.num_nodes, self.static_dim, device=device)
        static_real = static_real.to(device=device, dtype=torch.float32)
        if static_real.dim() == 2:
            static_real = static_real.unsqueeze(0).expand(batch_size, -1, -1)
        elif static_real.dim() == 3 and static_real.size(0) == 1:
            static_real = static_real.expand(batch_size, -1, -1)
        if static_real.size(1) != self.num_nodes:
            raise ValueError(f"Expected static_real node dimension {self.num_nodes}, got {static_real.size(1)}.")

        tokens = []
        for i, projector in enumerate(self.static_projectors):
            feat = static_real[..., i : i + 1]
            tokens.append(projector(feat))
        selected, _ = self.static_selector(tokens)
        return self.static_context(selected)

    def _past_tokens(
        self,
        covariates: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        past_dynamic = covariates.get("past_dynamic_real")
        if past_dynamic is None:
            raise ValueError("UniCA adapter requires past_dynamic_real covariates.")
        past_dynamic = past_dynamic.to(device=device, dtype=torch.float32)
        past_dynamic = self._expand_nodes(past_dynamic)
        past_mask = covariates.get("past_observed_mask")
        if past_mask is None:
            past_mask = torch.ones_like(past_dynamic)
        else:
            past_mask = self._expand_nodes(past_mask.to(device=device, dtype=torch.float32))
        past_dynamic = past_dynamic * past_mask

        tokens = []
        masks = []
        for i, projector in enumerate(self.past_projectors):
            feat = past_dynamic[..., i].permute(0, 2, 1).contiguous()
            tokens.append(projector(feat))
            masks.append(past_mask[..., i].any(dim=1))
        return tokens, torch.stack(masks, dim=-1).bool()

    def _future_tokens(
        self,
        covariates: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        future_dynamic = covariates.get("future_dynamic_real")
        if future_dynamic is None:
            raise ValueError("UniCA future fusion requires future_dynamic_real covariates.")
        future_dynamic = future_dynamic.to(device=device, dtype=torch.float32)
        future_dynamic = self._expand_nodes(future_dynamic)
        future_mask = covariates.get("future_observed_mask")
        if future_mask is None:
            future_mask = torch.ones_like(future_dynamic)
        else:
            future_mask = self._expand_nodes(future_mask.to(device=device, dtype=torch.float32))
        future_dynamic = future_dynamic * future_mask

        tokens = []
        masks = []
        for i, projector in enumerate(self.future_projectors):
            feat = future_dynamic[..., i].permute(0, 2, 1).contiguous()
            tokens.append(projector(feat))
            masks.append(future_mask[..., i].any(dim=1))
        return tokens, torch.stack(masks, dim=-1).bool()

    def fuse_past(self, node_tokens: torch.Tensor, covariates: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, num_nodes, _ = node_tokens.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} node tokens, got {num_nodes}.")
        static = self._static_tokens(covariates, batch_size, node_tokens.device)
        past_tokens, past_mask = self._past_tokens(covariates, node_tokens.device)
        target_mask = torch.ones(batch_size, num_nodes, 1, device=node_tokens.device, dtype=torch.bool)
        selected, _ = self.past_selector(
            [node_tokens] + past_tokens,
            static=static,
            mask=torch.cat([target_mask, past_mask], dim=-1),
        )
        if self.past_gate is not None:
            selected = self.past_gate(selected)
        return node_tokens + self.dropout(selected)

    def fuse_future(self, node_tokens: torch.Tensor, covariates: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.with_future:
            return node_tokens
        batch_size, num_nodes, _ = node_tokens.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} node tokens, got {num_nodes}.")
        static = self._static_tokens(covariates, batch_size, node_tokens.device)
        future_tokens, future_mask = self._future_tokens(covariates, node_tokens.device)
        selected, _ = self.future_selector(future_tokens, static=static, mask=future_mask)
        past_tokens = node_tokens + self.past_future_indicator[:, 0:1, :]
        known_future = selected + self.past_future_indicator[:, 1:2, :]
        fused = torch.cat([past_tokens, known_future], dim=1)
        fused = self.future_attention(fused, fused, fused, need_weights=False)[0]
        future_context = fused[:, :num_nodes, :]
        if self.future_gate is not None:
            future_context = self.future_gate(future_context)
        return node_tokens + self.dropout(future_context)


# Import compatibility; class renaming does not change state_dict keys.
PEMS04UniCACovariateAdapter = PEMSUniCACovariateAdapter
