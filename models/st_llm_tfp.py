import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import GPT2Model

from models.timefilter_attention import TimeFilterAttentionBias
from models.unica_adapter import PEMSUniCACovariateAdapter


@dataclass
class PFAOutput:
    last_hidden_state: torch.Tensor
    moe_loss: torch.Tensor
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class TemporalEmbedding(nn.Module):
    def __init__(self, time: int, features: int):
        super().__init__()
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_index = (x[..., 1][:, -1, :] * self.time).long().clamp(0, self.time - 1)
        week_index = x[..., 2][:, -1, :].long().clamp(0, 6)

        time_day = self.time_day[time_index].transpose(1, 2).unsqueeze(-1)
        time_week = self.time_week[week_index].transpose(1, 2).unsqueeze(-1)
        return time_day + time_week


class TimeFilteredPFA(nn.Module):
    """Partially frozen GPT-2 with TimeFilter-enhanced graph attention in last U layers."""

    def __init__(
        self,
        adj_mx,
        device: str = "cuda:0",
        gpt_layers: int = 6,
        U: int = 1,
        dropout_rate: float = 0.1,
        tfp_enabled: bool = True,
        tfp_config: Optional[Dict] = None,
    ):
        super().__init__()
        if U < 1 or U > gpt_layers:
            raise ValueError("U must be in [1, gpt_layers]")

        self.device_name = device
        self.gpt_layers = gpt_layers
        self.U = U
        self.dropout_rate = dropout_rate
        self.tfp_enabled = tfp_enabled
        self.tfp_config = tfp_config or {}
        self.last_attention_bias_shape: Optional[Tuple[int, ...]] = None

        base = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation="eager",
            output_attentions=True,
            output_hidden_states=True,
        )
        if len(adj_mx) > base.wpe.num_embeddings:
            raise ValueError(
                f"Graph has {len(adj_mx)} nodes, exceeding GPT-2 position capacity "
                f"{base.wpe.num_embeddings}"
            )
        if gpt_layers > len(base.h):
            raise ValueError(f"Requested {gpt_layers} GPT layers, available {len(base.h)}")
        base.h = base.h[:gpt_layers]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=dropout_rate,
            target_modules=["q_attn", "c_attn"],
            bias="none",
        )
        self.gpt2 = get_peft_model(base, lora_config)
        self.num_heads = self.transformer.config.n_head
        self.hidden_dim = self.transformer.config.n_embd
        self.head_dim = self.hidden_dim // self.num_heads

        self.register_buffer("static_adj", torch.as_tensor(adj_mx, dtype=torch.float32))
        self.tfp_bias = None
        if tfp_enabled:
            self.tfp_bias = TimeFilterAttentionBias(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_nodes=self.static_adj.size(0),
                alpha=float(self.tfp_config.get("alpha", 0.1)),
                top_p=float(self.tfp_config.get("top_p", 0.5)),
                static_weight=float(self.tfp_config.get("static_weight", 1.0)),
                dynamic_weight=float(self.tfp_config.get("dynamic_weight", 1.0)),
            )

        self.dropout = nn.Dropout(dropout_rate)
        self._configure_trainable_parameters()

    @property
    def transformer(self):
        return self.gpt2.base_model.model

    def _configure_trainable_parameters(self) -> None:
        for param in self.gpt2.parameters():
            param.requires_grad = False

        graph_start = self.gpt_layers - self.U
        for layer_index, layer in enumerate(self.transformer.h):
            for name, param in layer.named_parameters():
                if "ln" in name:
                    param.requires_grad = True
                elif layer_index >= graph_start and "mlp" not in name:
                    param.requires_grad = True

        for param in self.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in self.transformer.wpe.parameters():
            param.requires_grad = True

    def _static_attention_bias(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        adj = self.static_adj.to(device=device, dtype=dtype)
        num_nodes = adj.size(0)
        eye = torch.eye(num_nodes, device=device, dtype=torch.bool)
        allowed = (adj > 0) | eye
        bias = adj.masked_fill(~allowed, torch.finfo(dtype).min)
        bias = bias.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, num_nodes, num_nodes)
        self.last_attention_bias_shape = tuple(bias.shape)
        return bias

    def _build_graph_bias(
        self,
        hidden_states: torch.Tensor,
        is_training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tfp_bias is None:
            bias = self._static_attention_bias(
                batch_size=hidden_states.size(0),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            return bias, torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        bias, moe_loss = self.tfp_bias(hidden_states, self.static_adj, is_training=is_training)
        self.last_attention_bias_shape = self.tfp_bias.last_bias_shape
        return bias.to(dtype=hidden_states.dtype), moe_loss.to(dtype=hidden_states.dtype)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)

    def _graph_attention(
        self,
        attention,
        hidden_states: torch.Tensor,
        graph_bias: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query, key, value = attention.c_attn(hidden_states).split(self.hidden_dim, dim=2)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if attention.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(value.size(-1))
        if attention.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(attention.layer_idx + 1)

        attn_weights = attn_weights + graph_bias[:, :, :, : key.size(-2)]
        attn_probs = torch.softmax(attn_weights, dim=-1).type(value.dtype)
        attn_probs = attention.attn_dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = attention.c_proj(attn_output)
        attn_output = attention.resid_dropout(attn_output)
        return attn_output, attn_probs if output_attentions else None

    def _graph_block_forward(
        self,
        block,
        hidden_states: torch.Tensor,
        graph_bias: torch.Tensor,
        head_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = block.ln_1(hidden_states)
        attn_output, attn_weights = self._graph_attention(
            block.attn,
            hidden_states,
            graph_bias,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = block.ln_2(hidden_states)
        feed_forward_hidden_states = block.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states, attn_weights

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        is_training: bool = False,
        output_attentions: bool = False,
    ) -> PFAOutput:
        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device).unsqueeze(0)
        hidden_states = inputs_embeds + self.transformer.wpe(position_ids)

        all_self_attentions = () if output_attentions else None
        moe_loss = torch.zeros((), device=device, dtype=hidden_states.dtype)
        graph_layer_count = 0
        graph_start = self.gpt_layers - self.U

        for layer_index, block in enumerate(self.transformer.h):
            if layer_index < graph_start:
                outputs = block(
                    hidden_states,
                    attention_mask=None,
                    head_mask=None,
                    use_cache=False,
                    output_attentions=output_attentions,
                )
                if isinstance(outputs, (tuple, list)):
                    hidden_states = outputs[0]
                    if output_attentions and len(outputs) > 1:
                        all_self_attentions = all_self_attentions + (outputs[1],)
                else:
                    hidden_states = outputs
                continue

            graph_bias, layer_moe_loss = self._build_graph_bias(hidden_states, is_training=is_training)
            hidden_states, attn_weights = self._graph_block_forward(
                block,
                hidden_states,
                graph_bias=graph_bias,
                head_mask=None,
                output_attentions=output_attentions,
            )
            moe_loss = moe_loss + layer_moe_loss
            graph_layer_count += 1
            if output_attentions:
                all_self_attentions = all_self_attentions + (attn_weights,)

        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))
        hidden_states = self.dropout(hidden_states)
        if graph_layer_count > 0:
            moe_loss = moe_loss / graph_layer_count
        return PFAOutput(last_hidden_state=hidden_states, moe_loss=moe_loss, attentions=all_self_attentions)


class ST_LLM_TFP(nn.Module):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim: int = 3,
        num_nodes: int = 307,
        input_len: int = 12,
        output_len: int = 3,
        llm_layer: int = 6,
        U: int = 1,
        tfp_enabled: bool = True,
        tfp_config: Optional[Dict] = None,
        unica_enabled: bool = False,
        unica_config: Optional[Dict] = None,
        steps_per_day: Optional[int] = None,
    ):
        super().__init__()
        self.device_name = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.tfp_enabled = tfp_enabled
        self.tfp_config = tfp_config or {}
        self.unica_enabled = unica_enabled
        self.unica_config = unica_config or {}
        self.dropout_rate = 0.1

        time = steps_per_day if steps_per_day is not None else (48 if num_nodes in {250, 266} else 288)
        if time <= 0:
            raise ValueError("steps_per_day must be positive")
        if input_dim != 3 or min(num_nodes, input_len, output_len) <= 0:
            raise ValueError("Expected input_dim=3 and positive nodes/window lengths")
        if tuple(torch.as_tensor(adj_mx).shape) != (num_nodes, num_nodes):
            raise ValueError(f"Expected adjacency shape ({num_nodes}, {num_nodes})")

        gpt_channel = 256
        to_gpt_channel = 768
        self.start_conv = nn.Conv2d(self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1))
        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.in_layer = nn.Conv2d(gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1))
        self.gpt = TimeFilteredPFA(
            adj_mx=adj_mx,
            device=device,
            gpt_layers=self.llm_layer,
            U=self.U,
            dropout_rate=self.dropout_rate,
            tfp_enabled=self.tfp_enabled,
            tfp_config=self.tfp_config,
        )

        self.unica_adapter = None
        self.unica_past_pos = self.unica_config.get("past_pos", "pre")
        self.unica_future_pos = self.unica_config.get("future_pos", "post")
        self.unica_with_future = bool(self.unica_config.get("with_future", True))
        if self.unica_enabled:
            if self.unica_past_pos not in {"pre", "post"}:
                raise ValueError("unica past_pos must be 'pre' or 'post'")
            if self.unica_future_pos not in {"pre", "post"}:
                raise ValueError("unica future_pos must be 'pre' or 'post'")
            self.unica_adapter = PEMSUniCACovariateAdapter(
                hidden_dim=to_gpt_channel,
                input_len=self.input_len,
                output_len=self.output_len,
                num_nodes=self.num_nodes,
                past_dynamic_dim=int(self.unica_config.get("past_dynamic_dim", 2)),
                future_dynamic_dim=int(self.unica_config.get("future_dynamic_dim", 4)),
                static_dim=int(self.unica_config.get("static_dim", 3)),
                num_heads=int(self.unica_config.get("num_heads", 1)),
                dropout=float(self.unica_config.get("dropout", self.dropout_rate)),
                with_gate=bool(self.unica_config.get("with_gate", False)),
                with_future=self.unica_with_future,
            )

    def param_num(self) -> int:
        return sum(param.nelement() for param in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def freeze_backbone_for_adapter_only(self) -> None:
        if self.unica_adapter is None:
            raise ValueError("Cannot enable adapter-only training without a UniCA adapter.")
        for param in self.parameters():
            param.requires_grad = False
        for param in self.unica_adapter.parameters():
            param.requires_grad = True

    @property
    def last_attention_bias_shape(self) -> Optional[Tuple[int, ...]]:
        return self.gpt.last_attention_bias_shape

    def forward(
        self,
        history_data: torch.Tensor,
        covariates: Optional[Dict[str, torch.Tensor]] = None,
        is_training: bool = False,
        return_aux: bool = False,
    ):
        data = history_data.permute(0, 3, 2, 1)
        batch_size, _, num_nodes, _ = data.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}.")

        tem_emb = self.Temb(data)
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)

        input_data = data.permute(0, 3, 2, 1).transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        input_data = self.start_conv(input_data)

        data_st = torch.cat([input_data, tem_emb, node_emb], dim=1)
        data_st = F.leaky_relu(self.in_layer(data_st))
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)

        if self.unica_adapter is not None:
            if covariates is None:
                raise ValueError("UniCA adapter is enabled but no covariates were provided.")
            if self.unica_past_pos == "pre":
                data_st = self.unica_adapter.fuse_past(data_st, covariates)
            if self.unica_with_future and self.unica_future_pos == "pre":
                data_st = self.unica_adapter.fuse_future(data_st, covariates)

        pfa_output = self.gpt(data_st, is_training=is_training)
        outputs = pfa_output.last_hidden_state

        if self.unica_adapter is not None:
            if self.unica_past_pos == "post":
                outputs = self.unica_adapter.fuse_past(outputs, covariates)
            if self.unica_with_future and self.unica_future_pos == "post":
                outputs = self.unica_adapter.fuse_future(outputs, covariates)

        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        outputs = self.regression_layer(outputs)

        if return_aux:
            return outputs, {
                "moe_loss": pfa_output.moe_loss,
                "attention_bias_shape": self.last_attention_bias_shape,
            }
        return outputs
