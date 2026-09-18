from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MechDDIConfig:
    molecule_dim: int
    protein_dim: int
    path_dim: int
    text_dim: int
    hidden_dim: int = 128
    attention_heads: int = 4
    routing_channels: int = 5
    graph_layers: int = 2
    multiclass_classes: int = 8
    multilabel_classes: int = 12
    router_temperature: float = 0.1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.routing_channels < 1 or self.graph_layers < 1:
            raise ValueError("routing_channels and graph_layers must be positive")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")


class CascadeMultimodalFusion(nn.Module):
    """Fuse molecular, protein, pathway and text features in cascade order."""

    def __init__(self, config: MechDDIConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.molecule_projection = nn.Linear(config.molecule_dim, d)
        self.protein_projection = nn.Linear(config.protein_dim, d)
        self.path_projection = nn.Linear(config.path_dim, d)
        self.text_projection = nn.Linear(config.text_dim, d)
        self.micro_norm = nn.LayerNorm(d)
        self.macro_norm = nn.LayerNorm(d)
        self.output_norm = nn.LayerNorm(d)
        self.target_attention = nn.MultiheadAttention(
            d, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.macro_attention = nn.MultiheadAttention(
            d, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _padding_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.dtype != torch.bool:
            raise TypeError("token masks must be boolean")
        # Input masks mark valid positions, while MultiheadAttention expects padding.
        return ~mask

    def forward(
        self,
        molecule: Tensor,
        protein_tokens: Tensor,
        path_tokens: Tensor,
        text_tokens: Tensor,
        protein_mask: Optional[Tensor] = None,
        path_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        micro = self.micro_norm(self.molecule_projection(molecule))
        query = micro.unsqueeze(1)

        protein = self.protein_projection(protein_tokens)
        target, _ = self.target_attention(
            query,
            protein,
            protein,
            key_padding_mask=self._padding_mask(protein_mask),
            need_weights=False,
        )
        target = target.squeeze(1)

        # Path and text features share the macro-level attention sequence.
        path = self.path_projection(path_tokens)
        text = self.text_projection(text_tokens)
        macro = self.macro_norm(torch.cat((path, text), dim=1))
        macro_mask = None
        if path_mask is not None or text_mask is not None:
            if path_mask is None:
                path_mask = torch.ones(path.shape[:2], dtype=torch.bool, device=path.device)
            if text_mask is None:
                text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
            macro_mask = torch.cat((path_mask, text_mask), dim=1)

        omega, _ = self.macro_attention(
            target.unsqueeze(1),
            macro,
            macro,
            key_padding_mask=self._padding_mask(macro_mask),
            need_weights=False,
        )
        return self.output_norm(target + self.dropout(omega.squeeze(1)))


class DynamicRoutingGraphLayer(nn.Module):
    """Graph layer with edge-dependent routing over mechanism bases."""

    def __init__(self, hidden_dim: int, channels: int, temperature: float, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.temperature = temperature
        self.router = nn.Linear(hidden_dim, channels)
        self.bases = nn.Parameter(torch.empty(channels, hidden_dim, hidden_dim))
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.bases)

    def forward(self, node_states: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        source, target = edge_index.long()
        num_nodes = node_states.shape[0]

        affinity = self.router(node_states[source] * node_states[target])
        routing = F.softmax(affinity / self.temperature, dim=-1)

        # Apply every basis first, then mix the messages with edge routing weights.
        basis_messages = torch.einsum("koi,ei->eko", self.bases, node_states[source])
        messages = torch.einsum("ek,ekd->ed", routing, basis_messages)

        degree = torch.bincount(target, minlength=num_nodes).to(node_states.dtype).clamp_min_(1)
        normalization = (degree[source] * degree[target]).rsqrt().unsqueeze(-1)
        aggregated = node_states.new_zeros(node_states.shape)
        aggregated.index_add_(0, target, messages * normalization)

        updated = self.self_projection(node_states) + aggregated
        return self.dropout(F.leaky_relu(updated, negative_slope=0.01)), routing


class MechDDI(nn.Module):
    """Hierarchical multimodal encoder with mechanism-aware graph propagation."""

    def __init__(self, config: MechDDIConfig) -> None:
        super().__init__()
        self.config = config
        self.fusion = CascadeMultimodalFusion(config)
        self.graph_layers = nn.ModuleList(
            DynamicRoutingGraphLayer(
                config.hidden_dim,
                config.routing_channels,
                config.router_temperature,
                config.dropout,
            )
            for _ in range(config.graph_layers)
        )
        pair_dim = 2 * config.hidden_dim
        self.binary_head = nn.Linear(pair_dim, 1)
        self.multiclass_head = nn.Linear(pair_dim, config.multiclass_classes)
        self.multilabel_head = nn.Linear(pair_dim, config.multilabel_classes)

    def encode_nodes(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        states = self.fusion(
            batch["molecule"],
            batch["protein_tokens"],
            batch["path_tokens"],
            batch["text_tokens"],
            batch.get("protein_mask"),
            batch.get("path_mask"),
            batch.get("text_mask"),
        )
        routing_history = []
        for layer in self.graph_layers:
            states, routing = layer(states, batch["graph_edge_index"])
            routing_history.append(routing)
        return states, tuple(routing_history)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor | Tuple[Tensor, ...]]:
        states, routing_history = self.encode_nodes(batch)
        left, right = batch["pair_index"].long()
        pair = torch.cat((states[left], states[right]), dim=-1)
        return {
            "binary_logits": self.binary_head(pair).squeeze(-1),
            "multiclass_logits": self.multiclass_head(pair),
            "multilabel_logits": self.multilabel_head(pair),
            "node_embeddings": states,
            "routing": routing_history,
        }


def multitask_loss(
    outputs: Dict[str, Tensor | Tuple[Tensor, ...]],
    batch: Dict[str, Tensor],
    binary_weight: float = 1.0,
    multiclass_weight: float = 0.2,
    multilabel_weight: float = 0.05,
    task: str = "joint",
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute the weighted loss for the three prediction heads."""

    losses = {
        "binary": F.binary_cross_entropy_with_logits(
            outputs["binary_logits"], batch["binary_labels"].float()
        ),
        "multiclass": F.cross_entropy(
            outputs["multiclass_logits"], batch["multiclass_labels"].long()
        ),
        "multilabel": F.binary_cross_entropy_with_logits(
            outputs["multilabel_logits"], batch["multilabel_labels"].float()
        ),
    }
    weights = {
        "binary": binary_weight,
        "multiclass": multiclass_weight,
        "multilabel": multilabel_weight,
    }
    selected = tuple(weights) if task == "joint" else (task,)
    if any(name not in weights for name in selected):
        raise ValueError(f"unsupported task: {task}")
    total = sum(weights[name] * losses[name] for name in selected)
    return total, {name: losses[name] for name in selected}
