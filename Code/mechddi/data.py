from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import Tensor


FEATURE_KEYS = {
    "molecule",
    "protein_tokens",
    "path_tokens",
    "text_tokens",
    "graph_edge_index",
    "pair_index",
}

LABEL_KEYS = {
    "binary_labels",
    "multiclass_labels",
    "multilabel_labels",
}


def make_undirected(edge_index: Tensor) -> Tensor:
    reverse = edge_index.flip(0)
    return torch.unique(torch.cat((edge_index, reverse), dim=1), dim=1)


def synthetic_batch(
    *,
    num_nodes: int = 48,
    num_pairs: int = 128,
    molecule_dim: int = 32,
    protein_dim: int = 24,
    path_dim: int = 20,
    text_dim: int = 28,
    protein_length: int = 9,
    path_length: int = 5,
    text_length: int = 7,
    multiclass_classes: int = 8,
    multilabel_classes: int = 12,
    seed: int = 7,
) -> Dict[str, Tensor]:
    """Build a small deterministic dataset for local checks."""

    generator = torch.Generator().manual_seed(seed)
    molecule = torch.randn(num_nodes, molecule_dim, generator=generator)
    protein = torch.randn(num_nodes, protein_length, protein_dim, generator=generator)
    path = torch.randn(num_nodes, path_length, path_dim, generator=generator)
    text = torch.randn(num_nodes, text_length, text_dim, generator=generator)

    # Start with a ring so every node has at least one neighbor.
    nodes = torch.arange(num_nodes)
    ring = torch.stack((nodes, torch.roll(nodes, shifts=-1)))
    random_edges = torch.randint(0, num_nodes, (2, 3 * num_nodes), generator=generator)
    graph_edges = make_undirected(torch.cat((ring, random_edges), dim=1))

    pairs = torch.randint(0, num_nodes, (2, num_pairs), generator=generator)
    equal = pairs[0] == pairs[1]
    pairs[1, equal] = (pairs[1, equal] + 1) % num_nodes

    latent = molecule[:, : min(12, molecule_dim)]
    pair_signal = latent[pairs[0]] * latent[pairs[1]]
    scalar = pair_signal.mean(dim=-1)
    binary = (scalar > scalar.median()).float()

    class_projection = torch.randn(
        pair_signal.shape[-1], multiclass_classes, generator=generator
    )
    multiclass = (pair_signal @ class_projection).argmax(dim=-1)

    label_projection = torch.randn(
        pair_signal.shape[-1], multilabel_classes, generator=generator
    )
    multilabel = (pair_signal @ label_projection > 0).float()

    return {
        "molecule": molecule,
        "protein_tokens": protein,
        "path_tokens": path,
        "text_tokens": text,
        "protein_mask": torch.ones(num_nodes, protein_length, dtype=torch.bool),
        "path_mask": torch.ones(num_nodes, path_length, dtype=torch.bool),
        "text_mask": torch.ones(num_nodes, text_length, dtype=torch.bool),
        "graph_edge_index": graph_edges,
        "pair_index": pairs,
        "binary_labels": binary,
        "multiclass_labels": multiclass,
        "multilabel_labels": multilabel,
    }


def validate_batch(batch: Dict[str, Tensor]) -> None:
    missing = (FEATURE_KEYS | LABEL_KEYS) - batch.keys()
    if missing:
        raise ValueError(f"dataset is missing keys: {sorted(missing)}")
    num_nodes = batch["molecule"].shape[0]
    if any(batch[key].shape[0] != num_nodes for key in
           ("protein_tokens", "path_tokens", "text_tokens")):
        raise ValueError("all node modalities must have the same first dimension")
    if batch["graph_edge_index"].shape[0] != 2 or batch["pair_index"].shape[0] != 2:
        raise ValueError("graph_edge_index and pair_index must have shape [2, E]")
    if int(batch["graph_edge_index"].max()) >= num_nodes:
        raise ValueError("graph_edge_index contains an invalid node id")


def load_tensor_dataset(path: str | Path) -> Dict[str, Tensor]:
    try:
        batch = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # weights_only was added after PyTorch 2.0.
        batch = torch.load(Path(path), map_location="cpu")
    if not isinstance(batch, dict):
        raise TypeError("dataset file must contain a dictionary of tensors")
    validate_batch(batch)
    return batch


def move_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
