from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor

from .data import make_undirected, validate_batch


TASK_FILES = {
    "binary": "combined ddi multi.csv",
    "multiclass": "combined ddi multi.csv",
    "multilabel": "combined ddi multilabel.csv",
}


def _load_embedding_dict(path: Path) -> Dict[int, Tensor]:
    try:
        values = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        values = torch.load(path, map_location="cpu")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"invalid embedding file: {path}")
    return {int(key): value.float() for key, value in values.items()}


def _read_rows(path: Path, max_pairs: int | None) -> List[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    if not rows:
        raise ValueError(f"no interaction rows found in {path}")
    return rows


def _stack_features(
    drug_ids: Iterable[int],
    smiles: Dict[int, Tensor],
    path: Dict[int, Tensor],
    text: Dict[int, Tensor],
) -> Tuple[List[int], Tensor, Tensor, Tensor]:
    common = set(smiles) & set(path) & set(text)
    ordered_ids = sorted(set(drug_ids))
    missing = set(ordered_ids) - common
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"missing embeddings for drug ids: {sample}")
    return (
        ordered_ids,
        torch.stack([smiles[drug_id] for drug_id in ordered_ids]),
        torch.stack([path[drug_id] for drug_id in ordered_ids]).unsqueeze(1),
        torch.stack([text[drug_id] for drug_id in ordered_ids]).unsqueeze(1),
    )


def load_openddi(
    root: str | Path,
    task: str,
    max_pairs: int | None = None,
) -> Dict[str, Tensor]:
    """Load the OpenDDI files distributed in MultiMData/datasets."""

    if task not in TASK_FILES:
        raise ValueError(f"unsupported OpenDDI task: {task}")
    root = Path(root)
    emb_dir = root / "emb"
    matrix_path = root / "matrix" / TASK_FILES[task]
    for path in (
        emb_dir / "smiles_embeddings.pt",
        emb_dir / "path_embeddings.pt",
        emb_dir / "text_embeddings.pt",
        matrix_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = _read_rows(matrix_path, max_pairs)
    smiles = _load_embedding_dict(emb_dir / "smiles_embeddings.pt")
    path = _load_embedding_dict(emb_dir / "path_embeddings.pt")
    text = _load_embedding_dict(emb_dir / "text_embeddings.pt")

    raw_pairs = [(int(row["id1"]), int(row["id2"])) for row in rows]
    drug_ids, molecule, path_tokens, text_tokens = _stack_features(
        (drug_id for pair in raw_pairs for drug_id in pair), smiles, path, text
    )
    index = {drug_id: position for position, drug_id in enumerate(drug_ids)}
    pair_index = torch.tensor(
        [(index[left], index[right]) for left, right in raw_pairs], dtype=torch.long
    ).t().contiguous()

    num_pairs = len(rows)
    if task == "binary":
        binary_labels = torch.tensor(
            [int(row["label"]) for row in rows], dtype=torch.float32
        )
        multiclass_labels = torch.zeros(num_pairs, dtype=torch.long)
        multilabel_labels = torch.zeros(num_pairs, 1)
        positive_edges = pair_index[:, binary_labels.bool()]
        graph_edge_index = make_undirected(positive_edges)
    elif task == "multiclass":
        binary_labels = torch.ones(num_pairs)
        multiclass_labels = torch.tensor(
            [int(row["ddi"]) - 1 for row in rows], dtype=torch.long
        )
        multilabel_labels = torch.zeros(num_pairs, 1)
        graph_edge_index = make_undirected(pair_index)
    else:
        binary_labels = torch.ones(num_pairs)
        multiclass_labels = torch.zeros(num_pairs, dtype=torch.long)
        multilabel_labels = torch.tensor(
            [[int(value) for value in row["ddi"].split(",")] for row in rows],
            dtype=torch.float32,
        )
        graph_edge_index = make_undirected(pair_index)

    if graph_edge_index.numel() == 0:
        graph_edge_index = make_undirected(pair_index)

    num_nodes = len(drug_ids)
    batch = {
        "drug_ids": torch.tensor(drug_ids, dtype=torch.long),
        "molecule": molecule,
        # Protein and 3D embeddings are not present in this archive. A shared
        # placeholder keeps the missing modality explicit in the model input.
        "protein_tokens": torch.zeros(num_nodes, 1, 1),
        "path_tokens": path_tokens,
        "text_tokens": text_tokens,
        "protein_mask": torch.ones(num_nodes, 1, dtype=torch.bool),
        "path_mask": torch.ones(num_nodes, 1, dtype=torch.bool),
        "text_mask": torch.ones(num_nodes, 1, dtype=torch.bool),
        "graph_edge_index": graph_edge_index,
        "pair_index": pair_index,
        "binary_labels": binary_labels,
        "multiclass_labels": multiclass_labels,
        "multilabel_labels": multilabel_labels,
        "num_multiclass_classes": torch.tensor(
            167 if task == "multiclass" else 1
        ),
    }
    validate_batch(batch)
    return batch
