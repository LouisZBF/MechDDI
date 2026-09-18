import csv

import torch

from mechddi.model import MechDDI, MechDDIConfig, multitask_loss
from mechddi.openddi import load_openddi


def test_openddi_binary_adapter(tmp_path) -> None:
    emb_dir = tmp_path / "emb"
    matrix_dir = tmp_path / "matrix"
    emb_dir.mkdir()
    matrix_dir.mkdir()

    torch.save({1: torch.randn(3), 2: torch.randn(3)}, emb_dir / "smiles_embeddings.pt")
    torch.save({1: torch.randn(4), 2: torch.randn(4)}, emb_dir / "path_embeddings.pt")
    torch.save({1: torch.randn(5), 2: torch.randn(5)}, emb_dir / "text_embeddings.pt")
    with (matrix_dir / "combined ddi multi.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id1", "id2", "ddi", "label"))
        writer.writeheader()
        writer.writerow({"id1": 1, "id2": 2, "ddi": 1, "label": 1})

    batch = load_openddi(tmp_path, "binary")
    config = MechDDIConfig(
        molecule_dim=3,
        protein_dim=1,
        path_dim=4,
        text_dim=5,
        hidden_dim=8,
        attention_heads=2,
        routing_channels=2,
        graph_layers=1,
        multiclass_classes=1,
        multilabel_classes=1,
        dropout=0.0,
    )
    outputs = MechDDI(config)(batch)
    loss, parts = multitask_loss(outputs, batch, task="binary")

    assert batch["pair_index"].shape == (2, 1)
    assert set(parts) == {"binary"}
    assert torch.isfinite(loss)
