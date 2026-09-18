import torch

from mechddi.data import synthetic_batch, validate_batch
from mechddi.model import MechDDI, MechDDIConfig, multitask_loss


def test_forward_backward() -> None:
    batch = synthetic_batch(
        num_nodes=16,
        num_pairs=24,
        multiclass_classes=4,
        multilabel_classes=6,
        seed=11,
    )
    validate_batch(batch)
    config = MechDDIConfig(
        molecule_dim=batch["molecule"].shape[-1],
        protein_dim=batch["protein_tokens"].shape[-1],
        path_dim=batch["path_tokens"].shape[-1],
        text_dim=batch["text_tokens"].shape[-1],
        hidden_dim=16,
        attention_heads=4,
        routing_channels=5,
        graph_layers=2,
        multiclass_classes=4,
        multilabel_classes=6,
        dropout=0.0,
    )
    model = MechDDI(config)
    outputs = model(batch)
    loss, _ = multitask_loss(outputs, batch)
    loss.backward()

    assert outputs["binary_logits"].shape == (24,)
    assert outputs["multiclass_logits"].shape == (24, 4)
    assert outputs["multilabel_logits"].shape == (24, 6)
    assert len(outputs["routing"]) == 2
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

