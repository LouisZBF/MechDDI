from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from .data import load_tensor_dataset, move_to_device, synthetic_batch, validate_batch
from .model import MechDDI, MechDDIConfig, multitask_loss
from .openddi import load_openddi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or smoke-test MechDDI")
    parser.add_argument("--data", type=Path, help="Path to a torch .pt tensor dictionary")
    parser.add_argument(
        "--openddi-root",
        type=Path,
        help="Path to the extracted MultiMData/datasets directory",
    )
    parser.add_argument(
        "--task",
        choices=("joint", "binary", "multiclass", "multilabel"),
        default="joint",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Read at most this many OpenDDI interaction rows",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--routing-channels", type=int, default=5)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--binary-weight", type=float, default=1.0)
    parser.add_argument("--multiclass-weight", type=float, default=0.2)
    parser.add_argument("--multilabel-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--save", type=Path, help="Optional checkpoint output path")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_config(batch: Dict[str, Tensor], args: argparse.Namespace) -> MechDDIConfig:
    return MechDDIConfig(
        molecule_dim=batch["molecule"].shape[-1],
        protein_dim=batch["protein_tokens"].shape[-1],
        path_dim=batch["path_tokens"].shape[-1],
        text_dim=batch["text_tokens"].shape[-1],
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        routing_channels=args.routing_channels,
        graph_layers=args.graph_layers,
        multiclass_classes=int(
            batch.get(
                "num_multiclass_classes",
                batch["multiclass_labels"].max() + 1,
            )
        ),
        multilabel_classes=batch["multilabel_labels"].shape[-1],
        router_temperature=args.temperature,
        dropout=args.dropout,
    )


def metrics(
    outputs: Dict[str, Tensor], batch: Dict[str, Tensor], task: str
) -> Dict[str, float]:
    binary = ((outputs["binary_logits"] >= 0) == batch["binary_labels"].bool()).float().mean()
    multiclass = (
        outputs["multiclass_logits"].argmax(dim=-1) == batch["multiclass_labels"]
    ).float().mean()
    multilabel = (
        (outputs["multilabel_logits"] >= 0) == batch["multilabel_labels"].bool()
    ).float().mean()
    scores = {
        "binary_acc": float(binary),
        "multiclass_acc": float(multiclass),
        "multilabel_acc": float(multilabel),
    }
    if task == "joint":
        return scores
    return {f"{task}_acc": scores[f"{task}_acc"]}


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data and args.openddi_root:
        raise ValueError("--data and --openddi-root cannot be used together")
    if args.openddi_root:
        if args.task == "joint":
            raise ValueError("--openddi-root requires a single --task")
        batch = load_openddi(args.openddi_root, args.task, args.max_pairs)
    elif args.data:
        batch = load_tensor_dataset(args.data)
    else:
        batch = synthetic_batch(seed=args.seed)
    validate_batch(batch)
    config = infer_config(batch, args)
    device = select_device(args.device)
    batch = move_to_device(batch, device)
    model = MechDDI(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"device={device} nodes={batch['molecule'].shape[0]} pairs={batch['pair_index'].shape[1]}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, parts = multitask_loss(
            outputs,
            batch,
            args.binary_weight,
            args.multiclass_weight,
            args.multilabel_weight,
            args.task,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        scores = metrics(outputs, batch, args.task)
        loss_text = " ".join(f"{name}={value.item():.4f}" for name, value in parts.items())
        score_text = ",".join(f"{name}={value:.3f}" for name, value in scores.items())
        print(
            f"epoch={epoch:03d} loss={loss.item():.4f} "
            f"{loss_text} {score_text}"
        )

    with torch.no_grad():
        routing = outputs["routing"][-1]
        print("mean_route=" + ",".join(f"{value:.3f}" for value in routing.mean(0).tolist()))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config.__dict__}, args.save)
        print(f"checkpoint={args.save}")


if __name__ == "__main__":
    main()
