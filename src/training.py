"""Train a frozen-backbone classifier and persist head-only artifacts."""

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES
from .data import build_loaders
from .model import DEFAULT_MODEL, build_model, choose_device

if TYPE_CHECKING:
    import zygo

    from dataset import MarcoFeatures


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Accumulate sample-weighted loss and a true-row, predicted-column matrix."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_samples = 0

    with torch.enable_grad() if training else torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device=device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            count = targets.numel()
            total_loss += loss.detach().item() * count
            total_samples += count
            indices = targets.detach().cpu() * num_classes + logits.detach().argmax(dim=1).cpu()
            confusion += torch.bincount(indices, minlength=num_classes**2).reshape(
                num_classes, num_classes
            )

    if total_samples == 0:
        split = "training" if training else "validation"
        raise ValueError(f"The {split} loader must contain at least one sample")

    matrix = confusion.to(torch.float64)
    true_positives = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = true_positives / support.clamp_min(1)
    f1 = 2 * true_positives / (support + predicted).clamp_min(1)
    return {
        "loss": total_loss / total_samples,
        "accuracy": true_positives.sum().item() / total_samples,
        "macro_f1": f1.mean().item(),
        "per_class_recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    dataset: "zygo.Dataset[MarcoFeatures]",
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    image_size: int | None = None,
) -> Path:
    """Train only the linear head and return the best validation checkpoint.

    The output directory must not already exist. Macro-F1 includes every MARCO
    class, assigning zero F1 to classes with no true or predicted samples. Ties
    retain the earlier checkpoint. Image-size validation belongs to build_model.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")

    random.seed(seed)
    # NumPy's legacy seed API accepts only unsigned 32-bit values.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resolved_device = choose_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Building {model} on {resolved_device}", flush=True)
    network, data_config = build_model(model, len(CLASS_NAMES), image_size=image_size)
    network.backbone.requires_grad_(False)
    network.to(resolved_device)
    train_loader, validation_loader = build_loaders(
        dataset, data_config, batch_size=batch_size, seed=seed
    )
    optimizer = torch.optim.AdamW(
        network.classifier.parameters(), lr=lr, weight_decay=weight_decay
    )
    class_to_idx = {name: index for index, name in enumerate(CLASS_NAMES)}
    config = {
        "model_name": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": str(resolved_device),
        "seed": seed,
        "image_size": image_size,
        "class_to_idx": class_to_idx,
        "data_config": data_config,
        "selection_metric": "validation.macro_f1",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    checkpoint_path = output_dir / "best.pt"
    best_f1 = -math.inf
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}: training", flush=True)
            train_metrics = _run_epoch(
                network, train_loader, resolved_device, len(CLASS_NAMES), optimizer
            )
            print(f"Epoch {epoch}/{epochs}: validation", flush=True)
            validation_metrics = _run_epoch(
                network, validation_loader, resolved_device, len(CLASS_NAMES)
            )
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
            metrics_file.flush()

            improved = validation_metrics["macro_f1"] > best_f1
            if improved:
                best_f1 = validation_metrics["macro_f1"]
                torch.save(
                    {
                        "model_name": model,
                        "class_to_idx": class_to_idx,
                        "data_config": data_config,
                        "classifier_state_dict": {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in network.classifier.state_dict().items()
                        },
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
            print(
                f"Epoch {epoch}/{epochs}: "
                f"train loss={train_metrics['loss']:.4f} "
                f"accuracy={train_metrics['accuracy']:.4f} "
                f"macro_f1={train_metrics['macro_f1']:.4f} | "
                f"validation loss={validation_metrics['loss']:.4f} "
                f"accuracy={validation_metrics['accuracy']:.4f} "
                f"macro_f1={validation_metrics['macro_f1']:.4f}"
                + (" (saved best.pt)" if improved else ""),
                flush=True,
            )

    return checkpoint_path
