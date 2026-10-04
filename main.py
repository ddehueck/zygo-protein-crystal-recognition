"""Zygo model hooks for a frozen DINOv2 protein-crystal classifier."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, cast

import torch
import zygo
from PIL import Image
from pydantic import BaseModel, Field
from timm.data import create_transform

from dataset import MarcoFeatures
from src.model import DEFAULT_MODEL, FrozenClassifier, build_model, choose_device
from src.training import train_model


@dataclass
class CrystalClassifier:
    model: FrozenClassifier
    transform: Callable[[Image.Image], torch.Tensor]
    classes: tuple[str, ...]
    device: torch.device


class TrainingParams(zygo.HyperParams):
    model: str = DEFAULT_MODEL
    epochs: int = Field(default=10, gt=0, strict=True)
    batch_size: int = Field(default=16, gt=0, strict=True)
    image_size: int | None = Field(default=None, gt=0, strict=True)
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    seed: int = Field(default=42, strict=True)


class Prediction(BaseModel):
    label: int
    class_name: str
    confidence: float


app = zygo.Model("protein-crystal-recognition")


@app.train
def train(
    dataset: zygo.Dataset[MarcoFeatures], *, params: TrainingParams, ctx: zygo.TrainingContext
) -> None:
    # Stage the trainer's files locally, then persist them independently of that path.
    with TemporaryDirectory() as temporary:
        checkpoint = train_model(dataset, Path(temporary) / "run", **params.model_dump())
        for name in ("best.pt", "config.json", "metrics.jsonl"):
            ctx.store.put(name, (checkpoint.parent / name).read_bytes())
        ctx.store.put("params.json", params.model_dump_json(indent=2).encode())


@app.load
def load(store: zygo.ModelStore) -> CrystalClassifier:
    checkpoint = torch.load(BytesIO(store.get("best.pt")), map_location="cpu", weights_only=True)
    mapping = checkpoint["class_to_idx"]
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Checkpoint class indices must be contiguous starting at zero")
    classes = tuple(name for name, index in sorted(mapping.items(), key=lambda item: item[1]))
    config = checkpoint["data_config"]
    model, _ = build_model(
        checkpoint["model_name"], len(mapping), image_size=config["input_size"][-1]
    )
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    device = choose_device("auto")
    model.to(device).eval()
    return CrystalClassifier(
        model=model,
        transform=create_transform(**config, is_training=False),
        classes=classes,
        device=device,
    )


@app.infer
def infer(model: CrystalClassifier, image: zygo.Image) -> Prediction:
    """Classify one Pillow image with the checkpoint's evaluation preprocessing."""
    inputs = model.transform(cast(Image.Image, image).convert("RGB")).unsqueeze(0).to(model.device)
    with torch.inference_mode():
        probabilities = model.model(inputs).softmax(dim=1)
        confidence, label = probabilities[0].max(dim=0)
    index = label.item()
    return Prediction(label=index, class_name=model.classes[index], confidence=confidence.item())
