"""Frozen pretrained backbone and trainable linear classification head."""

from typing import Any

import timm
import torch
from timm.data import resolve_model_data_config
from torch import nn

DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


class FrozenClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        num_features = getattr(backbone, "num_features", None)
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError("Backbone must expose a positive integer num_features")
        self.classifier = nn.Linear(num_features, num_classes)

    def train(self, mode: bool = True) -> "FrozenClassifier":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(inputs)
        return self.classifier(features)


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(device)


def build_model(
    model_name: str = DEFAULT_MODEL,
    num_classes: int = 4,
    *,
    image_size: int | None = None,
) -> tuple[FrozenClassifier, dict]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if image_size is not None and (type(image_size) is not int or image_size <= 0):
        raise ValueError("image_size must be a positive integer")
    options: dict[str, Any] = {} if image_size is None else {"img_size": image_size}
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, **options)
    if image_size is not None:
        patch_size = getattr(getattr(backbone, "patch_embed", None), "patch_size", None)
        if patch_size is not None and any(image_size % size for size in patch_size):
            raise ValueError(f"image_size must be a multiple of the patch size {patch_size}")
    config = resolve_model_data_config(backbone)
    if image_size is not None:
        config["input_size"] = (3, image_size, image_size)
    return FrozenClassifier(backbone, num_classes), config
