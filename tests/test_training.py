"""Offline regression tests for the complete Zygo training path."""

import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
import zygo
from PIL import Image
from torch import nn

from dataset import CLASS_NAMES, MarcoFeatures
from main import Prediction, TrainingParams, app
from src.data import build_loaders
from src.model import DEFAULT_MODEL, FrozenClassifier, build_model
from src.training import train_model

DATA_CONFIG = {
    "input_size": (3, 28, 28),
    "interpolation": "bicubic",
    "mean": (0.5, 0.5, 0.5),
    "std": (0.5, 0.5, 0.5),
    "crop_pct": 1.0,
}


class TinyBackbone(nn.Module):
    num_features = 6

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, self.num_features, 1)

    def forward(self, inputs):
        return self.projection(inputs).mean(dim=(2, 3))


def tiny_model(*args, **kwargs):
    # Recreating a pretrained backbone must yield the same weights on load.
    with torch.random.fork_rng():
        torch.manual_seed(7)
        return TinyBackbone()


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_path = self.root / "dataset"
        image = BytesIO()
        Image.new("RGB", (32, 32), "white").save(image, format="JPEG")
        with zygo.Dataset.builder(self.dataset_path, schema=MarcoFeatures) as builder:
            for split in ("train", "validation"):
                for label in range(len(CLASS_NAMES)):
                    builder.add({"image": image.getvalue(), "split": split, "label": label})
        self.dataset = zygo.Dataset.open(str(self.dataset_path), features=MarcoFeatures)
        self.addCleanup(self.temporary.cleanup)

    def test_backbone_remains_frozen_in_training(self):
        model = FrozenClassifier(TinyBackbone(), 4)
        original = {name: value.clone() for name, value in model.backbone.state_dict().items()}
        original_head = model.classifier.weight.detach().clone()
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.classifier.training)
        optimizer = torch.optim.AdamW(model.classifier.parameters())
        model(torch.ones(2, 3, 28, 28)).sum().backward()
        optimizer.step()
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, original[name])
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertFalse(torch.equal(original_head, model.classifier.weight))

    def test_loaders_preserve_splits_and_all_labels(self):
        loaders = build_loaders(self.dataset, DATA_CONFIG, batch_size=3, seed=42)
        for loader in loaders:
            labels = []
            for images, targets in loader:
                self.assertEqual(tuple(images.shape[1:]), (3, 28, 28))
                labels.extend(targets.tolist())
            self.assertEqual(sorted(labels), list(range(4)))
        with self.assertRaisesRegex(ValueError, "validation"):
            build_loaders(self.dataset.where(split="train"), DATA_CONFIG, batch_size=2, seed=42)

    @patch("src.model.resolve_model_data_config", return_value=DATA_CONFIG.copy())
    @patch("src.model.timm.create_model", side_effect=tiny_model)
    def test_zygo_train_load_infer(self, factory, config):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            store = zygo.train(
                model=app,
                dataset=self.dataset_path,
                params=TrainingParams(epochs=1, batch_size=2, image_size=28, device="cpu"),
            )
            checkpoint = torch.load(BytesIO(store.get("best.pt")), weights_only=True)
            self.assertEqual(checkpoint["class_to_idx"], dict(zip(CLASS_NAMES, range(4))))
            self.assertEqual(set(checkpoint["classifier_state_dict"]), {"weight", "bias"})
            metrics = json.loads(store.get("metrics.jsonl"))
            self.assertEqual(metrics["epoch"], 1)
            self.assertEqual(sum(map(sum, metrics["validation"]["confusion_matrix"])), 4)
            self.assertEqual(json.loads(store.get("params.json"))["image_size"], 28)
            json.loads(store.get("config.json"))
            classifier = app.run_load(store)
            result = app.run_infer(classifier, Image.new("RGB", (32, 32), "white"))
            self.assertIsInstance(result, Prediction)
            prediction = cast(Prediction, result)
            self.assertEqual(prediction.class_name, CLASS_NAMES[prediction.label])
            self.assertGreaterEqual(prediction.confidence, 0)
            self.assertLessEqual(prediction.confidence, 1)
            self.assertTrue(factory.call_args.kwargs["pretrained"])
        finally:
            os.chdir(previous)

    def test_rejects_invalid_parameters_and_existing_output(self):
        invalid_params: list[dict[str, Any]] = [
            {"epochs": 0}, {"batch_size": 0}, {"lr": float("nan")}, {"weight_decay": -1}
        ]
        for kwargs in invalid_params:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_model(self.dataset, self.root / "run", **kwargs)
        with self.assertRaises(FileExistsError):
            train_model(self.dataset, self.root, device="cpu")
        with self.assertRaises(ValueError):
            build_model(DEFAULT_MODEL, image_size=0)


if __name__ == "__main__":
    unittest.main()
