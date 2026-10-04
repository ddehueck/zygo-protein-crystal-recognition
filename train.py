"""Train the Zygo model using settings edited directly in this file."""

from pathlib import Path

import zygo

from main import TrainingParams, app
from src.model import DEFAULT_MODEL

DATASET_PATH = Path(__file__).resolve().parent / "data" / "marco" / "zygo-small"
PARAMS = TrainingParams(
    model=DEFAULT_MODEL,
    epochs=10,
    batch_size=8,
    image_size=224,
    lr=1e-3,
    weight_decay=1e-4,
    device="auto",
    seed=42,
)


if __name__ == "__main__":
    store = zygo.train(model=app, dataset=DATASET_PATH, params=PARAMS)
    print(f"Model store: {store.root}")
