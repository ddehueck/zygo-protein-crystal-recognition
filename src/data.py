"""Stream decoded Zygo images with model-specific preprocessing."""

import random

import torch
import zygo
from timm.data import create_transform
from torch.utils.data import DataLoader, IterableDataset

from dataset import CLASS_NAMES, MarcoFeatures


class MarcoImages(IterableDataset):
    def __init__(self, dataset, transform, *, shuffle: bool, seed: int) -> None:
        self.dataset = dataset
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rows = iter(self.dataset)
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        # Shuffle a bounded buffer of lazy Pillow images rather than keeping the
        # full dataset or preprocessed tensors in memory.
        buffer = []
        if self.shuffle:
            for row in rows:
                if len(buffer) < 1024:
                    buffer.append(row)
                    continue
                index = rng.randrange(len(buffer))
                previous, buffer[index] = buffer[index], row
                yield self._prepare(previous)
            rng.shuffle(buffer)
            for row in buffer:
                yield self._prepare(row)
        else:
            for row in rows:
                yield self._prepare(row)

    def _prepare(self, row):
        if not 0 <= row.label < len(CLASS_NAMES):
            raise ValueError(f"Invalid MARCO label: {row.label}")
        return self.transform(row.image.convert("RGB")), row.label


def build_loaders(
    dataset: zygo.Dataset[MarcoFeatures],
    data_config: dict,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset.with_features(MarcoFeatures)
    splits = {name: dataset.where(split=name) for name in ("train", "validation")}
    for name, split in splits.items():
        if len(split) == 0:
            raise ValueError(f"Dataset has no rows with split={name!r}")
    if sum(len(split) for split in splits.values()) != len(dataset):
        raise ValueError("Dataset splits must be 'train' or 'validation'")
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for name, split in splits.items():
        training = name == "train"
        images = MarcoImages(
            split,
            create_transform(**data_config, is_training=training),
            shuffle=training,
            seed=seed,
        )
        # Zygo's Arrow reader is streamed in-process. Multiple workers would
        # duplicate rows unless the iterable were explicitly partitioned.
        loaders.append(DataLoader(images, batch_size=batch_size, num_workers=0, generator=generator))
    return loaders[0], loaders[1]
