"""Create one class-balanced Zygo MARCO subset, preserving row-level splits."""

from pathlib import Path
import sys

import zygo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import CLASS_NAMES, MarcoFeatures


# Edit these settings before running this script.
DATASET_PATH = ROOT / "data" / "marco" / "zygo"
SAMPLE_PATH = ROOT / "data" / "marco" / "zygo-small"
TRAIN_PER_CLASS = 1_000
VALIDATION_SIZE = 1_250  # Total across all four classes, not per class.
SEED = 42
SHARD_SIZE_MB = 128


def sample_dataset(
    dataset_path: Path = DATASET_PATH,
    sample_path: Path = SAMPLE_PATH,
    train_per_class: int = TRAIN_PER_CLASS,
    validation_size: int = VALIDATION_SIZE,
    seed: int = SEED,
    shard_size_mb: int = SHARD_SIZE_MB,
) -> None:
    if type(train_per_class) is not int or train_per_class <= 0:
        raise ValueError("train_per_class must be a positive integer")
    if type(validation_size) is not int or validation_size < len(CLASS_NAMES):
        raise ValueError("validation_size must include at least one image per class")
    dataset_path = Path(dataset_path)
    sample_path = Path(sample_path)
    per_class, remainder = divmod(validation_size, len(CLASS_NAMES))
    quotas = {
        "train": [train_per_class] * len(CLASS_NAMES),
        "validation": [per_class + (label < remainder) for label in range(len(CLASS_NAMES))],
    }
    if sample_path.exists() and (not sample_path.is_dir() or any(sample_path.iterdir())):
        raise FileExistsError(f"Output must be an empty directory: {sample_path}")
    source = zygo.Dataset.open(str(dataset_path), features=MarcoFeatures)
    samples = {}
    # Prepare both splits before writing, including checking each class has enough rows.
    for split, counts in quotas.items():

        groups = []
        for label, (name, count) in enumerate(zip(CLASS_NAMES, counts, strict=True)):
            group = source.where(split=split, label=label)
            available = len(group)
            if available < count:
                raise ValueError(
                    f"{split}/{name}: requested {count:,} images, only {available:,} available"
                )
            groups.append(group.sample(per_group=count, group_by="label", seed=seed + label))
        samples[split] = zygo.Dataset.concat(*groups)

    sample = zygo.Dataset.concat(*samples.values())
    # Stream encoded rows without decoding images or changing their JPEG bytes.
    sample.write(sample_path, shard_size_mb=shard_size_mb)
    print(f"Built {sample_path}: {len(sample):,} images", flush=True)
    for split, counts in quotas.items():
        print(f"{split} class counts: {dict(zip(CLASS_NAMES, counts, strict=True))}")


if __name__ == "__main__":
    sample_dataset()
