"""Build one Zygo dataset with row-level splits from extracted MARCO JPEGs."""

from pathlib import Path
import sys

import zygo

# Allow running this file directly as well as importing it as a module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import CLASS_NAMES, MarcoFeatures


# Edit these settings before running this script.
IMAGES_PATH = ROOT / "data" / "marco" / "images"
DATASET_PATH = ROOT / "data" / "marco" / "zygo"
SPLITS = ("train", "validation")


def build_dataset(
    images_path: Path = IMAGES_PATH,
    dataset_path: Path = DATASET_PATH,
) -> None:
    images_path = Path(images_path)
    dataset_path = Path(dataset_path)
    if dataset_path.exists() and (not dataset_path.is_dir() or any(dataset_path.iterdir())):
        raise FileExistsError(f"Output must be an empty directory: {dataset_path}")
    files = {}
    # Check both splits before writing anything. Never overwrite existing data.
    for split in SPLITS:
        source = images_path / split
        if not source.is_dir():
            raise FileNotFoundError(
                f"Missing {source}. Run scripts/prepare_marco.py first."
            )
        unknown = {path.name for path in source.iterdir() if path.is_dir()} - set(CLASS_NAMES)
        if unknown:
            raise ValueError(f"Unknown class folders in {source}: {sorted(unknown)}")

        for label, name in enumerate(CLASS_NAMES):
            paths = sorted((source / name).glob("*.jpg"))
            if not paths:
                raise ValueError(f"No JPEG images found in {source / name}")
            files[split, label] = paths

    total = 0
    with zygo.Dataset.builder(
        dataset_path, schema=MarcoFeatures
    ) as builder:
        for split in SPLITS:
            for label, name in enumerate(CLASS_NAMES):
                paths = files[split, label]
                for path in paths:
                    # Store the original JPEG bytes without decoding or re-encoding.
                    builder.add({"image": path.read_bytes(), "label": label, "split": split})
                    total += 1
                    if total % 10_000 == 0:
                        print(f"{split}: {total:,} images written", flush=True)
                print(f"{split}/{name}: {len(paths):,} images", flush=True)
    print(f"Built {dataset_path}: {total:,} images", flush=True)


if __name__ == "__main__":
    build_dataset()
