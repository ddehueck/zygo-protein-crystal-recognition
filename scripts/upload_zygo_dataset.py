"""Upload the sampled MARCO dataset to Zygo Cloud."""

import logging
import os
from pathlib import Path


# Edit these settings before running this script.
HOST = "https://postnasal-staring-stapling.ngrok-free.dev/"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "marco" / "zygo-small"
DATASET_ID = "marco-small"
CONCURRENCY = 4

# Zygo reads the host when its cloud API module is imported.
os.environ["ZYGO_CLOUD_HOST"] = HOST.rstrip("/")

from zygo._internal.cloud.datasets import push_dataset


# export ZYGO_CLOUD_API_KEY=... & uv run scripts/upload_zygo_dataset.py
if __name__ == "__main__":
    if not os.environ.get("ZYGO_CLOUD_API_KEY"):
        raise RuntimeError("Set ZYGO_CLOUD_API_KEY in your environment before uploading.")
    if not (DATASET_PATH / "_manifest.json").is_file():
        raise FileNotFoundError(
            f"Missing dataset manifest in {DATASET_PATH}. Run scripts/sample_marco_dataset.py first."
        )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    push_dataset(str(DATASET_PATH), dataset_id=DATASET_ID, concurrency=CONCURRENCY)
    print(f"Uploaded {DATASET_ID} to {HOST}")
