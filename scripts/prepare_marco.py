"""Extract MARCO JPEG TFRecords into PyTorch/Hugging Face ImageFolder directories."""

import argparse
import re
import struct
from collections import Counter
from pathlib import Path

from tfrecord import example_pb2
from tfrecord.reader import extract_feature_dict
from tfrecord.writer import TFRecordWriter

DATA = Path(__file__).resolve().parents[1] / "data" / "marco"
OUTPUT = DATA / "images"
CLASSES = {0: "clear", 1: "crystals", 2: "other", 3: "precipitate"}
# MARCO calls its validation split "test".
SPLITS = {"train": ("train", 407, 415775), "test": ("validation", 46, 47029)}
FEATURES = {"image/encoded": "byte", "image/id": "int", "image/class/label": "int"}


def find_shards(directory, split, expected):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {directory}. Run uv run scripts/fetch_marco.py first.")
    pattern = re.compile(rf"{split}-(\d{{5}})-of-{expected:05d}")
    shards = sorted(path for path in directory.iterdir() if path.is_file() and pattern.fullmatch(path.name))
    expected_names = [f"{split}-{index:05d}-of-{expected:05d}" for index in range(1, expected + 1)]
    if [path.name for path in shards] != expected_names:
        raise ValueError(f"Incomplete dataset in {directory}: expected {expected} shards, found {len(shards)}. Re-run the downloader.")
    return shards


def scalar(record, name):
    values = record[name]
    if len(values) != 1:
        raise ValueError(f"Expected one value for {name}, got {len(values)}")
    return int(values[0])


def read_records(stream, size):
    # Own the file handle in the caller and validate both TFRecord checksums.
    while stream.tell() < size:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8 or stream.read(4) != TFRecordWriter.masked_crc(length_bytes):
            raise ValueError("Invalid TFRecord length checksum or truncated header")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > size - stream.tell() - 4:
            raise ValueError("Truncated TFRecord payload")
        payload = stream.read(length)
        if stream.read(4) != TFRecordWriter.masked_crc(payload):
            raise ValueError("Invalid TFRecord payload checksum")
        example = example_pb2.Example()
        example.ParseFromString(payload)
        yield extract_feature_dict(example.features, FEATURES, {"byte": "bytes_list", "int": "int64_list"})


def extract_shard(shard, output, seen_ids, counts):
    saved = skipped = 0
    record_number = 0
    try:
        with shard.open("rb") as stream:
            for record_number, record in enumerate(read_records(stream, shard.stat().st_size), start=1):
                image_id = scalar(record, "image/id")
                label = scalar(record, "image/class/label")
                if image_id < 0:
                    raise ValueError(f"Invalid image ID: {image_id}")
                if image_id in seen_ids:
                    raise ValueError(f"Duplicate image ID: {image_id}")
                if label not in CLASSES:
                    raise ValueError(f"Unknown MARCO label: {label}")
                encoded = record["image/encoded"]
                if not isinstance(encoded, bytes) or not encoded.startswith(b"\xff\xd8"):
                    raise ValueError(f"Image {image_id} does not contain JPEG bytes")
                seen_ids.add(image_id)

                destination = output / CLASSES[label] / f"{image_id}.jpg"
                if destination.exists():
                    if destination.read_bytes() != encoded:
                        raise FileExistsError(f"Existing image differs from the TFRecord: {destination}")
                    skipped += 1
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_suffix(".jpg.part")
                    partial.write_bytes(encoded)
                    partial.replace(destination)
                    saved += 1
                counts[CLASSES[label]] += 1
    except Exception as error:
        raise RuntimeError(f"Failed in {shard}, near record {max(record_number, 1)}: {error}") from error
    return saved, skipped


def prepare(split):
    source_splits = ("train", "test") if split == "both" else (split,)
    # Check all requested splits before starting a potentially lengthy extraction.
    shards_by_split = {
        source: find_shards(DATA / f"{source}-jpg", source, SPLITS[source][1])
        for source in source_splits
    }
    seen_ids = set()
    for source in source_splits:
        destination_split, _, expected_images = SPLITS[source]
        output = OUTPUT / destination_split
        for name in CLASSES.values():
            (output / name).mkdir(parents=True, exist_ok=True)
        counts = Counter()
        saved = skipped = 0
        shards = shards_by_split[source]
        for index, shard in enumerate(shards, start=1):
            new, existing = extract_shard(shard, output, seen_ids, counts)
            saved += new
            skipped += existing
            print(f"{destination_split}: shard {index}/{len(shards)}, {saved:,} saved, {skipped:,} already extracted", flush=True)
        if sum(counts.values()) != expected_images:
            raise ValueError(f"{destination_split}: expected {expected_images:,} images, found {sum(counts.values()):,}")
        print(f"{destination_split}: {dict(counts)}", flush=True)
    print(f"Images ready in {OUTPUT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract MARCO JPEG TFRecords into data/marco/images/")
    parser.add_argument("--split", choices=("both", "train", "test"), default="both")
    args = parser.parse_args()
    prepare(args.split)


if __name__ == "__main__":
    main()
