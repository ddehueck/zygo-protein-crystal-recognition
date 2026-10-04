"""Download MARCO's JPEG-encoded TFRecord shards from https://marco.ccr.buffalo.edu/download."""

import argparse
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE_URL = "https://marco.ccr.buffalo.edu/data/"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "marco"
LABELS = ("labels-norm.csv", "labels-raw.csv", "sources.csv")
TIMEOUT = 60


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def list_shards(split, directory):
    url = urljoin(BASE_URL, f"{directory}/")
    with urlopen(url, timeout=TIMEOUT) as response:
        parser = Links()
        parser.feed(response.read().decode("utf-8"))

    pattern = re.compile(rf"{split}-(\d{{5}})-of-(\d{{5}})")
    shards = []
    numbers = []
    counts = set()
    for href in sorted(parser.hrefs):
        match = pattern.fullmatch(href)
        if match is not None:
            shards.append(href)
            numbers.append(int(match.group(1)))
            counts.add(int(match.group(2)))
    if not shards:
        raise ValueError(f"No {split} shards found at {url}")
    if len(counts) != 1 or len(shards) != next(iter(counts)) or len(set(shards)) != len(shards):
        raise ValueError(f"Incomplete shard listing at {url}")
    if numbers != list(range(1, len(shards) + 1)):
        raise ValueError(f"Missing shard numbers at {url}")
    return [(urljoin(url, name), DESTINATION / directory / name) for name in shards]


def download(url, destination):
    with urlopen(Request(url, method="HEAD"), timeout=TIMEOUT) as response:
        length = response.headers.get("Content-Length")
    if length is None or int(length) <= 0:
        raise ValueError(f"Missing or invalid Content-Length for {url}")
    size = int(length)

    if destination.exists():
        if destination.stat().st_size == size:
            print(f"Already downloaded: {destination.relative_to(DESTINATION)}", flush=True)
            return
        raise ValueError(f"Existing file has unexpected size: {destination}. Move it aside before retrying.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Partial file is larger than the source: {partial}")
    if offset == size:
        partial.replace(destination)
        return

    request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    print(f"Downloading: {destination.relative_to(DESTINATION)} ({offset:,}/{size:,} bytes)", flush=True)

    with urlopen(request, timeout=TIMEOUT) as response:
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not re.fullmatch(rf"bytes {offset}-\d+/{size}", content_range):
                raise ValueError(f"Unexpected Content-Range for {url}: {content_range}")
        elif response.status != 200:
            raise ValueError(f"Unexpected HTTP status {response.status} for {url}")
        # If the server ignores Range, restart rather than append duplicate data.
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)

    if partial.stat().st_size != size:
        raise IOError(f"Incomplete download: {partial} ({partial.stat().st_size:,}/{size:,} bytes)")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description="Download MARCO JPEG-encoded TFRecords and label metadata into data/marco/")
    parser.add_argument("--split", choices=("both", "train", "test"), default="both")
    args = parser.parse_args()

    for name in LABELS:
        download(urljoin(BASE_URL, name), DESTINATION / name)

    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        directory = f"{split}-jpg"
        for url, destination in list_shards(split, directory):
            download(url, destination)


if __name__ == "__main__":
    main()
