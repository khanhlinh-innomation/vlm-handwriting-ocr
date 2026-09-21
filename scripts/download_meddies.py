#!/usr/bin/env python3
"""Download the pinned MeddiesOCR revision outside Git."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_handwriting.meddies import MEDDIES_REPO_ID, MEDDIES_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("Install the data extra before downloading MeddiesOCR") from error

    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    result = snapshot_download(
        repo_id=MEDDIES_REPO_ID,
        repo_type="dataset",
        revision=MEDDIES_REVISION,
        local_dir=output_dir,
        max_workers=args.max_workers,
    )
    print(f"Downloaded {MEDDIES_REPO_ID}@{MEDDIES_REVISION} to {result}")


if __name__ == "__main__":
    main()
