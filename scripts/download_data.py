#!/usr/bin/env python3
"""Download raw and frozen-manifest Kaggle datasets into an external cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vlm_handwriting.data import find_manifest_root, find_raw_root, load_manifest

RAW_HANDLE = "ntklinhfitus/uit-hwdb"
MANIFEST_HANDLE = "ntklinhfitus/hwdb-manifest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    args = parser.parse_args()
    cache_root = args.data_root.expanduser().resolve() / "kagglehub"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_root))

    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit('Install the data extra first: uv pip install -e ".[data]"') from error

    try:
        raw_path = Path(kagglehub.dataset_download(RAW_HANDLE, path="UIT_HWDB_line"))
    except Exception:
        raw_path = Path(kagglehub.dataset_download(RAW_HANDLE))
    manifest_path = Path(kagglehub.dataset_download(MANIFEST_HANDLE))
    manifest_root = find_manifest_root(manifest_path)
    first_train_row = load_manifest(manifest_root / "train.csv")[0]
    raw_root = find_raw_root(raw_path, first_train_row["relative_path"])
    print(
        json.dumps(
            {
                "raw_download": str(raw_path),
                "manifest_download": str(manifest_path),
                "raw_root": str(raw_root),
                "manifest_root": str(manifest_root),
                "kagglehub_cache": str(cache_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
