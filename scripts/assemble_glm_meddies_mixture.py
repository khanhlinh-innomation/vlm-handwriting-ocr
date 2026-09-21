#!/usr/bin/env python3
"""Add the frozen UIT-HWDB replay datasets to prepared MeddiesOCR data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from vlm_handwriting.artifacts import write_json

HANDWRITING_DATASETS = (
    "uit_hwdb_line_train",
    "uit_hwdb_line_validation",
    "uit_hwdb_line_train_smoke",
    "uit_hwdb_line_validation_smoke",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meddies-dir", type=Path, required=True)
    parser.add_argument("--handwriting-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_image_link(meddies_dir: Path, handwriting_dir: Path) -> Path:
    source = handwriting_dir / "images"
    if not source.exists():
        raise FileNotFoundError(f"Missing handwriting image path: {source}")
    source = source.resolve()
    destination = meddies_dir / "images"
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Existing {destination} points to {destination.resolve()}")
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace existing non-symlink path: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=True)
    return destination


def main() -> None:
    args = parse_args()
    meddies_dir = args.meddies_dir.expanduser().resolve()
    handwriting_dir = args.handwriting_dir.expanduser().resolve()
    meddies_info_path = meddies_dir / "dataset_info.json"
    handwriting_info_path = handwriting_dir / "dataset_info.json"
    meddies_info = load_json(meddies_info_path)
    handwriting_info = load_json(handwriting_info_path)
    ensure_image_link(meddies_dir, handwriting_dir)

    copied: dict[str, str] = {}
    for dataset_name in HANDWRITING_DATASETS:
        if dataset_name not in handwriting_info:
            raise KeyError(f"Missing frozen handwriting dataset entry: {dataset_name}")
        entry = dict(handwriting_info[dataset_name])
        source_name = str(entry["file_name"])
        source = handwriting_dir / source_name
        target_name = f"uit_hwdb_{source_name}"
        target = meddies_dir / target_name
        shutil.copy2(source, target)
        entry["file_name"] = target_name
        meddies_info[dataset_name] = entry
        copied[dataset_name] = target_name

    if any("test" in name.lower() for name in copied):
        raise AssertionError("Frozen handwriting test data must never enter training exports")
    write_json(meddies_info_path, meddies_info)
    write_json(
        meddies_dir / "mixture_assembly_summary.json",
        {
            "meddies_dir": str(meddies_dir),
            "handwriting_dir": str(handwriting_dir),
            "copied_datasets": copied,
            "handwriting_images": str((meddies_dir / "images").resolve()),
            "meddies_images": str((meddies_dir / "meddies_images").resolve()),
            "train_mixture": "meddies_train,uit_hwdb_line_train",
            "mix_strategy": "interleave_over",
            "interleave_probs": [0.8, 0.2],
        },
    )
    print(f"Assembled GLM two-domain dataset under {meddies_dir}")


if __name__ == "__main__":
    main()
