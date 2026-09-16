#!/usr/bin/env python3
"""Prepare frozen UIT-HWDB-line data for GLM-OCR LLaMA-Factory training."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_handwriting.artifacts import write_json
from vlm_handwriting.data import find_manifest_root, load_manifest, validate_frozen_manifests
from vlm_handwriting.llamafactory_data import dataset_entry, to_sharegpt
from vlm_handwriting.smoke import select_smoke_rows, smoke_indices

TRAIN_SMOKE_SIZE = 32
VAL_SMOKE_SIZE = 8
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def ensure_image_link(output_dir: Path, raw_root: Path) -> Path:
    image_link = output_dir / "images"
    expected = raw_root.expanduser().resolve()
    if image_link.is_symlink():
        if image_link.resolve() != expected:
            raise RuntimeError(
                f"Existing image symlink points to {image_link.resolve()}, not {expected}"
            )
    elif image_link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {image_link}")
    else:
        image_link.symlink_to(expected, target_is_directory=True)
    return image_link


def main() -> None:
    args = parse_args()
    manifest_root = find_manifest_root(args.manifest_root)
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = validate_frozen_manifests(
        manifest_root,
        raw_root=raw_root,
        check_images=True,
    )
    ensure_image_link(output_dir, raw_root)

    train_rows = load_manifest(manifest_root / "train.csv")
    validation_rows = load_manifest(manifest_root / "val.csv")
    smoke_train = select_smoke_rows(train_rows, size=TRAIN_SMOKE_SIZE, seed=SEED)
    smoke_validation = select_smoke_rows(validation_rows, size=VAL_SMOKE_SIZE, seed=SEED)

    files = {
        "uit_hwdb_line_train": ("train.json", train_rows),
        "uit_hwdb_line_validation": ("validation.json", validation_rows),
        "uit_hwdb_line_train_smoke": ("smoke_train.json", smoke_train),
        "uit_hwdb_line_validation_smoke": ("smoke_validation.json", smoke_validation),
    }
    dataset_info: dict[str, object] = {}
    for name, (file_name, rows) in files.items():
        write_json(output_dir / file_name, to_sharegpt(rows))
        dataset_info[name] = dataset_entry(file_name)
    write_json(output_dir / "dataset_info.json", dataset_info)

    summary = {
        "contract": contract,
        "manifest_root": str(manifest_root),
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "smoke_train_rows": len(smoke_train),
        "smoke_validation_rows": len(smoke_validation),
        "smoke_train_indices": smoke_indices(len(train_rows), TRAIN_SMOKE_SIZE, SEED),
        "smoke_validation_indices": smoke_indices(
            len(validation_rows), VAL_SMOKE_SIZE, SEED
        ),
        "seed": SEED,
        "prompt": "<image>Text Recognition:",
    }
    write_json(output_dir / "preparation_summary.json", summary)
    print(f"Prepared LLaMA-Factory data: {output_dir}")
    print(f"Train={len(train_rows)} Validation={len(validation_rows)}")
    print(f"Smoke train={len(smoke_train)} Smoke validation={len(smoke_validation)}")


if __name__ == "__main__":
    main()
