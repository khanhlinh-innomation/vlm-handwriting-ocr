#!/usr/bin/env python3
"""Prepare frozen UIT-HWDB-line data for PaddleOCR-VL ERNIEKit SFT."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_handwriting.artifacts import write_json
from vlm_handwriting.data import find_manifest_root, load_manifest, validate_frozen_manifests
from vlm_handwriting.paddle_training import to_erniekit_record, write_jsonl
from vlm_handwriting.smoke import select_smoke_rows, smoke_indices

TRAIN_SMOKE_SIZE = 128
VAL_SMOKE_SIZE = 32
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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
    train_rows = load_manifest(manifest_root / "train.csv")
    validation_rows = load_manifest(manifest_root / "val.csv")
    smoke_train = select_smoke_rows(train_rows, size=TRAIN_SMOKE_SIZE, seed=SEED)
    smoke_validation = select_smoke_rows(validation_rows, size=VAL_SMOKE_SIZE, seed=SEED)

    datasets = {
        "train.jsonl": train_rows,
        "validation.jsonl": validation_rows,
        "smoke_train.jsonl": smoke_train,
        "smoke_validation.jsonl": smoke_validation,
    }
    counts = {}
    for file_name, rows in datasets.items():
        records = (to_erniekit_record(row, raw_root) for row in rows)
        counts[file_name] = write_jsonl(output_dir / file_name, records)

    summary = {
        "contract": contract,
        "format": "erniekit_jsonl",
        "manifest_root": str(manifest_root),
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "files": counts,
        "smoke_train_indices": smoke_indices(len(train_rows), TRAIN_SMOKE_SIZE, SEED),
        "smoke_validation_indices": smoke_indices(
            len(validation_rows), VAL_SMOKE_SIZE, SEED
        ),
        "seed": SEED,
        "prompt": "OCR:",
        "test_rows_exported": 0,
    }
    write_json(output_dir / "preparation_summary.json", summary)

    print(f"Prepared ERNIEKit data: {output_dir}")
    print(f"Train={len(train_rows)} Validation={len(validation_rows)} Test exported=0")
    print(f"Smoke train={len(smoke_train)} Smoke validation={len(smoke_validation)}")


if __name__ == "__main__":
    main()
