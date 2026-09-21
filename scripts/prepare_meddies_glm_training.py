#!/usr/bin/env python3
"""Prepare MeddiesOCR page data for continued GLM-OCR LoRA training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from vlm_handwriting.artifacts import write_csv, write_json
from vlm_handwriting.llamafactory_data import dataset_entry, to_sharegpt
from vlm_handwriting.meddies import (
    SPLIT_NAMES,
    dataset_summary,
    extract_zip_safely,
    load_annotations,
    split_by_document,
    token_length_summary,
    validate_rows,
)

SEED = 42
TRAIN_SMOKE_SIZE = 16
VALIDATION_SMOKE_SIZE = 4
MANIFEST_FIELDS = [
    "id",
    "relative_path",
    "text",
    "doc_id",
    "page_number",
    "total_pages",
    "source_url",
    "processor",
    "source_line",
    "width",
    "height",
    "split",
]
REJECTION_FIELDS = ["source_line", "image_path", "doc_id", "reason"]


def select_smoke_rows(
    rows: list[dict[str, Any]], *, size: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic dependency-free smoke subset."""
    indices = sorted(random.Random(seed).sample(range(len(rows)), size))
    return [rows[index] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Use an already extracted image root instead of output-dir/meddies_images",
    )
    parser.add_argument("--image-prefix", default="meddies_images")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--skip-image-verify", action="store_true")
    return parser.parse_args()


def source_revision(dataset_root: Path) -> str | None:
    """Read the Hugging Face commit recorded by a local-dir download."""
    metadata_root = dataset_root / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir():
        return None
    revisions: set[str] = set()
    for path in metadata_root.rglob("*.metadata"):
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError, UnicodeError):
            continue
        if len(first_line) == 40 and all(char in "0123456789abcdef" for char in first_line.lower()):
            revisions.add(first_line)
    if len(revisions) > 1:
        raise RuntimeError(f"Download contains multiple source revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = dataset_root / "annotations.jsonl"
    archive = dataset_root / "images.zip"
    if not annotations.is_file() or not archive.is_file():
        raise FileNotFoundError(
            f"Expected annotations.jsonl and images.zip under {dataset_root}"
        )

    if args.image_root is None:
        image_root = output_dir / "meddies_images"
        extraction = extract_zip_safely(archive, image_root)
    else:
        image_root = args.image_root.expanduser().resolve()
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing extracted image root: {image_root}")
        extraction = {"external_image_root": str(image_root)}
    source_rows = load_annotations(annotations)
    accepted, rejected = validate_rows(
        source_rows,
        image_root=image_root,
        check_images=not args.skip_image_verify,
        progress_every=1000,
    )
    splits = split_by_document(
        accepted,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    manifest_dir = output_dir / "manifests"
    for name in SPLIT_NAMES:
        write_csv(manifest_dir / f"meddies_{name}.csv", splits[name], MANIFEST_FIELDS)
    write_csv(manifest_dir / "meddies_rejected.csv", rejected, REJECTION_FIELDS)

    dataset_info: dict[str, object] = {}
    for name in SPLIT_NAMES:
        dataset_name = f"meddies_{name}"
        file_name = f"{dataset_name}.json"
        write_json(
            output_dir / file_name,
            to_sharegpt(splits[name], image_prefix=args.image_prefix),
        )
        dataset_info[dataset_name] = dataset_entry(file_name)

    smoke_train = select_smoke_rows(
        splits["train"], size=min(TRAIN_SMOKE_SIZE, len(splits["train"])), seed=args.seed
    )
    smoke_validation = select_smoke_rows(
        splits["validation"],
        size=min(VALIDATION_SMOKE_SIZE, len(splits["validation"])),
        seed=args.seed,
    )
    for dataset_name, file_name, rows in (
        ("meddies_train_smoke", "meddies_train_smoke.json", smoke_train),
        ("meddies_validation_smoke", "meddies_validation_smoke.json", smoke_validation),
    ):
        write_json(output_dir / file_name, to_sharegpt(rows, image_prefix=args.image_prefix))
        dataset_info[dataset_name] = dataset_entry(file_name)
    write_json(output_dir / "dataset_info.json", dataset_info)

    summary: dict[str, Any] = dataset_summary(splits, rejected)
    if args.tokenizer_json is not None:
        summary["target_tokens"] = token_length_summary(
            [str(row["text"]) for name in SPLIT_NAMES for row in splits[name]],
            args.tokenizer_json,
        )
    summary.update(
        {
            "dataset": "Leo1903/meddiesOCR",
            "source_revision": source_revision(dataset_root),
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "image_root": str(image_root),
            "image_prefix": args.image_prefix,
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": 1 - args.train_ratio - args.validation_ratio,
            },
            "image_verification": not args.skip_image_verify,
            "extraction": extraction,
            "source_files": {
                "annotations.jsonl": {
                    "bytes": annotations.stat().st_size,
                    "sha256": file_sha256(annotations),
                },
                "images.zip": {
                    "bytes": archive.stat().st_size,
                    "sha256": file_sha256(archive),
                },
            },
            "prompt": "<image>Text Recognition:",
            "manual_audit": False,
        }
    )
    write_json(output_dir / "preparation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
